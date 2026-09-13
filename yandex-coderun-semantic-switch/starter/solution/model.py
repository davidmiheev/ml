"""Episodic adapter: cosine nearest-centroid plus a mined rule codebook.

The frozen backbone is used only as a sentence encoder.  Everything that is
learned lives in this module and is small enough to fit the adapter budget:

``feature_mean`` / ``feature_scale``
    global standardisation statistics of the frozen CLS features;
``codebook``
    rule directions mined from the training episodes -- an episode whose
    eight support examples point at a known rule can borrow the much cleaner
    direction estimated from hundreds of training episodes;
``attn_temp`` / ``null_logit`` / ``blend_cap``
    the self-calibrating gate that decides, per episode, how much of that
    codebook direction to trust;
``log_scale`` / ``bias`` / ``magnitude_power``
    logit calibration;
``log_count_prior``
    distribution of the number of positive queries per episode, used by the
    joint decoder that maximises the episode-exact-match metric.

Nothing here ever touches the backbone weights, and prediction needs no
gradients.
"""

from __future__ import annotations

import torch
from torch import nn


def unit(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2-normalise along *dim* with a numerically safe denominator."""

    return tensor / tensor.norm(dim=dim, keepdim=True).clamp_min(1e-6)


class ParticipantModule(nn.Module):
    """Query-only head kept for the public technical contract.

    The competitive path is :class:`EpisodicAdapter`; this class only preserves
    the starter signature ``[batch, queries, hidden] -> [batch, queries]`` that
    ``starter/tests/test_contract.py`` exercises.
    """

    def __init__(self, hidden_size: int = 312) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        if query_embeddings.ndim != 3:
            raise ValueError("query_embeddings must have shape [batch, queries, hidden]")
        return self.classifier(query_embeddings).squeeze(-1)


class EpisodicAdapter(nn.Module):
    """Nearest-centroid episodic classifier with optional rule retrieval.

    The frozen embedding space is deliberately left undistorted.  A learned
    projection does lift the score on episodes whose rule also occurs in the
    training corpus, but measurement on held-out *rules* shows that the lift is
    memorisation of those rules and that it costs more than it gains as soon as
    a rule is new -- and, unlike retrieval, that failure cannot be simulated at
    training time, so it cannot be guarded against.  Retrieval carries the same
    knowledge in a form that can be switched off per episode.
    """

    def __init__(
        self,
        hidden_size: int = 312,
        *,
        codebook_size: int = 0,
        center: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.center = bool(center)
        self.register_buffer("feature_mean", torch.zeros(hidden_size))
        self.register_buffer("feature_scale", torch.ones(hidden_size))
        self.log_scale = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.zeros(1))
        self.magnitude_power = nn.Parameter(torch.tensor(0.0))
        if codebook_size:
            self.register_buffer("codebook", torch.zeros(codebook_size, hidden_size))
            self.attn_temp = nn.Parameter(torch.tensor(12.0))
            self.null_logit = nn.Parameter(torch.tensor(6.0))
            self.blend_cap = nn.Parameter(torch.tensor(1.5))
        self.register_buffer("log_count_prior", torch.full((5,), -1.6094379))
        # log-odds of the marginal positive rate of a query, used to keep the
        # joint decoder from counting the corpus base rate twice
        self.register_buffer("base_logit", torch.zeros(1))

    # -- feature space -----------------------------------------------------
    @property
    def has_codebook(self) -> bool:
        return hasattr(self, "codebook") and self.codebook.numel() > 0

    def standardize(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean) / self.feature_scale

    def episode_view(
        self, support_features: torch.Tensor, query_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardise, centre the episode and L2-normalise both grids."""

        xs = self.standardize(support_features)
        xq = self.standardize(query_features)
        if self.center:
            centre = torch.cat([xs, xq], dim=-2).mean(dim=-2, keepdim=True)
            xs, xq = xs - centre, xq - centre
        return unit(xs), unit(xq)

    # -- episodic scoring --------------------------------------------------
    def rule_direction(
        self, projected: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the centroid-difference direction and its magnitude."""

        weights = labels.unsqueeze(-1)
        positive = (projected * weights).sum(-2) / weights.sum(-2).clamp_min(1e-6)
        negative = (projected * (1 - weights)).sum(-2) / (1 - weights).sum(-2).clamp_min(1e-6)
        delta = unit(positive) - unit(negative)
        magnitude = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return delta / magnitude, magnitude

    def retrieve(
        self, direction: torch.Tensor, rule_dropout: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend the episode direction with the closest mined rule directions.

        The codebook competes with a learnable ``null_logit``: when no stored
        rule resembles the episode, the null slot wins and the episode keeps
        its own eight-shot estimate.

        ``rule_dropout`` hides the best matching entry together with its whole
        neighbourhood -- every entry closer to it than that cosine threshold --
        which simulates an episode whose rule family was never mined.  A single
        rule spreads over many codebook entries, so hiding only the top match
        would leave its siblings behind and make the simulation far too kind.
        It is used while calibrating the gate and while scoring candidates,
        never at prediction time.
        """

        similarity = direction @ self.codebook.t()
        if rule_dropout is not None:
            best = similarity.argmax(dim=-1)
            neighbourhood = self.codebook @ self.codebook[best].transpose(-1, -2)
            neighbourhood = neighbourhood.transpose(-1, -2)
            similarity = similarity.masked_fill(neighbourhood >= rule_dropout, -1.0e4)
        logits = self.attn_temp.clamp(0.5, 60.0) * similarity
        null = self.null_logit.expand(*logits.shape[:-1], 1)
        weights = torch.softmax(torch.cat([logits, null], dim=-1), dim=-1)
        gate = (1.0 - weights[..., -1:]) * torch.sigmoid(self.blend_cap)
        retrieved = unit(weights[..., :-1] @ self.codebook)
        return unit(gate * retrieved + (1.0 - gate) * direction), gate.squeeze(-1)

    def episode_logits(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        *,
        rule_dropout: float | None = None,
    ) -> torch.Tensor:
        """Return one logit per query."""

        xs, xq = self.episode_view(support_features, query_features)
        zs, zq = xs, xq
        direction, magnitude = self.rule_direction(zs, support_labels)
        if self.has_codebook:
            direction, _ = self.retrieve(direction, rule_dropout)
        weight = magnitude ** torch.sigmoid(self.magnitude_power)
        scores = torch.einsum("...qd,...d->...q", zq, direction) * weight
        return self.log_scale.exp() * scores + self.bias

    def support_loo_logits(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
    ) -> torch.Tensor:
        """Leave-one-out logits for the eight support items (extra supervision)."""

        xs, _ = self.episode_view(support_features, query_features)
        zs = xs
        weights = support_labels.unsqueeze(-1)
        n_pos = weights.sum(-2, keepdim=True)
        n_neg = (1 - weights).sum(-2, keepdim=True)
        pos = ((zs * weights).sum(-2, keepdim=True) - zs * weights) / (n_pos - weights).clamp_min(1e-6)
        neg = ((zs * (1 - weights)).sum(-2, keepdim=True) - zs * (1 - weights)) / (
            n_neg - (1 - weights)
        ).clamp_min(1e-6)
        delta = unit(pos) - unit(neg)
        magnitude = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        weight = magnitude.squeeze(-1) ** torch.sigmoid(self.magnitude_power)
        scores = (zs * (delta / magnitude)).sum(-1) * weight
        return self.log_scale.exp() * scores + self.bias


def _assignments(queries: int, device=None) -> torch.Tensor:
    """All 2**queries binary label assignments as a float matrix."""

    table = torch.arange(2 ** queries, device=device).unsqueeze(1)
    bits = torch.arange(queries, device=device).unsqueeze(0)
    return ((table >> bits) & 1).float()


def decode_episode(
    logits: torch.Tensor,
    log_count_prior: torch.Tensor | None = None,
    base_logit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint MAP decoding of one episode's queries.

    Episode exact match rewards the most likely *joint* assignment rather than
    four independent decisions, so the learned distribution of the number of
    positive queries is folded in.  Returned probabilities always agree with
    the returned labels (``label == 1`` iff ``probability >= 0.5``).
    """

    probabilities = torch.sigmoid(logits)
    if log_count_prior is None:
        return (logits >= 0).float(), probabilities

    n_queries = logits.shape[-1]
    table = _assignments(n_queries, logits.device).to(logits.dtype)
    counts = table.sum(dim=1)
    log_choose = (
        torch.lgamma(torch.tensor(float(n_queries + 1), dtype=logits.dtype))
        - torch.lgamma(counts + 1)
        - torch.lgamma(float(n_queries) - counts + 1)
    )
    prior = log_count_prior.to(logits.dtype)[counts.long()] - log_choose
    joint = (logits - base_logit) @ table.t() + prior
    labels = table[joint.argmax(dim=-1)]
    marginal = torch.softmax(joint, dim=-1) @ table
    probabilities = torch.where(
        labels > 0.5,
        marginal.clamp(0.5, 1.0 - 1e-6),
        marginal.clamp(1e-6, 0.5 - 1e-6),
    )
    return labels, probabilities


def adapter_config(adapter: "EpisodicAdapter") -> torch.Tensor:
    """Serialise the structural options that are not implied by tensor shapes."""

    return torch.tensor(
        [int(adapter.center), int(adapter.has_codebook)], dtype=torch.int64
    )


def build_from_state(state) -> "EpisodicAdapter":
    """Rebuild an adapter from a saved state mapping without extra metadata."""

    config = state.get("config")
    hidden_size = int(state["feature_mean"].shape[-1])
    center = True if config is None else bool(int(config[0]))
    codebook = state.get("codebook")
    adapter = EpisodicAdapter(
        hidden_size,
        codebook_size=0 if codebook is None else int(codebook.shape[0]),
        center=center,
    )
    adapter.load_state_dict({k: v for k, v in state.items() if k != "config"}, strict=True)
    adapter.eval()
    return adapter
