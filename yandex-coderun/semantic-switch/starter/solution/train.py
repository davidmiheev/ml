"""Train the episodic adapter offline on CPU.

The backbone stays frozen and is used once, as a sentence encoder, to cache CLS
features for every distinct message.  Everything after that operates on cached
vectors, so the whole run is dominated by that single encoding pass.

Model selection is data driven.  Training episodes are split into three parts:
one to calibrate the scores and mine the rule codebook, one to calibrate the
retrieval gate, and one that no fitting ever sees and that picks the
configuration by episode exact match.  That last split is what decides whether rule retrieval
helps at all on the hidden corpus -- if the corpus never repeats a rule, the
gate closes and the adapter falls back to the plain frozen-space classifier.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn

from starter.runtime.backbone import FrozenBackbone
from starter.runtime.meta import (
    load_task_config,
    validate_artifact,
    validate_trainable_parameter_count,
)
from starter.solution.data import (
    encode_unique_texts,
    episode_index,
    episode_texts,
    read_episodes,
)
from starter.solution.model import (
    EpisodicAdapter,
    adapter_config,
    decode_episode,
    unit,
)

DEFAULT_EPOCHS = 12
MIN_EPISODE_BATCH = 32
CODEBOOK_LIMIT = 384
SELECTION_MIN_EPISODES = 240
# The selection split is finite, so a richer candidate has to win by more than
# its sampling noise before it displaces a simpler one.
SELECTION_MARGIN = 0.015
SELECTION_FOLDS = 3


class EpisodeTensors:
    """Cached feature rows and labels for a set of episodes."""

    def __init__(self, support, queries, support_labels, query_labels):
        self.support = support
        self.queries = queries
        self.support_labels = support_labels
        self.query_labels = query_labels

    def __len__(self) -> int:
        return int(self.support.shape[0])

    def select(self, rows: torch.Tensor) -> "EpisodeTensors":
        return EpisodeTensors(
            self.support[rows],
            self.queries[rows],
            self.support_labels[rows],
            None if self.query_labels is None else self.query_labels[rows],
        )


def count_prior(query_labels: torch.Tensor, smoothing: float = 1.0) -> torch.Tensor:
    """Smoothed log-distribution of the number of positive queries per episode."""

    counts = query_labels.sum(dim=1).long()
    histogram = torch.bincount(counts, minlength=5).float()[:5] + smoothing
    return (histogram / histogram.sum()).log()


NO_DROPOUT = 2.0
# Cosine radius around the best codebook match that counts as "the same rule".
# Calibrated so that the simulated score of a mined-rule episode matches the
# measured score on episodes whose rule family was genuinely never mined.
RULE_DROPOUT = 0.25


def batched_logits(
    adapter: EpisodicAdapter,
    features,
    data: EpisodeTensors,
    chunk: int = 512,
    rule_dropout: float | None = None,
):
    outputs = []
    adapter.eval()
    with torch.no_grad():
        for start in range(0, len(data), chunk):
            stop = start + chunk
            outputs.append(
                adapter.episode_logits(
                    features[data.support[start:stop]],
                    data.support_labels[start:stop],
                    features[data.queries[start:stop]],
                    rule_dropout=rule_dropout,
                )
            )
    return torch.cat(outputs) if outputs else torch.zeros(0, 4)


def exact_match(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prior: torch.Tensor | None,
    base_logit: float = 0.0,
):
    predicted, _ = decode_episode(logits, prior, base_logit)
    return (
        float((predicted == labels).all(dim=1).float().mean()),
        float((predicted == labels).float().mean()),
    )


def score_candidate(
    adapter: EpisodicAdapter,
    features: torch.Tensor,
    select_part: EpisodeTensors,
    prior: torch.Tensor,
    *,
    sweep_cap: bool,
):
    """Score a candidate on episodes no fitting stage has seen.

    Two regimes are measured: the hidden corpus reusing a mined rule, and an
    episode whose rule family was never mined.  The candidate is ranked by
    their mean, so retrieval has to earn its place under both assumptions.  For
    adapters with a codebook the ceiling on the blend is swept here too, since
    that single number is what trades one regime against the other.
    """

    base = float(adapter.base_logit)

    def measure():
        logits = batched_logits(adapter, features, select_part)
        seen, accuracy = exact_match(logits, select_part.query_labels, prior, base)
        plain = exact_match(logits, select_part.query_labels, None)[0]
        if adapter.has_codebook:
            dropped = batched_logits(
                adapter, features, select_part, rule_dropout=RULE_DROPOUT
            )
            unseen = exact_match(dropped, select_part.query_labels, prior, base)[0]
        else:
            unseen = seen
        return 0.5 * (seen + unseen), seen, accuracy, plain, unseen

    if not sweep_cap:
        return (*measure(), 1.0)

    best = None
    with torch.no_grad():
        original = adapter.blend_cap.detach().clone()
        options = [float(torch.sigmoid(original)), 0.25, 0.5, 0.75]
        for cap in options:
            adapter.blend_cap.copy_(
                torch.logit(torch.tensor(min(max(cap, 1e-4), 1.0 - 1e-4)))
            )
            measured = measure()
            if best is None or measured[0] > best[0][0]:
                best = (measured, cap)
        adapter.blend_cap.copy_(
            torch.logit(torch.tensor(min(max(best[1], 1e-4), 1.0 - 1e-4)))
        )
    return (*best[0], best[1])


def mine_codebook(
    adapter: EpisodicAdapter,
    features: torch.Tensor,
    data: EpisodeTensors,
    size: int,
    *,
    iterations: int = 20,
    seed: int = 0,
) -> torch.Tensor:
    """Cluster per-episode rule directions into a compact codebook.

    Training episodes expose the labels of all twelve messages, so each mined
    direction is a cleaner estimate than the eight-shot direction available at
    prediction time.
    """

    directions = []
    with torch.no_grad():
        for start in range(0, len(data), 512):
            stop = start + 512
            support = features[data.support[start:stop]]
            queries = features[data.queries[start:stop]]
            xs, xq = adapter.episode_view(support, queries)
            grid = torch.cat([xs, xq], dim=1)
            labels = torch.cat(
                [data.support_labels[start:stop], data.query_labels[start:stop]], dim=1
            )
            directions.append(adapter.rule_direction(grid, labels)[0])
    directions = torch.cat(directions)
    size = min(size, directions.shape[0])
    generator = torch.Generator().manual_seed(seed)
    centres = directions[torch.randperm(directions.shape[0], generator=generator)[:size]].clone()
    for _ in range(iterations):
        assignment = (directions @ centres.t()).argmax(dim=1)
        for index in range(size):
            members = assignment == index
            if bool(members.any()):
                centres[index] = unit(directions[members].mean(dim=0))
    return unit(centres)


def fit_parameters(
    adapter: EpisodicAdapter,
    features: torch.Tensor,
    data: EpisodeTensors,
    parameters: list[nn.Parameter],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    loo_weight: float = 0.5,
    rule_dropout_prob: float = 0.0,
    deadline: float | None = None,
) -> None:
    """Optimise *parameters* with the episodic binary cross-entropy objective.

    With ``rule_dropout_prob`` a random half of the episodes are scored as if
    their rule had never been mined, so the retrieval gate learns to fall back
    to the eight-shot direction instead of always trusting the codebook.
    """

    if not parameters or len(data) == 0:
        return
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    steps = max(1, epochs * math.ceil(len(data) / batch_size))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=steps
    )
    criterion = nn.BCEWithLogitsLoss()
    adapter.train()
    for _ in range(epochs):
        order = torch.randperm(len(data), generator=generator)
        for start in range(0, len(data), batch_size):
            rows = order[start : start + batch_size]
            support = features[data.support[rows]]
            queries = features[data.queries[rows]]
            labels = data.support_labels[rows]
            dropout = None
            if rule_dropout_prob > 0.0:
                drawn = torch.rand(len(rows), 1, generator=generator) < rule_dropout_prob
                dropout = torch.where(
                    drawn, torch.tensor(RULE_DROPOUT), torch.tensor(NO_DROPOUT)
                )
            logits = adapter.episode_logits(
                support, labels, queries, rule_dropout=dropout
            )
            loss = criterion(logits, data.query_labels[rows])
            if loo_weight:
                loss = loss + loo_weight * criterion(
                    adapter.support_loo_logits(support, labels, queries), labels
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
        if deadline is not None and time.perf_counter() > deadline:
            break
    adapter.eval()


def build_candidate(
    codebook_size: int,
    features: torch.Tensor,
    stats: tuple[torch.Tensor, torch.Tensor],
    fit_part: EpisodeTensors,
    gate_part: EpisodeTensors,
    *,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    deadline: float | None,
) -> EpisodicAdapter:
    adapter = EpisodicAdapter(hidden_size, codebook_size=codebook_size)
    adapter.feature_mean.copy_(stats[0])
    adapter.feature_scale.copy_(stats[1])
    calibration = [adapter.log_scale, adapter.bias, adapter.magnitude_power]
    fit_parameters(
        adapter,
        features,
        fit_part,
        calibration,
        epochs=max(2, epochs // 3),
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        deadline=deadline,
    )
    if codebook_size:
        adapter.codebook.copy_(
            mine_codebook(adapter, features, fit_part, codebook_size, seed=seed)
        )
        fit_parameters(
            adapter,
            features,
            gate_part,
            [adapter.attn_temp, adapter.null_logit, adapter.blend_cap] + calibration,
            epochs=max(4, epochs),
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + 1,
            loo_weight=0.0,
            rule_dropout_prob=0.5,
            deadline=deadline,
        )
    return adapter


def split_rows(total: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three disjoint episode splits: codebook, gate, model selection."""

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(total, generator=generator)
    if total < 3 * SELECTION_MIN_EPISODES:
        return order, order, order
    select = max(SELECTION_MIN_EPISODES, int(0.15 * total))
    gate = max(SELECTION_MIN_EPISODES, int(0.15 * total))
    return order[select + gate :], order[select : select + gate], order[:select]


def final_split(total: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit/gate split for the refit that follows model selection."""

    generator = torch.Generator().manual_seed(seed + 977)
    order = torch.randperm(total, generator=generator)
    if total < 2 * SELECTION_MIN_EPISODES:
        return order, order
    gate = max(SELECTION_MIN_EPISODES, int(0.15 * total))
    return order[gate:], order[:gate]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--time-budget", type=float, default=840.0)
    parser.add_argument(
        "--report-eval",
        type=Path,
        help="optional labelled JSONL scored for every candidate (diagnostics only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive")
    task_config = load_task_config()
    torch.set_num_threads(int(task_config.get("cpu_cores", 2)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    deadline = started + args.time_budget

    backbone = FrozenBackbone()
    episodes = read_episodes(args.train, require_query_labels=True)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    encode_started = time.perf_counter()
    features, row_of_text = encode_unique_texts(
        backbone,
        episode_texts(episodes),
        max_length=int(task_config.get("max_original_sequence_length", 58)),
        batch_size=min(64, max(8, args.batch_size)),
    )
    encode_seconds = time.perf_counter() - encode_started

    support, queries, support_labels, query_labels = episode_index(episodes, row_of_text)
    if query_labels is None:
        raise ValueError("training queries must have labels")
    data = EpisodeTensors(support, queries, support_labels, query_labels)
    stats = (
        features.mean(dim=0),
        features.std(dim=0).clamp_min(1e-5),
    )
    prior = count_prior(query_labels)
    positive_rate = float(query_labels.mean().clamp(1e-3, 1 - 1e-3))
    base_logit = math.log(positive_rate / (1.0 - positive_rate))

    probe_fit, probe_gate, _ = split_rows(len(data), args.seed)
    hidden_size = int(features.shape[1])
    scalar_budget = int(task_config.get("max_trainable_parameters", 150000))
    largest = max(0, (scalar_budget - 4 * hidden_size - 64) // hidden_size)
    # A coarse codebook averages each rule over more episodes and matches fewer
    # unrelated ones; a fine one covers a corpus with many distinct rules.  The
    # selection split decides which the hidden corpus needs.
    sizes = sorted(
        {
            min(largest, CODEBOOK_LIMIT // 6, max(8, len(probe_fit) // 44)),
            min(largest, CODEBOOK_LIMIT, max(8, len(probe_fit) // 8)),
        }
    )

    plan = [("frozen-space", 0)]
    if largest >= 8 and len(probe_gate) >= SELECTION_MIN_EPISODES:
        plan += [(f"codebook{size}", size) for size in sizes]
    epochs = max(args.epochs, DEFAULT_EPOCHS)
    batch_size = max(args.batch_size, MIN_EPISODE_BATCH)
    learning_rate = max(args.learning_rate, 3e-3)

    report_part = None
    if args.report_eval:
        report_episodes = read_episodes(args.report_eval, require_query_labels=True)
        report_features, report_rows = encode_unique_texts(
            backbone,
            episode_texts(report_episodes),
            max_length=int(task_config.get("max_original_sequence_length", 58)),
            batch_size=64,
        )
        report_part = EpisodeTensors(*episode_index(report_episodes, report_rows))

    def make(size: int, fit: EpisodeTensors, gate: EpisodeTensors, seed: int):
        adapter = build_candidate(
            size,
            features,
            stats,
            fit,
            gate,
            hidden_size=hidden_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            deadline=deadline,
        )
        adapter.log_count_prior.copy_(prior)
        adapter.base_logit.fill_(base_logit)
        return adapter

    # Selection is repeated over several independent splits: one 600-episode
    # split cannot separate candidates that sit a couple of points apart.
    folds: dict[str, list[dict[str, float]]] = {name: [] for name, _ in plan}
    completed_folds = 0
    for fold in range(SELECTION_FOLDS):
        if fold and time.perf_counter() > deadline - 0.25 * (time.perf_counter() - started):
            break
        fit_rows, gate_rows, select_rows = split_rows(len(data), args.seed + 101 * fold)
        fit_part, gate_part, select_part = (
            data.select(fit_rows),
            data.select(gate_rows),
            data.select(select_rows),
        )
        for name, size in plan:
            adapter = make(size, fit_part, gate_part, args.seed + fold)
            robust, score, accuracy, plain, unseen, cap = score_candidate(
                adapter, features, select_part, prior, sweep_cap=bool(size)
            )
            measured = {
                "robust": robust,
                "seen": score,
                "unseen": unseen,
                "accuracy": accuracy,
                "without_prior": plain,
                "blend_cap": cap,
            }
            if report_part is not None:
                held = exact_match(
                    batched_logits(adapter, report_features, report_part),
                    report_part.query_labels,
                    prior,
                    base_logit,
                )
                measured["report_exact_match"] = held[0]
                measured["report_query_accuracy"] = held[1]
            folds[name].append(measured)
        completed_folds += 1
        if time.perf_counter() > deadline:
            break

    def mean(name: str, key: str) -> float:
        values = [entry[key] for entry in folds[name]]
        return sum(values) / len(values) if values else float("-inf")

    results = []
    best = None
    for name, size in plan:
        if not folds[name]:
            continue
        entry = {
            "candidate": name,
            "codebook": size,
            "folds": len(folds[name]),
            "robust_score": round(mean(name, "robust"), 4),
            "selection_exact_match": round(mean(name, "seen"), 4),
            "selection_exact_match_unseen_rule": round(mean(name, "unseen"), 4),
            "selection_query_accuracy": round(mean(name, "accuracy"), 4),
            "selection_exact_match_without_prior": round(mean(name, "without_prior"), 4),
        }
        if size:
            entry["blend_cap"] = round(mean(name, "blend_cap"), 3)
        if report_part is not None:
            entry["report_exact_match"] = round(mean(name, "report_exact_match"), 4)
            entry["report_query_accuracy"] = round(mean(name, "report_query_accuracy"), 4)
        results.append(entry)
        if best is None or entry["robust_score"] > best[0] + SELECTION_MARGIN:
            best = (entry["robust_score"], name, size, entry.get("blend_cap", 1.0))

    # Refit the winner with the selection split folded back into the training
    # data; only the gate still needs episodes the codebook has not seen.
    _, chosen_name, chosen_size, chosen_cap = best
    fit_rows, gate_rows = final_split(len(data), args.seed)
    adapter = make(chosen_size, data.select(fit_rows), data.select(gate_rows), args.seed)
    if chosen_size:
        with torch.no_grad():
            adapter.blend_cap.copy_(
                torch.logit(torch.tensor(min(max(float(chosen_cap), 1e-4), 1.0 - 1e-4)))
            )
    if report_part is not None:
        held = exact_match(
            batched_logits(adapter, report_features, report_part),
            report_part.query_labels,
            prior,
            base_logit,
        )
        results.append({"candidate": "final refit", "report_exact_match": round(held[0], 4),
                        "report_query_accuracy": round(held[1], 4)})

    trainable = validate_trainable_parameter_count(adapter)

    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in adapter.state_dict().items()
    }
    state["config"] = adapter_config(adapter)
    args.adapter.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(args.adapter))
    artifact = validate_artifact(args.adapter)

    print(
        json.dumps(
            {
                "adapter_bytes": artifact.size_bytes,
                "adapter_scalars": artifact.scalar_count,
                "chosen": chosen_name,
                "selection_folds": completed_folds,
                "count_prior": [round(float(value), 4) for value in prior.exp()],
                "query_positive_rate": round(positive_rate, 4),
                "encode_seconds": round(encode_seconds, 1),
                "episodes": len(episodes),
                "results": results,
                "seed": args.seed,
                "total_seconds": round(time.perf_counter() - started, 1),
                "trainable_parameters": trainable,
                "unique_texts": int(features.shape[0]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
