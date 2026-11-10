"""Participant-visible episodic JSONL reader and tensor preparation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from transformers import PreTrainedTokenizerBase


class DatasetContractError(ValueError):
    """Raised when an episodic file violates the public schema."""


@dataclass(frozen=True)
class Example:
    item_id: str
    text: str
    label: int | None


@dataclass(frozen=True)
class Episode:
    episode_id: str
    support: tuple[Example, ...]
    queries: tuple[Example, ...]


@dataclass
class PreparedEpisodes:
    """Dense tensors used by both participant and author training loops."""

    episodes: list[Episode]
    support_input_ids: torch.Tensor
    support_attention_mask: torch.Tensor
    support_token_type_ids: torch.Tensor | None
    support_labels: torch.Tensor
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
    query_token_type_ids: torch.Tensor | None
    query_labels: torch.Tensor | None

    def to(self, device: str | torch.device) -> PreparedEpisodes:
        for name in (
            "support_input_ids",
            "support_attention_mask",
            "support_labels",
            "query_input_ids",
            "query_attention_mask",
        ):
            setattr(self, name, getattr(self, name).to(device))
        if self.support_token_type_ids is not None:
            self.support_token_type_ids = self.support_token_type_ids.to(device)
        if self.query_token_type_ids is not None:
            self.query_token_type_ids = self.query_token_type_ids.to(device)
        if self.query_labels is not None:
            self.query_labels = self.query_labels.to(device)
        return self


@dataclass
class PreparedQueries:
    """Query-only tensors for the lightweight supplied baseline."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    labels: torch.Tensor | None


def _parse_label(value: Any, *, required: bool, location: str) -> int | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise DatasetContractError(f"{location} must have a binary integer label")
    return value


def _parse_example(
    raw: Any,
    *,
    id_field: str,
    require_label: bool,
    location: str,
) -> Example:
    if not isinstance(raw, dict):
        raise DatasetContractError(f"{location} must be an object")
    allowed = {id_field, "text", "label"}
    if set(raw) - allowed:
        raise DatasetContractError(f"{location} contains unsupported fields")
    item_id = raw.get(id_field)
    text = raw.get("text")
    if not isinstance(item_id, str) or not item_id:
        raise DatasetContractError(f"{location}.{id_field} must be a non-empty string")
    if not isinstance(text, str) or not text or "\n" in text or "\r" in text:
        raise DatasetContractError(f"{location}.text must be one non-empty line")
    label = _parse_label(raw.get("label"), required=require_label, location=location)
    return Example(item_id=item_id, text=text, label=label)


def parse_episode(raw: Any, *, require_query_labels: bool | None = None) -> Episode:
    if not isinstance(raw, dict) or set(raw) != {"episode_id", "support", "queries"}:
        raise DatasetContractError("episode must contain exactly episode_id, support, queries")
    episode_id = raw["episode_id"]
    if not isinstance(episode_id, str) or not episode_id:
        raise DatasetContractError("episode_id must be a non-empty string")
    if not isinstance(raw["support"], list) or len(raw["support"]) != 8:
        raise DatasetContractError("each episode must contain exactly eight support examples")
    if not isinstance(raw["queries"], list) or len(raw["queries"]) != 4:
        raise DatasetContractError("each episode must contain exactly four queries")

    support = tuple(
        _parse_example(
            item,
            id_field="example_id",
            require_label=True,
            location=f"support[{index}]",
        )
        for index, item in enumerate(raw["support"])
    )
    if sum(example.label == 1 for example in support) != 4:
        raise DatasetContractError("support labels must be balanced four/four")

    if require_query_labels is None:
        present = ["label" in item for item in raw["queries"] if isinstance(item, dict)]
        if len(present) != 4 or len(set(present)) != 1:
            raise DatasetContractError("query labels must be present for all queries or none")
        require_query_labels = present[0]
    queries = tuple(
        _parse_example(
            item,
            id_field="query_id",
            require_label=require_query_labels,
            location=f"queries[{index}]",
        )
        for index, item in enumerate(raw["queries"])
    )
    ids = [example.item_id for example in support + queries]
    if len(ids) != len(set(ids)):
        raise DatasetContractError("example/query IDs must be unique within an episode")
    return Episode(episode_id=episode_id, support=support, queries=queries)


def read_episodes(
    path: str | Path,
    *,
    require_query_labels: bool | None = None,
) -> list[Episode]:
    episodes: list[Episode] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            try:
                raw = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise DatasetContractError(f"invalid JSON on line {line_number}") from error
            episode = parse_episode(raw, require_query_labels=require_query_labels)
            if episode.episode_id in seen_ids:
                raise DatasetContractError("episode IDs must be unique")
            seen_ids.add(episode.episode_id)
            episodes.append(episode)
    if not episodes:
        raise DatasetContractError("dataset is empty")
    return episodes


def _tokenize_grid(
    tokenizer: PreTrainedTokenizerBase,
    rows: Sequence[Sequence[str]],
    *,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    row_count = len(rows)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise DatasetContractError("text grid is ragged")
    flat = [text for row in rows for text in row]
    encoded = tokenizer(
        flat,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    shape = (row_count, width, max_length)
    input_ids = encoded["input_ids"].reshape(shape)
    attention_mask = encoded["attention_mask"].reshape(shape)
    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.reshape(shape)
    return input_ids, attention_mask, token_type_ids


def prepare_episodes(
    episodes: Sequence[Episode],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 58,
) -> PreparedEpisodes:
    """Tokenize the 8-by-4 episode grids once with no worker processes."""

    support_ids, support_mask, support_types = _tokenize_grid(
        tokenizer,
        [[example.text for example in episode.support] for episode in episodes],
        max_length=max_length,
    )
    query_ids, query_mask, query_types = _tokenize_grid(
        tokenizer,
        [[example.text for example in episode.queries] for episode in episodes],
        max_length=max_length,
    )
    support_labels = torch.tensor(
        [[example.label for example in episode.support] for episode in episodes],
        dtype=torch.long,
    )
    has_query_labels = all(
        example.label is not None for episode in episodes for example in episode.queries
    )
    query_labels = (
        torch.tensor(
            [[example.label for example in episode.queries] for episode in episodes],
            dtype=torch.float32,
        )
        if has_query_labels
        else None
    )
    return PreparedEpisodes(
        episodes=list(episodes),
        support_input_ids=support_ids,
        support_attention_mask=support_mask,
        support_token_type_ids=support_types,
        support_labels=support_labels,
        query_input_ids=query_ids,
        query_attention_mask=query_mask,
        query_token_type_ids=query_types,
        query_labels=query_labels,
    )


def prepare_queries(
    episodes: Sequence[Episode],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 58,
) -> PreparedQueries:
    """Tokenize only queries, avoiding unused support work in the starter."""

    query_ids, query_mask, query_types = _tokenize_grid(
        tokenizer,
        [[example.text for example in episode.queries] for episode in episodes],
        max_length=max_length,
    )
    has_labels = all(
        example.label is not None
        for episode in episodes
        for example in episode.queries
    )
    labels = (
        torch.tensor(
            [[example.label for example in episode.queries] for episode in episodes],
            dtype=torch.float32,
        )
        if has_labels
        else None
    )
    return PreparedQueries(
        input_ids=query_ids,
        attention_mask=query_mask,
        token_type_ids=query_types,
        labels=labels,
    )


def iter_episode_batches(
    count: int,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    if count <= 0 or batch_size <= 0:
        raise ValueError("count and batch_size must be positive")
    order = (
        torch.randperm(count, generator=generator)
        if shuffle
        else torch.arange(count)
    )
    for start in range(0, count, batch_size):
        yield order[start : start + batch_size]


def prediction_rows(
    episodes: Sequence[Episode],
    probabilities: torch.Tensor,
) -> Iterable[dict[str, str | int | float]]:
    if tuple(probabilities.shape) != (len(episodes), 4):
        raise ValueError("probabilities must have shape [episodes, 4]")
    for episode_index, episode in enumerate(episodes):
        for query_index, query in enumerate(episode.queries):
            probability = float(probabilities[episode_index, query_index])
            yield {
                "episode_id": episode.episode_id,
                "query_id": query.item_id,
                "label": int(probability >= 0.5),
                "probability": probability,
            }


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def episode_texts(episodes: Sequence[Episode]) -> list[str]:
    """Return every text of every episode, support first, in episode order."""

    texts: list[str] = []
    for episode in episodes:
        texts.extend(example.text for example in episode.support)
        texts.extend(example.text for example in episode.queries)
    return texts


def encode_unique_texts(
    backbone: Any,
    texts: Sequence[str],
    *,
    max_length: int = 58,
    batch_size: int = 64,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Encode every distinct text exactly once with the frozen backbone.

    Episodic corpora reuse phrasings heavily and messages are far shorter than
    the 58-token limit, so deduplicating and padding to the longest item of a
    length-sorted batch cuts the encoder cost by an order of magnitude while
    producing exactly the same CLS vectors.
    """

    unique = sorted(set(texts))
    row_of_text = {text: index for index, text in enumerate(unique)}
    tokenizer = backbone.tokenizer
    encoded = tokenizer(
        unique,
        padding=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    order = sorted(range(len(unique)), key=lambda index: len(encoded[index]))
    features: torch.Tensor | None = None
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        batch = tokenizer(
            [unique[index] for index in rows],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        vectors = backbone.encode_support(
            batch["input_ids"],
            batch.get("attention_mask"),
            batch.get("token_type_ids"),
        ).cpu()
        if features is None:
            features = torch.zeros(len(unique), vectors.shape[-1], dtype=vectors.dtype)
        features[torch.tensor(rows)] = vectors
    if features is None:
        raise DatasetContractError("no texts to encode")
    return features, row_of_text


def episode_index(
    episodes: Sequence[Episode],
    row_of_text: Mapping[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Map episodes onto rows of the encoded feature table."""

    support = torch.tensor(
        [[row_of_text[example.text] for example in episode.support] for episode in episodes],
        dtype=torch.long,
    )
    queries = torch.tensor(
        [[row_of_text[example.text] for example in episode.queries] for episode in episodes],
        dtype=torch.long,
    )
    support_labels = torch.tensor(
        [[example.label for example in episode.support] for episode in episodes],
        dtype=torch.float32,
    )
    has_query_labels = all(
        example.label is not None for episode in episodes for example in episode.queries
    )
    query_labels = (
        torch.tensor(
            [[example.label for example in episode.queries] for episode in episodes],
            dtype=torch.float32,
        )
        if has_query_labels
        else None
    )
    return support, queries, support_labels, query_labels
