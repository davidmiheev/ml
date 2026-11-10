"""Generate predictions with the trained episodic adapter.

Prediction is strictly gradient free: the frozen backbone encodes each distinct
message once, the adapter turns the eight labelled support examples of an
episode into a decision direction, and the four queries are decoded jointly so
that the emitted labels maximise the episode-exact-match objective.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file

from starter.runtime.backbone import FrozenBackbone
from starter.runtime.meta import load_task_config, validate_artifact, verify_backbone_sha
from starter.solution.data import (
    encode_unique_texts,
    episode_index,
    episode_texts,
    prediction_rows,
    read_episodes,
    write_jsonl,
)
from starter.solution.model import build_from_state, decode_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    args = build_parser().parse_args(argv)
    task_config = load_task_config()
    torch.set_num_threads(int(task_config.get("cpu_cores", 2)))
    verify_backbone_sha()
    validate_artifact(args.adapter)

    backbone = FrozenBackbone()
    episodes = read_episodes(args.data)
    adapter = build_from_state(load_file(str(args.adapter), device="cpu"))
    adapter.requires_grad_(False)

    encode_started = time.perf_counter()
    features, row_of_text = encode_unique_texts(
        backbone,
        episode_texts(episodes),
        max_length=int(task_config.get("max_original_sequence_length", 58)),
        batch_size=min(64, max(8, args.batch_size)),
    )
    encode_seconds = time.perf_counter() - encode_started

    support, queries, support_labels, _ = episode_index(episodes, row_of_text)
    prior = adapter.log_count_prior
    chunk = max(1, min(512, args.batch_size * 16))
    probabilities = []
    inference_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(episodes), chunk):
            stop = start + chunk
            logits = adapter.episode_logits(
                features[support[start:stop]],
                support_labels[start:stop],
                features[queries[start:stop]],
            )
            probabilities.append(
                decode_episode(logits, prior, float(adapter.base_logit))[1]
            )
    inference_seconds = time.perf_counter() - inference_started

    output_probabilities = torch.cat(probabilities).clamp(0.0, 1.0)
    write_jsonl(prediction_rows(episodes, output_probabilities), args.output)
    print(
        json.dumps(
            {
                "encode_seconds": round(encode_seconds, 1),
                "episodes": len(episodes),
                "inference_seconds": round(inference_seconds, 2),
                "output": str(args.output.resolve()),
                "positive_rate": round(float((output_probabilities >= 0.5).float().mean()), 4),
                "queries": len(episodes) * 4,
                "total_seconds": round(time.perf_counter() - started, 1),
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
