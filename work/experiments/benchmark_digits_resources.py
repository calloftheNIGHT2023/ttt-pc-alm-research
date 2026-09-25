"""Synchronized end-to-end timing replay for the locked digits methods.

This script never uses query labels. It replays validation-split episodes after
the confirmation decision, solely to obtain honest CUDA wall times and peaks.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from digits_domain_shift import (
    CONFIRMATION_SEED_OFFSET,
    SelectedConfig,
    bank_features,
    batched_features_and_logits,
    episode_tensors,
    fit_internal,
    fit_shallow_head,
    fixed_geometric_bank,
    internal_probabilities,
    load_model_and_data,
    probabilities,
    rbf_probabilities,
    ridge_probabilities,
    tangent_jacobian,
    tangent_raw_solution,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, function) -> tuple[float, int]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    function()
    synchronize(device)
    elapsed = time.perf_counter() - start
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return elapsed, peak


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    payload = json.loads(args.selected_config.read_text(encoding="utf-8"))
    config = SelectedConfig(**payload["selected"])
    model, images, labels, splits = load_model_and_data(args.checkpoint_dir, device)
    bank = fixed_geometric_bank(config.geometric_bank_size, device)
    rows: list[dict] = []

    # One unrecorded warm-up removes kernel initialization from the comparison.
    warmup = episode_tensors(images, labels, splits["validation"], 4_300_000, device)
    _ = model(warmup["query_images"])
    synchronize(device)

    for episode_index in range(args.episodes):
        seed = 4_300_001 + episode_index
        episode = episode_tensors(images, labels, splits["validation"], seed, device)
        adapt_images = episode["adapt_images"]
        adapt_labels = episode["adapt_labels"]
        context_images = episode["context_images"]
        context_labels = episode["context_labels"]
        query_images = episode["query_images"]

        def no_adaptation() -> None:
            with torch.no_grad():
                _ = probabilities(model(query_images))

        def ridge() -> None:
            context_features, _ = batched_features_and_logits(model, context_images)
            query_features, _ = batched_features_and_logits(model, query_images)
            _ = ridge_probabilities(context_features, context_labels, query_features, config.ridge_alpha)

        def rbf() -> None:
            context_features, _ = batched_features_and_logits(model, context_images)
            query_features, _ = batched_features_and_logits(model, query_images)
            _ = rbf_probabilities(context_features, context_labels, query_features, config.rbf_alpha, config.rbf_gamma_multiplier)

        def shallow() -> None:
            context_features, context_logits = batched_features_and_logits(model, context_images)
            query_features, query_logits = batched_features_and_logits(model, query_images)
            _ = fit_shallow_head(
                context_features,
                context_logits,
                context_labels,
                query_features,
                query_logits,
                config.shallow_lr,
                config.shallow_weight_decay,
                config.shallow_width,
                config.shallow_steps,
                seed,
            )

        def geometric() -> None:
            context_bank = bank_features(model, context_images, bank)
            query_bank = bank_features(model, query_images, bank)
            _ = ridge_probabilities(context_bank, context_labels, query_bank, config.geometric_bank_alpha)

        def tangent() -> None:
            base, jacobian = tangent_jacobian(model, context_images)
            raw = tangent_raw_solution(base, jacobian, context_labels, config.tangent_alpha)
            _ = internal_probabilities(model, query_images, raw)

        def internal_full() -> None:
            raw, _ = fit_internal(
                model,
                context_images,
                context_labels,
                config.internal_lr,
                config.internal_regularization,
                2 * config.internal_stage_steps,
            )
            _ = internal_probabilities(model, query_images, raw)

        functions = {
            "no_adaptation": no_adaptation,
            "frozen_feature_ridge": ridge,
            "rbf_kernel_ridge": rbf,
            "shallow_residual_head": shallow,
            "geometric_bank_ridge": geometric,
            "closed_form_tangent": tangent,
            "internal_full_equal_compute": internal_full,
        }
        for method, function in functions.items():
            elapsed, peak = timed(device, function)
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "wall_time_seconds": elapsed,
                    "peak_allocated_bytes": peak,
                    "query_labels_read": False,
                }
            )
        print(f"timing episode {episode_index + 1:2d}/{args.episodes}", flush=True)

    write_csv(args.output_dir / "resource_timing_episodes.csv", rows)
    summaries = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        times = np.asarray([float(row["wall_time_seconds"]) for row in method_rows])
        peaks = np.asarray([int(row["peak_allocated_bytes"]) for row in method_rows])
        summaries.append(
            {
                "method": method,
                "episodes": times.size,
                "mean_wall_time_seconds": float(times.mean()),
                "median_wall_time_seconds": float(np.median(times)),
                "p90_wall_time_seconds": float(np.quantile(times, 0.9)),
                "max_peak_allocated_bytes": int(peaks.max()),
            }
        )
    write_csv(args.output_dir / "resource_timing_summary.csv", summaries)
    metadata = {
        "timing_split": "validation",
        "timing_seed_range": [4_300_001, 4_300_000 + args.episodes],
        "performed_after_confirmation": True,
        "selection_or_performance_claims_from_timing_replay": False,
        "cuda_synchronized_before_and_after_each_method": device.type == "cuda",
        "each_method_timed_end_to_end": True,
        "query_labels_read": False,
    }
    (args.output_dir / "resource_timing_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
