"""Validation-only optimizer pilot for local PC-ALM affine credit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from digits_domain_shift import (
    VALIDATION_SEED_OFFSET,
    correction_theta,
    episode_tensors,
    fit_internal,
    internal_probabilities,
    load_model_and_data,
    metric_values,
    warp,
)
from digits_pcalm_credit import PCALMConfig, bp_gradient, cosine, pcalm_gradient


def fit_pcalm_adam(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: PCALMConfig,
    parameter_learning_rate: float,
    parameter_steps: int,
    regularization: float,
    adam_epsilon: float,
    optimizer_name: str,
) -> tuple[torch.Tensor, list[dict]]:
    raw = torch.zeros(6, device=images.device)
    first = torch.zeros_like(raw)
    second = torch.zeros_like(raw)
    trace = []
    checkpoints = {1, 5, 10, 20, parameter_steps}
    for step in range(1, parameter_steps + 1):
        local, diagnostics = pcalm_gradient(model, raw, images, labels, config)
        gradient = local + 2.0 * regularization * raw / raw.numel()
        if step in checkpoints:
            reference = bp_gradient(model, raw, images, labels)
            with torch.no_grad():
                context_loss = float(
                    F.cross_entropy(
                        model(warp(images, correction_theta(raw))),
                        labels,
                    )
                )
            trace.append(
                {
                    "step": step,
                    "cosine_to_bp": cosine(local, reference),
                    "relative_norm_to_bp": float(local.norm() / reference.norm()),
                    "context_cross_entropy": context_loss,
                    "first_constraint_rms": diagnostics[
                        "final_residual_rms_by_constraint"
                    ][0],
                    "last_constraint_rms": diagnostics[
                        "final_residual_rms_by_constraint"
                    ][-1],
                }
            )
        if optimizer_name == "adam":
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient.square()
            corrected_first = first / (1.0 - 0.9**step)
            corrected_second = second / (1.0 - 0.999**step)
            update = corrected_first / (corrected_second.sqrt() + adam_epsilon)
        elif optimizer_name == "normalized_sgd":
            update = gradient / (gradient.norm() + 1e-12)
        else:
            raise ValueError(optimizer_name)
        raw = (raw - parameter_learning_rate * update).detach()
    return raw, trace


def fit_bp_normalized(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float,
    steps: int,
    regularization: float,
) -> torch.Tensor:
    raw = torch.zeros(6, device=images.device)
    for _ in range(steps):
        gradient = bp_gradient(model, raw, images, labels)
        gradient = gradient + 2.0 * regularization * raw / raw.numel()
        raw = (raw - learning_rate * gradient / (gradient.norm() + 1e-12)).detach()
    return raw


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--parameter-steps", type=int, default=40)
    parser.add_argument("--parameter-learning-rate", type=float, default=0.05)
    parser.add_argument("--activity-steps", type=int, default=8)
    parser.add_argument("--state-learning-rate", type=float, default=0.03)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--dual-learning-rate", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--optimizer", choices=("adam", "normalized_sgd"), default="adam"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, images, labels, splits = load_model_and_data(args.checkpoint_dir, device)
    config = PCALMConfig(
        activity_steps=args.activity_steps,
        state_learning_rate=args.state_learning_rate,
        rho=args.rho,
        dual_learning_rate=args.dual_learning_rate,
    )
    rows = []
    traces = []
    for episode_index in range(args.episodes):
        seed = VALIDATION_SEED_OFFSET + episode_index
        episode = episode_tensors(images, labels, splits["validation"], seed, device)
        context_images = episode["context_images"]
        context_labels = episode["context_labels"]
        query_images = episode["query_images"]
        query_labels = episode["query_labels"]
        local_raw, trace = fit_pcalm_adam(
            model,
            context_images,
            context_labels,
            config,
            args.parameter_learning_rate,
            args.parameter_steps,
            args.regularization,
            args.adam_epsilon,
            args.optimizer,
        )
        if args.optimizer == "adam":
            bp_raw, _ = fit_internal(
                model,
                context_images,
                context_labels,
                args.parameter_learning_rate,
                args.regularization,
                args.parameter_steps,
            )
        else:
            bp_raw = fit_bp_normalized(
                model,
                context_images,
                context_labels,
                args.parameter_learning_rate,
                args.parameter_steps,
                args.regularization,
            )
        local_method = f"pcalm_local_{args.optimizer}"
        bp_method = f"bp_{args.optimizer}_same_steps"
        predictions = {
            local_method: internal_probabilities(model, query_images, local_raw),
            bp_method: internal_probabilities(model, query_images, bp_raw),
        }
        for method, prediction in predictions.items():
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    **metric_values(prediction, query_labels),
                    "raw_norm": float(
                        (local_raw if method == local_method else bp_raw).norm()
                    ),
                }
            )
        for trace_row in trace:
            traces.append({"seed": seed, **trace_row})
        print(
            f"episode {episode_index + 1}/{args.episodes}: "
            f"PCALM={rows[-2]['query_brier']:.4f}, BP={rows[-1]['query_brier']:.4f}",
            flush=True,
        )
    write_csv(args.output_dir / "pilot_episodes.csv", rows)
    write_csv(args.output_dir / "credit_trace.csv", traces)
    summaries = []
    for method in sorted({row["method"] for row in rows}):
        values = np.asarray(
            [row["query_brier"] for row in rows if row["method"] == method]
        )
        summaries.append(
            {
                "method": method,
                "episodes": values.size,
                "mean_query_brier": float(values.mean()),
                "median_query_brier": float(np.median(values)),
            }
        )
    write_csv(args.output_dir / "pilot_summary.csv", summaries)
    metadata = vars(args) | {
        "checkpoint_dir": str(args.checkpoint_dir),
        "output_dir": str(args.output_dir),
        "split": "validation",
        "query_targets_used_for_update": False,
        "bp_gradient_used_by_pcalm_update": False,
        "bp_gradient_computed_only_at_trace_checkpoints": True,
        "optimizer_same_for_local_and_global_credit": args.optimizer,
    }
    (args.output_dir / "pilot_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
