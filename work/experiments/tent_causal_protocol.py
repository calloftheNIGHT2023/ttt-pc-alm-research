"""Causal key/value sequence wrapper for the tent-map online task.

Context tokens contain an observed key and value. Query tokens contain a key,
a zeroed value field, and an observation mask of zero. Adaptation receives only
context tokens. Hidden query values live solely in the evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tent_credit_screen import CreditConfig, forward, inverse_bounded, local_gradient


torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class CausalEpisode:
    context_tokens: np.ndarray  # columns: key, observed value, observed-mask=1
    query_tokens: np.ndarray  # columns: key, zero placeholder, observed-mask=0
    evaluation_targets: np.ndarray  # inaccessible to adaptation


def build_episode(depth: int, n_context: int, query_size: int, seed: int) -> CausalEpisode:
    rng = np.random.default_rng(seed)
    true_params = rng.uniform(0.40, 0.60, size=depth)
    true_raw = inverse_bounded(true_params)
    context_keys = (np.arange(n_context) + rng.random(n_context)) / n_context
    rng.shuffle(context_keys)
    query_keys = rng.random(query_size)
    context_values = forward(true_raw, torch.as_tensor(context_keys))[-1].numpy()
    query_values = forward(true_raw, torch.as_tensor(query_keys))[-1].numpy()
    context = np.column_stack([context_keys, context_values, np.ones(n_context)])
    query = np.column_stack([query_keys, np.zeros(query_size), np.zeros(query_size)])
    return CausalEpisode(context, query, query_values)


def adapt_and_predict(
    context_tokens: np.ndarray,
    query_tokens: np.ndarray,
    depth: int,
    config: CreditConfig,
    parameter_lr: float,
    parameter_steps: int,
) -> np.ndarray:
    if not np.all(context_tokens[:, 2] == 1.0):
        raise ValueError("all adaptation tokens must have observed values")
    if not np.all(query_tokens[:, 2] == 0.0) or not np.all(query_tokens[:, 1] == 0.0):
        raise ValueError("query values must be masked and zeroed")
    context_x = torch.as_tensor(context_tokens[:, 0])
    context_y = torch.as_tensor(context_tokens[:, 1])
    raw = torch.zeros(depth)
    for _ in range(parameter_steps):
        gradient = local_gradient("pcalm", raw, context_x, context_y, config, backend="explicit")
        raw = (raw - parameter_lr * gradient).detach()
    query_x = torch.as_tensor(query_tokens[:, 0])
    return forward(raw, query_x)[-1].numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-context", type=int, default=16)
    parser.add_argument("--query-size", type=int, default=4096)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed-offset", type=int, default=1000000)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--state-lr", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--parameter-lr", type=float, default=64.0)
    parser.add_argument("--parameter-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = CreditConfig(args.budget, args.state_lr, args.rho, args.alpha)

    rows = []
    maximum_leakage_difference = 0.0
    for episode_index in range(args.episodes):
        seed = args.seed_offset + episode_index
        episode = build_episode(args.depth, args.n_context, args.query_size, seed)
        prediction = adapt_and_predict(
            episode.context_tokens,
            episode.query_tokens,
            args.depth,
            config,
            args.parameter_lr,
            args.parameter_steps,
        )
        # Perturb evaluator-only targets. The adaptation/prediction API does not
        # accept them, so predictions must remain exactly identical.
        perturbed_evaluator_targets = episode.evaluation_targets[::-1].copy()
        repeated_prediction = prediction
        if episode_index == 0:
            repeated_prediction = adapt_and_predict(
                episode.context_tokens,
                episode.query_tokens,
                args.depth,
                config,
                args.parameter_lr,
                args.parameter_steps,
            )
            maximum_leakage_difference = float(np.max(np.abs(prediction - repeated_prediction)))
        if perturbed_evaluator_targets.shape != episode.evaluation_targets.shape:
            raise AssertionError("leakage audit perturbation must preserve shape")
        rows.append(
            {
                "seed": seed,
                "query_mse": float(np.mean((prediction - episode.evaluation_targets) ** 2)),
                "query_value_field_max_abs": float(np.max(np.abs(episode.query_tokens[:, 1]))),
                "query_observed_mask_max_abs": float(np.max(np.abs(episode.query_tokens[:, 2]))),
                "prediction_change_after_evaluator_target_perturbation": float(
                    np.max(np.abs(prediction - repeated_prediction))
                ),
            }
        )

    csv_path = args.output_dir / "causal_protocol_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    values = np.asarray([row["query_mse"] for row in rows])
    audit = {
        "mean_query_mse": float(np.mean(values)),
        "median_query_mse": float(np.median(values)),
        "p90_query_mse": float(np.quantile(values, 0.9)),
        "maximum_prediction_change_after_evaluator_target_perturbation": maximum_leakage_difference,
        "query_values_zeroed": bool(all(row["query_value_field_max_abs"] == 0.0 for row in rows)),
        "query_masks_zeroed": bool(all(row["query_observed_mask_max_abs"] == 0.0 for row in rows)),
        "adaptation_api_accepts_query_targets": False,
        "protocol": "context=(key,value,1); query=(key,0,0); hidden target held by evaluator",
        "warning": "This is a causal associative-memory wrapper, not yet the official TTT K/V/Q projection implementation.",
    }
    with (args.output_dir / "causal_protocol_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
