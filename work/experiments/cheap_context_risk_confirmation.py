"""Confirm the frozen two-fit context-risk mixture on new episode seeds.

Frozen development choice: fold 0 of four is held out for risk calibration;
PC-ALM is first fit on the other 12 observations and then warm-started on all
16. The risk weight has no fitted threshold and never receives query targets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from cheap_context_risk_sweep import fit_pcalm_raw, predict_raw
from context_cv_router import episode_arrays, ridge_predict
from official_ttt_meta_bridge import MetaRidge
from tent_credit_screen import CreditConfig


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ridge-checkpoint", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-context", type=int, default=16)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--frozen-fold-index", type=int, default=0)
    parser.add_argument("--ridge-features", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed-offset", type=int, default=3000000)
    parser.add_argument("--query-size", type=int, default=4096)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--state-lr", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--parameter-lr", type=float, default=64.0)
    parser.add_argument("--parameter-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.n_context % args.folds:
        raise ValueError("n_context must be divisible by folds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)
    ridge_model = MetaRidge(args.ridge_features).to(device)
    ridge_model.load_state_dict(
        torch.load(args.ridge_checkpoint, map_location=device, weights_only=True)
    )
    ridge_model.eval()
    torch.set_default_dtype(torch.float64)
    config = CreditConfig(args.budget, args.state_lr, args.rho, args.alpha)
    all_indices = np.arange(args.n_context)
    held_out = np.array_split(all_indices, args.folds)[args.frozen_fold_index]
    train = np.setdiff1d(all_indices, held_out, assume_unique=True)

    rows = []
    for episode in range(args.episodes):
        seed = args.seed_offset + episode
        context_x, context_y, query_x, query_y = episode_arrays(
            seed, args.depth, args.n_context, args.query_size
        )
        calibration_raw = fit_pcalm_raw(
            context_x[train],
            context_y[train],
            args.depth,
            config,
            args.parameter_lr,
            args.parameter_steps,
        )
        pcalm_calibration = predict_raw(calibration_raw, context_x[held_out])
        ridge_calibration = ridge_predict(
            ridge_model,
            context_x[train],
            context_y[train],
            context_x[held_out],
            device,
        )
        pcalm_risk = float(np.mean((pcalm_calibration - context_y[held_out]) ** 2))
        ridge_risk = float(np.mean((ridge_calibration - context_y[held_out]) ** 2))
        epsilon = 1e-12
        weight = (ridge_risk + epsilon) / (
            pcalm_risk + ridge_risk + 2.0 * epsilon
        )
        warm_raw = fit_pcalm_raw(
            context_x,
            context_y,
            args.depth,
            config,
            args.parameter_lr,
            args.parameter_steps,
            initial_raw=calibration_raw,
        )
        equal_compute_raw = fit_pcalm_raw(
            context_x,
            context_y,
            args.depth,
            config,
            args.parameter_lr,
            2 * args.parameter_steps,
        )
        warm_prediction = predict_raw(warm_raw, query_x)
        equal_compute_prediction = predict_raw(equal_compute_raw, query_x)
        ridge_prediction = ridge_predict(
            ridge_model, context_x, context_y, query_x, device
        )
        blend_prediction = (
            weight * warm_prediction + (1.0 - weight) * ridge_prediction
        )
        predictions = {
            "cheap_context_risk_blend": blend_prediction,
            "warm_two_stage_pcalm": warm_prediction,
            "full_context_pcalm_200_steps": equal_compute_prediction,
            "meta_feature_ridge": ridge_prediction,
        }
        for method, prediction in predictions.items():
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "query_mse": float(np.mean((prediction - query_y) ** 2)),
                    "pcalm_calibration_mse": pcalm_risk,
                    "ridge_calibration_mse": ridge_risk,
                    "pcalm_weight": weight,
                }
            )
        print(f"episode {episode + 1:3d}/{args.episodes}")

    write_csv(args.output_dir / "cheap_risk_confirmation_episodes.csv", rows)
    summaries = []
    for method in sorted({row["method"] for row in rows}):
        values = np.asarray(
            [float(row["query_mse"]) for row in rows if row["method"] == method]
        )
        summaries.append(
            {
                "method": method,
                "episodes": values.size,
                "mean_query_mse": float(np.mean(values)),
                "median_query_mse": float(np.median(values)),
                "p90_query_mse": float(np.quantile(values, 0.9)),
            }
        )
    write_csv(args.output_dir / "cheap_risk_confirmation_summary.csv", summaries)
    metadata = vars(args) | {
        "ridge_checkpoint": str(args.ridge_checkpoint),
        "output_dir": str(args.output_dir),
        "selection_frozen_before_this_seed_range": True,
        "development_seed_range": [1000000, 1000063],
        "confirmation_seed_range": [args.seed_offset, args.seed_offset + args.episodes - 1],
        "query_targets_available_to_risk_weight": False,
        "weight_rule": "(ridge_calibration_mse+1e-12)/(pcalm_calibration_mse+ridge_calibration_mse+2e-12)",
        "candidate_compute": "1600 PC-ALM activity sweeps plus two ridge solves",
        "equal_compute_control": "200 PC-ALM parameter steps on all 16 observations",
    }
    (args.output_dir / "cheap_risk_confirmation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
