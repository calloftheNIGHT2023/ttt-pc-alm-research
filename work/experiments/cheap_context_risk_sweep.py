"""Develop a two-fit context-risk mixture from one fixed calibration fold.

This is a development-only sweep. Four possible held-out folds and either a
fresh or warm-started full-context PC-ALM fit are evaluated. A later script must
freeze one variant before evaluating new episode seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from context_cv_router import episode_arrays, ridge_predict
from official_ttt_meta_bridge import MetaRidge
from tent_credit_screen import CreditConfig, forward, local_gradient


torch.set_default_dtype(torch.float64)


def fit_pcalm_raw(
    x: np.ndarray,
    y: np.ndarray,
    depth: int,
    config: CreditConfig,
    learning_rate: float,
    steps: int,
    initial_raw: torch.Tensor | None = None,
) -> torch.Tensor:
    context_x = torch.as_tensor(x, dtype=torch.float64)
    context_y = torch.as_tensor(y, dtype=torch.float64)
    raw = (
        torch.zeros(depth, dtype=torch.float64)
        if initial_raw is None
        else initial_raw.detach().clone().to(dtype=torch.float64)
    )
    for _ in range(steps):
        gradient = local_gradient(
            "pcalm", raw, context_x, context_y, config, backend="explicit"
        )
        raw = (raw - learning_rate * gradient).detach()
    return raw


def predict_raw(raw: torch.Tensor, x: np.ndarray) -> np.ndarray:
    return forward(raw, torch.as_tensor(x, dtype=torch.float64))[-1].numpy()


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
    parser.add_argument("--ridge-features", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed-offset", type=int, default=1000000)
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
    held_out_folds = np.array_split(all_indices, args.folds)

    rows = []
    for episode in range(args.episodes):
        seed = args.seed_offset + episode
        context_x, context_y, query_x, query_y = episode_arrays(
            seed, args.depth, args.n_context, args.query_size
        )
        ridge_query = ridge_predict(
            ridge_model, context_x, context_y, query_x, device
        )
        fresh_raw = fit_pcalm_raw(
            context_x,
            context_y,
            args.depth,
            config,
            args.parameter_lr,
            args.parameter_steps,
        )
        fresh_query = predict_raw(fresh_raw, query_x)
        for fold_index, held_out in enumerate(held_out_folds):
            train = np.setdiff1d(all_indices, held_out, assume_unique=True)
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
            predictions = {
                "fresh": fresh_query,
                "warm": predict_raw(warm_raw, query_x),
            }
            for full_initialization, pcalm_query in predictions.items():
                blend = weight * pcalm_query + (1.0 - weight) * ridge_query
                rows.append(
                    {
                        "seed": seed,
                        "fold_index": fold_index,
                        "full_initialization": full_initialization,
                        "query_mse": float(np.mean((blend - query_y) ** 2)),
                        "pcalm_only_query_mse": float(
                            np.mean((pcalm_query - query_y) ** 2)
                        ),
                        "ridge_only_query_mse": float(
                            np.mean((ridge_query - query_y) ** 2)
                        ),
                        "pcalm_calibration_mse": pcalm_risk,
                        "ridge_calibration_mse": ridge_risk,
                        "pcalm_weight": weight,
                    }
                )
        print(f"episode {episode + 1:3d}/{args.episodes}")

    write_csv(args.output_dir / "cheap_risk_sweep_episodes.csv", rows)
    summaries = []
    for fold_index in range(args.folds):
        for initialization in ("fresh", "warm"):
            group = [
                row
                for row in rows
                if row["fold_index"] == fold_index
                and row["full_initialization"] == initialization
            ]
            values = np.asarray([float(row["query_mse"]) for row in group])
            pcalm_values = np.asarray(
                [float(row["pcalm_only_query_mse"]) for row in group]
            )
            summaries.append(
                {
                    "fold_index": fold_index,
                    "full_initialization": initialization,
                    "episodes": values.size,
                    "mean_query_mse": float(np.mean(values)),
                    "median_query_mse": float(np.median(values)),
                    "p90_query_mse": float(np.quantile(values, 0.9)),
                    "mean_pcalm_only_query_mse": float(np.mean(pcalm_values)),
                }
            )
    write_csv(args.output_dir / "cheap_risk_sweep_summary.csv", summaries)
    metadata = vars(args) | {
        "ridge_checkpoint": str(args.ridge_checkpoint),
        "output_dir": str(args.output_dir),
        "query_targets_available_to_risk_weight": False,
        "development_only": True,
        "fit_count": {
            "fresh_variant": "one 12-point calibration fit plus one fresh 16-point fit",
            "warm_variant": "one 12-point calibration fit plus one continued 16-point fit",
        },
    }
    (args.output_dir / "cheap_risk_sweep_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
