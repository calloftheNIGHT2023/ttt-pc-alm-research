"""Context-only cross-validation between PC-ALM and meta-feature ridge.

The router never sees query targets. Four folds of the 16 observed context
pairs estimate each adapter's interpolation risk. We report both a hard choice
and a parameter-free inverse-CV-error soft blend.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from official_ttt_meta_bridge import MetaRidge, compose_numpy
from tent_credit_screen import CreditConfig, forward, local_gradient


torch.set_default_dtype(torch.float32)


def episode_arrays(
    seed: int, depth: int, n_context: int, n_query: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    parameters = rng.uniform(0.40, 0.60, size=depth)
    context_x = (np.arange(n_context) + rng.random(n_context)) / n_context
    rng.shuffle(context_x)
    query_x = rng.random(n_query)
    return (
        context_x,
        compose_numpy(context_x, parameters),
        query_x,
        compose_numpy(query_x, parameters),
    )


def pcalm_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    depth: int,
    config: CreditConfig,
    parameter_lr: float,
    parameter_steps: int,
) -> np.ndarray:
    context_x = torch.as_tensor(train_x, dtype=torch.float64)
    context_y = torch.as_tensor(train_y, dtype=torch.float64)
    raw = torch.zeros(depth, dtype=torch.float64)
    for _ in range(parameter_steps):
        gradient = local_gradient(
            "pcalm", raw, context_x, context_y, config, backend="explicit"
        )
        raw = (raw - parameter_lr * gradient).detach()
    query_x = torch.as_tensor(predict_x, dtype=torch.float64)
    return forward(raw, query_x)[-1].numpy()


@torch.no_grad()
def ridge_predict(
    model: MetaRidge,
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    context_x = torch.as_tensor(train_x[None, :], dtype=torch.float32, device=device)
    context_y = torch.as_tensor(train_y[None, :], dtype=torch.float32, device=device)
    query_x = torch.as_tensor(predict_x[None, :], dtype=torch.float32, device=device)
    coefficients = model.solve(context_x, context_y)
    return (model.features(query_x) @ coefficients).squeeze(0).squeeze(-1).cpu().numpy()


def context_cv_errors(
    context_x: np.ndarray,
    context_y: np.ndarray,
    folds: int,
    pcalm_args: tuple,
    ridge_args: tuple,
) -> tuple[float, float]:
    all_indices = np.arange(context_x.size)
    fold_indices = np.array_split(all_indices, folds)
    pcalm_squared_errors = []
    ridge_squared_errors = []
    for held_out in fold_indices:
        train = np.setdiff1d(all_indices, held_out, assume_unique=True)
        pcalm_prediction = pcalm_predict(
            context_x[train], context_y[train], context_x[held_out], *pcalm_args
        )
        ridge_prediction = ridge_predict(
            ridge_args[0], context_x[train], context_y[train], context_x[held_out], ridge_args[1]
        )
        pcalm_squared_errors.extend((pcalm_prediction - context_y[held_out]) ** 2)
        ridge_squared_errors.extend((ridge_prediction - context_y[held_out]) ** 2)
    return float(np.mean(pcalm_squared_errors)), float(np.mean(ridge_squared_errors))


def run_split(
    split: str,
    seed_offset: int,
    episodes: int,
    query_size: int,
    depth: int,
    n_context: int,
    folds: int,
    pcalm_args: tuple,
    ridge_model: MetaRidge,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    rows = []
    for episode in range(episodes):
        seed = seed_offset + episode
        context_x, context_y, query_x, query_y = episode_arrays(
            seed, depth, n_context, query_size
        )
        pcalm_cv, ridge_cv = context_cv_errors(
            context_x,
            context_y,
            folds,
            pcalm_args,
            (ridge_model, device),
        )
        pcalm_prediction = pcalm_predict(
            context_x, context_y, query_x, *pcalm_args
        )
        ridge_prediction = ridge_predict(
            ridge_model, context_x, context_y, query_x, device
        )
        hard_uses_pcalm = pcalm_cv <= ridge_cv
        hard_prediction = pcalm_prediction if hard_uses_pcalm else ridge_prediction
        epsilon = 1e-12
        pcalm_weight = (ridge_cv + epsilon) / (pcalm_cv + ridge_cv + 2.0 * epsilon)
        soft_prediction = pcalm_weight * pcalm_prediction + (1.0 - pcalm_weight) * ridge_prediction
        losses = {
            "pcalm": float(np.mean((pcalm_prediction - query_y) ** 2)),
            "meta_feature_ridge": float(np.mean((ridge_prediction - query_y) ** 2)),
            "context_cv_hard_router": float(np.mean((hard_prediction - query_y) ** 2)),
            "context_cv_soft_blend": float(np.mean((soft_prediction - query_y) ** 2)),
        }
        for method, query_mse in losses.items():
            rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "method": method,
                    "query_mse": query_mse,
                    "pcalm_context_cv_mse": pcalm_cv,
                    "ridge_context_cv_mse": ridge_cv,
                    "hard_router_uses_pcalm": int(hard_uses_pcalm),
                    "soft_blend_pcalm_weight": pcalm_weight,
                }
            )
        print(
            f"{split:10s} {episode + 1:3d}/{episodes} "
            f"cv_pc={pcalm_cv:.3e} cv_ridge={ridge_cv:.3e} "
            f"pick={'pc' if hard_uses_pcalm else 'ridge'}"
        )
    return rows


def summaries(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    result = []
    for split in sorted({str(row["split"]) for row in rows}):
        for method in sorted({str(row["method"]) for row in rows}):
            values = np.asarray(
                [
                    float(row["query_mse"])
                    for row in rows
                    if row["split"] == split and row["method"] == method
                ]
            )
            result.append(
                {
                    "split": split,
                    "method": method,
                    "episodes": values.size,
                    "mean_query_mse": float(np.mean(values)),
                    "median_query_mse": float(np.median(values)),
                    "p90_query_mse": float(np.quantile(values, 0.9)),
                }
            )
    return result


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
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--state-lr", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--parameter-lr", type=float, default=64.0)
    parser.add_argument("--parameter-steps", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=16)
    parser.add_argument("--validation-queries", type=int, default=512)
    parser.add_argument("--validation-seed-offset", type=int, default=980000)
    parser.add_argument("--test-episodes", type=int, default=64)
    parser.add_argument("--test-queries", type=int, default=4096)
    parser.add_argument("--test-seed-offset", type=int, default=1000000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ridge_model = MetaRidge(args.ridge_features).to(device)
    ridge_model.load_state_dict(
        torch.load(args.ridge_checkpoint, map_location=device, weights_only=True)
    )
    ridge_model.eval()
    config = CreditConfig(args.budget, args.state_lr, args.rho, args.alpha)
    pcalm_args = (args.depth, config, args.parameter_lr, args.parameter_steps)
    validation_rows = run_split(
        "validation",
        args.validation_seed_offset,
        args.validation_episodes,
        args.validation_queries,
        args.depth,
        args.n_context,
        args.folds,
        pcalm_args,
        ridge_model,
        device,
    )
    test_rows = run_split(
        "locked_test",
        args.test_seed_offset,
        args.test_episodes,
        args.test_queries,
        args.depth,
        args.n_context,
        args.folds,
        pcalm_args,
        ridge_model,
        device,
    )
    rows = validation_rows + test_rows
    summary_rows = summaries(rows)
    write_csv(args.output_dir / "context_cv_router_episodes.csv", rows)
    write_csv(args.output_dir / "context_cv_router_summary.csv", summary_rows)
    metadata = vars(args) | {
        "ridge_checkpoint": str(args.ridge_checkpoint),
        "output_dir": str(args.output_dir),
        "query_targets_available_to_router": False,
        "hard_rule": "choose the method with smaller four-fold context CV MSE",
        "soft_rule": "weight PC-ALM by ridge_cv/(pcalm_cv+ridge_cv); no fitted threshold",
        "warning": "This is a diagnostic of observable failure, not a pre-registered primary result.",
    }
    (args.output_dir / "context_cv_router_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
