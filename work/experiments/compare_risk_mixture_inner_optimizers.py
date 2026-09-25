"""Apply the same context-risk mixture to matched internal optimizers."""

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


def inner_predict(
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    depth: int,
    config: CreditConfig,
    learning_rate: float,
    parameter_steps: int,
) -> np.ndarray:
    context_x = torch.as_tensor(train_x, dtype=torch.float64)
    context_y = torch.as_tensor(train_y, dtype=torch.float64)
    raw = torch.zeros(depth, dtype=torch.float64)
    first_moment = torch.zeros_like(raw)
    second_moment = torch.zeros_like(raw)
    for step in range(1, parameter_steps + 1):
        gradient = local_gradient(
            method, raw, context_x, context_y, config, backend="explicit"
        )
        if method == "bp_adam":
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient.square()
            corrected_first = first_moment / (1.0 - 0.9**step)
            corrected_second = second_moment / (1.0 - 0.999**step)
            update = corrected_first / (torch.sqrt(corrected_second) + 1e-8)
        elif method == "bp_normalized":
            update = gradient / (torch.linalg.vector_norm(gradient) + 1e-12)
        else:
            update = gradient
        raw = (raw - learning_rate * update).detach()
    return forward(raw, torch.as_tensor(predict_x, dtype=torch.float64))[-1].numpy()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ridge-checkpoint", type=Path, required=True)
    parser.add_argument("--methods", default="bp,bp_adam,bp_normalized,pc")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-context", type=int, default=16)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ridge-features", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed-offset", type=int, default=2000000)
    parser.add_argument("--query-size", type=int, default=4096)
    parser.add_argument("--parameter-steps", type=int, default=100)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--state-lr", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--bp-lr", type=float, default=3.0)
    parser.add_argument("--bp-adam-lr", type=float, default=0.1)
    parser.add_argument("--bp-normalized-lr", type=float, default=0.5)
    parser.add_argument("--pc-lr", type=float, default=64.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    learning_rates = {
        "bp": args.bp_lr,
        "bp_adam": args.bp_adam_lr,
        "bp_normalized": args.bp_normalized_lr,
        "pc": args.pc_lr,
    }
    unknown = set(methods) - set(learning_rates)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    device = torch.device(args.device)
    # MetaRidge was trained in float32.
    torch.set_default_dtype(torch.float32)
    ridge_model = MetaRidge(args.ridge_features).to(device)
    ridge_model.load_state_dict(
        torch.load(args.ridge_checkpoint, map_location=device, weights_only=True)
    )
    ridge_model.eval()
    torch.set_default_dtype(torch.float64)
    config = CreditConfig(args.budget, args.state_lr, args.rho, args.alpha)

    rows = []
    all_indices = np.arange(args.n_context)
    folds = np.array_split(all_indices, args.folds)
    for episode in range(args.episodes):
        seed = args.seed_offset + episode
        context_x, context_y, query_x, query_y = episode_arrays(
            seed, args.depth, args.n_context, args.query_size
        )
        ridge_cv_squared_errors = []
        inner_cv_squared_errors = {method: [] for method in methods}
        for held_out in folds:
            train = np.setdiff1d(all_indices, held_out, assume_unique=True)
            ridge_fold_prediction = ridge_predict(
                ridge_model,
                context_x[train],
                context_y[train],
                context_x[held_out],
                device,
            )
            ridge_cv_squared_errors.extend(
                (ridge_fold_prediction - context_y[held_out]) ** 2
            )
            for method in methods:
                prediction = inner_predict(
                    method,
                    context_x[train],
                    context_y[train],
                    context_x[held_out],
                    args.depth,
                    config,
                    learning_rates[method],
                    args.parameter_steps,
                )
                inner_cv_squared_errors[method].extend(
                    (prediction - context_y[held_out]) ** 2
                )
        ridge_cv = float(np.mean(ridge_cv_squared_errors))
        ridge_query_prediction = ridge_predict(
            ridge_model, context_x, context_y, query_x, device
        )
        for method in methods:
            inner_cv = float(np.mean(inner_cv_squared_errors[method]))
            inner_query_prediction = inner_predict(
                method,
                context_x,
                context_y,
                query_x,
                args.depth,
                config,
                learning_rates[method],
                args.parameter_steps,
            )
            epsilon = 1e-12
            inner_weight = (ridge_cv + epsilon) / (
                inner_cv + ridge_cv + 2.0 * epsilon
            )
            mixture_prediction = (
                inner_weight * inner_query_prediction
                + (1.0 - inner_weight) * ridge_query_prediction
            )
            rows.extend(
                [
                    {
                        "seed": seed,
                        "inner_method": method,
                        "result": "inner_only",
                        "query_mse": float(
                            np.mean((inner_query_prediction - query_y) ** 2)
                        ),
                        "inner_context_cv_mse": inner_cv,
                        "ridge_context_cv_mse": ridge_cv,
                        "inner_weight": inner_weight,
                    },
                    {
                        "seed": seed,
                        "inner_method": method,
                        "result": "context_cv_soft_blend",
                        "query_mse": float(
                            np.mean((mixture_prediction - query_y) ** 2)
                        ),
                        "inner_context_cv_mse": inner_cv,
                        "ridge_context_cv_mse": ridge_cv,
                        "inner_weight": inner_weight,
                    },
                ]
            )
        print(f"episode {episode + 1:3d}/{args.episodes}")

    write_csv(args.output_dir / "inner_optimizer_mixture_episodes.csv", rows)
    summary_rows = []
    for method in methods:
        for result in ("inner_only", "context_cv_soft_blend"):
            values = np.asarray(
                [
                    float(row["query_mse"])
                    for row in rows
                    if row["inner_method"] == method and row["result"] == result
                ]
            )
            summary_rows.append(
                {
                    "inner_method": method,
                    "result": result,
                    "episodes": values.size,
                    "mean_query_mse": float(np.mean(values)),
                    "median_query_mse": float(np.median(values)),
                    "p90_query_mse": float(np.quantile(values, 0.9)),
                }
            )
    write_csv(args.output_dir / "inner_optimizer_mixture_summary.csv", summary_rows)
    metadata = vars(args) | {
        "ridge_checkpoint": str(args.ridge_checkpoint),
        "output_dir": str(args.output_dir),
        "query_targets_available_to_weight_rule": False,
        "weight_rule": "(ridge_cv+1e-12)/(inner_cv+ridge_cv+2e-12)",
        "note": "Same context folds, ridge expert, parameter steps, and validation-selected optimizer learning rates as prior comparisons.",
    }
    (args.output_dir / "inner_optimizer_mixture_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
