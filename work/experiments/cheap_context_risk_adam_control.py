"""Adam-BP control for the frozen two-fit context-risk mechanism."""

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


def fit_adam(
    x: np.ndarray,
    y: np.ndarray,
    depth: int,
    config: CreditConfig,
    learning_rate: float,
    steps: int,
    raw: torch.Tensor | None = None,
    first_moment: torch.Tensor | None = None,
    second_moment: torch.Tensor | None = None,
    step_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context_x = torch.as_tensor(x, dtype=torch.float64)
    context_y = torch.as_tensor(y, dtype=torch.float64)
    raw = torch.zeros(depth, dtype=torch.float64) if raw is None else raw.detach().clone()
    first_moment = (
        torch.zeros_like(raw) if first_moment is None else first_moment.detach().clone()
    )
    second_moment = (
        torch.zeros_like(raw) if second_moment is None else second_moment.detach().clone()
    )
    for local_step in range(1, steps + 1):
        global_step = step_offset + local_step
        gradient = local_gradient(
            "bp_adam", raw, context_x, context_y, config, backend="explicit"
        )
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient.square()
        corrected_first = first_moment / (1.0 - 0.9**global_step)
        corrected_second = second_moment / (1.0 - 0.999**global_step)
        update = corrected_first / (torch.sqrt(corrected_second) + 1e-8)
        raw = (raw - learning_rate * update).detach()
    return raw, first_moment, second_moment


def predict(raw: torch.Tensor, x: np.ndarray) -> np.ndarray:
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
    parser.add_argument("--frozen-fold-index", type=int, default=0)
    parser.add_argument("--ridge-features", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed-offset", type=int, default=3000000)
    parser.add_argument("--query-size", type=int, default=4096)
    parser.add_argument("--parameter-steps", type=int, default=100)
    parser.add_argument("--adam-learning-rate", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)
    ridge_model = MetaRidge(args.ridge_features).to(device)
    ridge_model.load_state_dict(
        torch.load(args.ridge_checkpoint, map_location=device, weights_only=True)
    )
    ridge_model.eval()
    torch.set_default_dtype(torch.float64)
    config = CreditConfig(2 * args.depth, 0.1, 1.0, 1.0)
    indices = np.arange(args.n_context)
    held_out = np.array_split(indices, args.folds)[args.frozen_fold_index]
    train = np.setdiff1d(indices, held_out, assume_unique=True)

    rows = []
    for episode in range(args.episodes):
        seed = args.seed_offset + episode
        context_x, context_y, query_x, query_y = episode_arrays(
            seed, args.depth, args.n_context, args.query_size
        )
        calibration_raw, first, second = fit_adam(
            context_x[train],
            context_y[train],
            args.depth,
            config,
            args.adam_learning_rate,
            args.parameter_steps,
        )
        adam_calibration = predict(calibration_raw, context_x[held_out])
        ridge_calibration = ridge_predict(
            ridge_model,
            context_x[train],
            context_y[train],
            context_x[held_out],
            device,
        )
        adam_risk = float(np.mean((adam_calibration - context_y[held_out]) ** 2))
        ridge_risk = float(np.mean((ridge_calibration - context_y[held_out]) ** 2))
        epsilon = 1e-12
        weight = (ridge_risk + epsilon) / (
            adam_risk + ridge_risk + 2.0 * epsilon
        )
        warm_raw, _, _ = fit_adam(
            context_x,
            context_y,
            args.depth,
            config,
            args.adam_learning_rate,
            args.parameter_steps,
            raw=calibration_raw,
            first_moment=first,
            second_moment=second,
            step_offset=args.parameter_steps,
        )
        full_raw, _, _ = fit_adam(
            context_x,
            context_y,
            args.depth,
            config,
            args.adam_learning_rate,
            2 * args.parameter_steps,
        )
        warm_prediction = predict(warm_raw, query_x)
        full_prediction = predict(full_raw, query_x)
        ridge_prediction = ridge_predict(
            ridge_model, context_x, context_y, query_x, device
        )
        blend_prediction = (
            weight * warm_prediction + (1.0 - weight) * ridge_prediction
        )
        predictions = {
            "cheap_context_risk_blend_adam": blend_prediction,
            "warm_two_stage_adam": warm_prediction,
            "full_context_adam_200_steps": full_prediction,
        }
        for method, prediction_value in predictions.items():
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "query_mse": float(
                        np.mean((prediction_value - query_y) ** 2)
                    ),
                    "adam_calibration_mse": adam_risk,
                    "ridge_calibration_mse": ridge_risk,
                    "adam_weight": weight,
                }
            )
        print(f"episode {episode + 1:3d}/{args.episodes}")

    write_csv(args.output_dir / "cheap_adam_control_episodes.csv", rows)
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
    write_csv(args.output_dir / "cheap_adam_control_summary.csv", summaries)
    metadata = vars(args) | {
        "ridge_checkpoint": str(args.ridge_checkpoint),
        "output_dir": str(args.output_dir),
        "same_frozen_fold_and_weight_as_pcalm_candidate": True,
        "adam_moments_preserved_across_two_stages": True,
        "query_targets_available_to_risk_weight": False,
    }
    (args.output_dir / "cheap_adam_control_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
