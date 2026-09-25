"""Stage-B few-context screen for compositional tent-map episodes.

The nonlinear least-squares method is an oracle-style internal-parameter solver,
not PC-ALM.  Its purpose is to test identifiability before comparing credit
assignment algorithms.  Query labels are never used for fitting or selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from tent_depth_screen import compose_tents, fit_ridge, relu_features


def compose_variable(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    for a in params:
        out = np.where(out <= a, out / a, (1.0 - out) / (1.0 - a))
    return out


def finite_difference_jacobian(x: np.ndarray, init: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    columns = []
    for index in range(init.size):
        plus = init.copy()
        minus = init.copy()
        plus[index] += eps
        minus[index] -= eps
        columns.append((compose_variable(x, plus) - compose_variable(x, minus)) / (2.0 * eps))
    return np.column_stack(columns)


def fit_kernel_ridge_cv(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    gamma_grid = np.logspace(0.0, 5.0, 21)
    best_score = float("inf")
    best_alpha = None
    best_gamma = None
    distances = (x[:, None] - x[None, :]) ** 2
    for gamma in gamma_grid:
        kernel = np.exp(-gamma * distances)
        system = kernel + ridge * np.eye(x.size)
        inverse = np.linalg.inv(system)
        alpha = inverse @ y
        diagonal = np.diag(inverse)
        loo_residual = alpha / diagonal
        score = float(np.mean(loo_residual**2))
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_gamma = float(gamma)
    assert best_alpha is not None and best_gamma is not None
    return best_alpha, best_gamma


def one_episode(depth: int, n_context: int, seed: int, query_size: int, ridge: float) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    true_params = rng.uniform(0.40, 0.60, size=depth)
    # Stratification reduces accidental holes without using target/query answers.
    context_x = (np.arange(n_context) + rng.random(n_context)) / n_context
    rng.shuffle(context_x)
    context_y = compose_variable(context_x, true_params)
    query_x = rng.random(query_size)
    query_y = compose_variable(query_x, true_params)

    init = np.full(depth, 0.5)
    nonlinear = least_squares(
        lambda p: compose_variable(context_x, p) - context_y,
        init,
        bounds=(np.full(depth, 0.35), np.full(depth, 0.65)),
        max_nfev=1000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    nonlinear_pred = compose_variable(query_x, nonlinear.x)

    base_context = compose_variable(context_x, init)
    context_jacobian = finite_difference_jacobian(context_x, init)
    delta = fit_ridge(
        np.column_stack([np.ones(n_context), context_jacobian]),
        context_y - base_context,
        ridge=ridge,
    )
    base_query = compose_variable(query_x, init)
    query_jacobian = finite_difference_jacobian(query_x, init)
    jacobian_pred = base_query + np.column_stack([np.ones(query_size), query_jacobian]) @ delta
    one_step_params = np.clip(init + delta[1:], 0.35, 0.65)
    one_step_pred = compose_variable(query_x, one_step_params)

    width = 3 * depth
    knots = np.linspace(0.0, 1.0, width + 2)[1:-1]
    fixed_coef = fit_ridge(relu_features(context_x, knots), context_y, ridge=ridge)
    fixed_pred = relu_features(query_x, knots) @ fixed_coef

    kernel_alpha, kernel_gamma = fit_kernel_ridge_cv(context_x, context_y, ridge=ridge)
    kernel_pred = np.exp(-kernel_gamma * (query_x[:, None] - context_x[None, :]) ** 2) @ kernel_alpha

    return {
        "depth": depth,
        "n_context": n_context,
        "seed": seed,
        "nonlinear_internal_mse": float(np.mean((nonlinear_pred - query_y) ** 2)),
        "nonlinear_parameter_mse": float(np.mean((nonlinear.x - true_params) ** 2)),
        "nonlinear_nfev": int(nonlinear.nfev),
        "nonlinear_success": int(nonlinear.success),
        "jacobian_ridge_mse": float(np.mean((jacobian_pred - query_y) ** 2)),
        "one_gauss_newton_step_mse": float(np.mean((one_step_pred - query_y) ** 2)),
        "fixed_relu_ridge_mse": float(np.mean((fixed_pred - query_y) ** 2)),
        "kernel_ridge_mse": float(np.mean((kernel_pred - query_y) ** 2)),
        "kernel_selected_gamma": kernel_gamma,
    }


def summarize(rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    metric_names = [
        "nonlinear_internal_mse",
        "nonlinear_parameter_mse",
        "nonlinear_nfev",
        "jacobian_ridge_mse",
        "one_gauss_newton_step_mse",
        "fixed_relu_ridge_mse",
        "kernel_ridge_mse",
    ]
    keys = sorted({(int(row["depth"]), int(row["n_context"])) for row in rows})
    result = []
    for depth, n_context in keys:
        group = [row for row in rows if row["depth"] == depth and row["n_context"] == n_context]
        summary: dict[str, float | int] = {"depth": depth, "n_context": n_context, "episodes": len(group)}
        for name in metric_names:
            values = np.array([float(row[name]) for row in group])
            summary[f"median_{name}"] = float(np.median(values))
            summary[f"p90_{name}"] = float(np.quantile(values, 0.9))
        result.append(summary)
    return result


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", default="2,3,4,5")
    parser.add_argument("--context-multipliers", default="2,4,8,16")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--query-size", type=int, default=4096)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    depths = [int(value) for value in args.depths.split(",")]
    multipliers = [int(value) for value in args.context_multipliers.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for depth in depths:
        for multiplier in multipliers:
            n_context = multiplier * depth
            for episode in range(args.episodes):
                seed = 100_000 * depth + 1_000 * n_context + episode
                rows.append(one_episode(depth, n_context, seed, args.query_size, args.ridge))
    summaries = summarize(rows)
    write_csv(args.output_dir / "tent_online_episodes.csv", rows)
    write_csv(args.output_dir / "tent_online_summary.csv", summaries)
    metadata = {
        "purpose": "identifiability screen; nonlinear least squares is not PC-ALM",
        "query_targets_used_for_fitting_or_selection": False,
        "context_sampling": "stratified uniform with within-bin jitter",
        "episode_parameter_range": [0.4, 0.6],
        "solver_bounds": [0.35, 0.65],
        "depths": depths,
        "context_multipliers": multipliers,
        "episodes": args.episodes,
        "query_size": args.query_size,
        "ridge": args.ridge,
    }
    with (args.output_dir / "tent_online_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    for row in summaries:
        print(
            f"L={row['depth']} n={row['n_context']:3d} "
            f"NLS={row['median_nonlinear_internal_mse']:.3e} "
            f"J-ridge={row['median_jacobian_ridge_mse']:.3e} "
            f"GN1={row['median_one_gauss_newton_step_mse']:.3e} "
            f"ReLU={row['median_fixed_relu_ridge_mse']:.3e} "
            f"KRR={row['median_kernel_ridge_mse']:.3e}"
        )


if __name__ == "__main__":
    main()
