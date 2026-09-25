"""Stage-A representational screen for the TTT x PC-ALM project.

This is deliberately not a PC-ALM experiment.  It checks whether a compositional
task gives a resource separation from frozen closed-form heads before we spend
time on a local-credit update rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def tent(x: np.ndarray, a: float = 0.5) -> np.ndarray:
    if not 0.0 < a < 1.0:
        raise ValueError("a must lie strictly between 0 and 1")
    return np.where(x <= a, x / a, (1.0 - x) / (1.0 - a))


def compose_tents(x: np.ndarray, depth: int, a: float = 0.5) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    for _ in range(depth):
        out = tent(out, a)
    return out


def relu_features(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x, np.maximum(x[:, None] - knots[None, :], 0.0)])


def fit_ridge(phi: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    gram = phi.T @ phi
    penalty = np.eye(gram.shape[0]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, phi.T @ y)


def rbf_features(x: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    return np.column_stack([np.ones_like(x), np.exp(-gamma * (x[:, None] - centers[None, :]) ** 2)])


def alternating_query_points(depth: int) -> tuple[np.ndarray, np.ndarray]:
    crossings = (2 * np.arange(2**depth) + 1) / (2 ** (depth + 1))
    boundaries = np.concatenate(([0.0], crossings, [1.0]))
    points = 0.5 * (boundaries[:-1] + boundaries[1:])
    labels = (compose_tents(points, depth) >= 0.5).astype(np.int64)
    expected = np.arange(labels.size) % 2
    if not np.array_equal(labels, expected):
        raise AssertionError("constructed labels must alternate")
    return points, labels


def screen_depth(depth: int, grid_size: int, ridge: float) -> dict[str, float | int]:
    width = 3 * depth
    train_x = (np.arange(grid_size) + 0.25) / grid_size
    valid_x = (np.arange(grid_size) + 0.50) / grid_size
    query_x = (np.arange(grid_size) + 0.75) / grid_size
    train_y = compose_tents(train_x, depth)
    valid_y = compose_tents(valid_x, depth)
    query_y = compose_tents(query_x, depth)

    knots = np.linspace(0.0, 1.0, width + 2)[1:-1]
    relu_coef = fit_ridge(relu_features(train_x, knots), train_y, ridge=0.0)
    relu_pred = relu_features(query_x, knots) @ relu_coef
    ridge_coef = fit_ridge(relu_features(train_x, knots), train_y, ridge=ridge)
    ridge_pred = relu_features(query_x, knots) @ ridge_coef

    centers = np.linspace(0.0, 1.0, width)
    gamma_grid = np.logspace(0.0, 5.0, 31)
    best_gamma = None
    best_valid = float("inf")
    best_coef = None
    for gamma in gamma_grid:
        phi_train = rbf_features(train_x, centers, gamma)
        coef = fit_ridge(phi_train, train_y, ridge=ridge)
        pred = rbf_features(valid_x, centers, gamma) @ coef
        mse = float(np.mean((pred - valid_y) ** 2))
        if mse < best_valid:
            best_valid = mse
            best_gamma = float(gamma)
            best_coef = coef
    assert best_gamma is not None and best_coef is not None
    rbf_pred = rbf_features(query_x, centers, best_gamma) @ best_coef

    discrete_x, discrete_y = alternating_query_points(depth)
    exact_pred = (compose_tents(discrete_x, depth) >= 0.5).astype(np.int64)
    n_runs = 2**depth + 1
    lower_bound_errors = max(0, int(np.ceil((n_runs - (width + 2)) / 2)))

    return {
        "depth": depth,
        "deep_relu_units": width,
        "episode_fast_parameters": depth,
        "target_affine_pieces": 2**depth,
        "minimum_shallow_width_exact": 2**depth - 1,
        "matched_shallow_width": width,
        "shallow_classification_error_lower_bound": lower_bound_errors / n_runs,
        "deep_exact_classification_error": float(np.mean(exact_pred != discrete_y)),
        "fixed_relu_ols_query_mse": float(np.mean((relu_pred - query_y) ** 2)),
        "fixed_relu_ridge_query_mse": float(np.mean((ridge_pred - query_y) ** 2)),
        "matched_rbf_query_mse": float(np.mean((rbf_pred - query_y) ** 2)),
        "matched_rbf_selected_gamma": best_gamma,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=16384)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [screen_depth(d, args.grid_size, args.ridge) for d in range(args.min_depth, args.max_depth + 1)]
    csv_path = args.output_dir / "tent_depth_screen.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "purpose": "representational screen only; not an online-learning or PC-ALM result",
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "grid_size_per_split": args.grid_size,
        "ridge": args.ridge,
        "rbf_gamma_selection": "selected on a disjoint dense validation grid",
        "query_targets_used_for_model_selection": False,
    }
    with (args.output_dir / "tent_depth_screen_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(csv_path)
    for row in rows:
        print(
            f"L={row['depth']:2d} exact_width>={row['minimum_shallow_width_exact']:4d} "
            f"matched_W={row['matched_shallow_width']:2d} "
            f"class_lb={row['shallow_classification_error_lower_bound']:.3f} "
            f"relu_mse={row['fixed_relu_ols_query_mse']:.3e} "
            f"rbf_mse={row['matched_rbf_query_mse']:.3e}"
        )


if __name__ == "__main__":
    main()
