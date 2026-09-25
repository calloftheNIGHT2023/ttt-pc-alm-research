"""Strong adaptive one-hidden-layer ReLU baselines for tent episodes.

For scalar input, c + d*x + sum_j alpha_j relu(x - t_j) represents the
continuous piecewise-linear functions expressible by a one-hidden-layer ReLU
network. Both coefficients and knots are adapted with Adam. Restarts are chosen
only by observed-context loss; query targets are evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from tent_credit_screen import forward, inverse_bounded
from tent_depth_screen import fit_ridge, relu_features


torch.set_default_dtype(torch.float64)


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def predict_hinge(
    x: torch.Tensor,
    intercept: torch.Tensor,
    slope: torch.Tensor,
    coefficients: torch.Tensor,
    raw_knots: torch.Tensor,
) -> torch.Tensor:
    knots = torch.sigmoid(raw_knots)
    return intercept + slope * x + torch.relu(x[:, None] - knots[None, :]) @ coefficients


def initialize_restart(
    context_x: np.ndarray,
    context_y: np.ndarray,
    width: int,
    restart: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_knots = np.linspace(0.0, 1.0, width + 2)[1:-1]
    if restart:
        base_knots = np.clip(base_knots + rng.normal(0.0, 0.25 / (width + 1), size=width), 1e-4, 1 - 1e-4)
        base_knots.sort()
    initial = fit_ridge(relu_features(context_x, base_knots), context_y, ridge=1e-6)
    intercept = torch.tensor(initial[0], requires_grad=True)
    slope = torch.tensor(initial[1], requires_grad=True)
    coefficients = torch.as_tensor(initial[2:]).clone().requires_grad_(True)
    raw_knots = torch.as_tensor(logit(base_knots)).clone().requires_grad_(True)
    return intercept, slope, coefficients, raw_knots


def fit_adaptive_hinge(
    context_x: np.ndarray,
    context_y: np.ndarray,
    width: int,
    learning_rate: float,
    steps: int,
    restarts: int,
    seed: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float, int]:
    x = torch.as_tensor(context_x)
    y = torch.as_tensor(context_y)
    rng = np.random.default_rng(seed)
    best_state = None
    best_loss = float("inf")
    best_restart = -1
    for restart in range(restarts):
        parameters = initialize_restart(context_x, context_y, width, restart, rng)
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = predict_hinge(x, *parameters)
            loss = torch.mean((prediction - y) ** 2)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final_loss = float(torch.mean((predict_hinge(x, *parameters) - y) ** 2))
        if final_loss < best_loss:
            best_loss = final_loss
            best_restart = restart
            best_state = tuple(parameter.detach().clone() for parameter in parameters)
    assert best_state is not None
    return best_state, best_loss, best_restart


def predict_hinge_batch(
    x: torch.Tensor,
    intercept: torch.Tensor,
    slope: torch.Tensor,
    coefficients: torch.Tensor,
    raw_knots: torch.Tensor,
) -> torch.Tensor:
    """Predict [episode, restart, sample] from batched model states."""
    knots = torch.sigmoid(raw_knots)
    hinges = torch.relu(x[:, None, :, None] - knots[:, :, None, :])
    return (
        intercept[:, :, None]
        + slope[:, :, None] * x[:, None, :]
        + torch.sum(hinges * coefficients[:, :, None, :], dim=-1)
    )


def fit_adaptive_hinge_batch(
    context_x: np.ndarray,
    context_y: np.ndarray,
    query_x: np.ndarray,
    width: int,
    learning_rate: float,
    steps: int,
    restarts: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes = context_x.shape[0]
    rng = np.random.default_rng(seed)
    knots = np.empty((episodes, restarts, width), dtype=np.float64)
    coefficients = np.empty((episodes, restarts, width + 2), dtype=np.float64)
    uniform = np.linspace(0.0, 1.0, width + 2)[1:-1]
    for episode in range(episodes):
        for restart in range(restarts):
            current = uniform.copy()
            if restart:
                current = np.clip(
                    current + rng.normal(0.0, 0.25 / (width + 1), size=width),
                    1e-4,
                    1.0 - 1e-4,
                )
                current.sort()
            knots[episode, restart] = current
            coefficients[episode, restart] = fit_ridge(
                relu_features(context_x[episode], current),
                context_y[episode],
                ridge=1e-6,
            )

    intercept = torch.as_tensor(coefficients[:, :, 0]).clone().requires_grad_(True)
    slope = torch.as_tensor(coefficients[:, :, 1]).clone().requires_grad_(True)
    hinge_coefficients = torch.as_tensor(coefficients[:, :, 2:]).clone().requires_grad_(True)
    raw_knots = torch.as_tensor(logit(knots)).clone().requires_grad_(True)
    parameters = (intercept, slope, hinge_coefficients, raw_knots)
    x_tensor = torch.as_tensor(context_x)
    y_tensor = torch.as_tensor(context_y)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = predict_hinge_batch(x_tensor, *parameters)
        per_model_loss = torch.mean((prediction - y_tensor[:, None, :]) ** 2, dim=-1)
        per_model_loss.sum().backward()
        optimizer.step()

    with torch.no_grad():
        context_prediction = predict_hinge_batch(x_tensor, *parameters)
        context_losses = torch.mean((context_prediction - y_tensor[:, None, :]) ** 2, dim=-1)
        selected = torch.argmin(context_losses, dim=1)
        episode_index = torch.arange(episodes)
        selected_state = tuple(parameter[episode_index, selected] for parameter in parameters)
    with torch.no_grad():
        selected_knots = torch.sigmoid(selected_state[3])
        hinges = torch.relu(torch.as_tensor(query_x)[:, :, None] - selected_knots[:, None, :])
        query_prediction = (
            selected_state[0][:, None]
            + selected_state[1][:, None] * torch.as_tensor(query_x)
            + torch.sum(hinges * selected_state[2][:, None, :], dim=-1)
        )
    return query_prediction.numpy(), context_losses[episode_index, selected].numpy(), selected.numpy()


def run_episode(
    depth: int,
    n_context: int,
    query_size: int,
    seed: int,
    widths: list[int],
    learning_rate: float,
    steps: int,
    restarts: int,
) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    true_params = rng.uniform(0.40, 0.60, size=depth)
    true_raw = inverse_bounded(true_params)
    context_x = (np.arange(n_context) + rng.random(n_context)) / n_context
    rng.shuffle(context_x)
    query_x = rng.random(query_size)
    context_y = forward(true_raw, torch.as_tensor(context_x))[-1].numpy()
    query_y = forward(true_raw, torch.as_tensor(query_x))[-1].numpy()
    rows = []
    for width in widths:
        state, context_mse, selected_restart = fit_adaptive_hinge(
            context_x,
            context_y,
            width,
            learning_rate,
            steps,
            restarts,
            seed + 10_000 * width,
        )
        with torch.no_grad():
            prediction = predict_hinge(torch.as_tensor(query_x), *state).numpy()
        rows.append(
            {
                "depth": depth,
                "n_context": n_context,
                "seed": seed,
                "width": width,
                "fast_parameter_count": 2 * width + 2,
                "learning_rate": learning_rate,
                "steps": steps,
                "restarts": restarts,
                "selected_restart": selected_restart,
                "context_mse": context_mse,
                "query_mse": float(np.mean((prediction - query_y) ** 2)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-context", type=int, default=16)
    parser.add_argument("--query-size", type=int, default=4096)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed-offset", type=int, default=980000)
    parser.add_argument("--widths", default="12,15,30")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    widths = [int(value) for value in args.widths.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context_x_all = []
    context_y_all = []
    query_x_all = []
    query_y_all = []
    seeds = []
    for episode in range(args.episodes):
        seed = args.seed_offset + episode
        rng = np.random.default_rng(seed)
        true_params = rng.uniform(0.40, 0.60, size=args.depth)
        true_raw = inverse_bounded(true_params)
        context_x = (np.arange(args.n_context) + rng.random(args.n_context)) / args.n_context
        rng.shuffle(context_x)
        query_x = rng.random(args.query_size)
        context_x_all.append(context_x)
        query_x_all.append(query_x)
        context_y_all.append(forward(true_raw, torch.as_tensor(context_x))[-1].numpy())
        query_y_all.append(forward(true_raw, torch.as_tensor(query_x))[-1].numpy())
        seeds.append(seed)
    context_x_array = np.stack(context_x_all)
    context_y_array = np.stack(context_y_all)
    query_x_array = np.stack(query_x_all)
    query_y_array = np.stack(query_y_all)

    rows = []
    for width in widths:
        prediction, context_losses, selected = fit_adaptive_hinge_batch(
            context_x_array,
            context_y_array,
            query_x_array,
            width,
            args.learning_rate,
            args.steps,
            args.restarts,
            args.seed_offset + 10_000 * width,
        )
        query_losses = np.mean((prediction - query_y_array) ** 2, axis=1)
        for episode, seed in enumerate(seeds):
            rows.append(
                {
                    "depth": args.depth,
                    "n_context": args.n_context,
                    "seed": seed,
                    "width": width,
                    "fast_parameter_count": 2 * width + 2,
                    "learning_rate": args.learning_rate,
                    "steps": args.steps,
                    "restarts": args.restarts,
                    "selected_restart": int(selected[episode]),
                    "context_mse": float(context_losses[episode]),
                    "query_mse": float(query_losses[episode]),
                }
            )
    write_csv(args.output_dir / "shallow_adaptive_episodes.csv", rows)
    summaries = []
    for width in widths:
        group = [row for row in rows if row["width"] == width]
        values = np.asarray([float(row["query_mse"]) for row in group])
        context = np.asarray([float(row["context_mse"]) for row in group])
        summaries.append(
            {
                "width": width,
                "fast_parameter_count": 2 * width + 2,
                "episodes": len(group),
                "mean_query_mse": float(np.mean(values)),
                "median_query_mse": float(np.median(values)),
                "p90_query_mse": float(np.quantile(values, 0.9)),
                "median_context_mse": float(np.median(context)),
            }
        )
    write_csv(args.output_dir / "shallow_adaptive_summary.csv", summaries)
    metadata = vars(args) | {
        "widths_resolved": widths,
        "query_targets_used_for_training_or_restart_selection": False,
        "model": "c + d*x + sum_j alpha_j*relu(x-t_j); all coefficients and knots adapted",
    }
    metadata["output_dir"] = str(metadata["output_dir"])
    with (args.output_dir / "shallow_adaptive_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    for row in summaries:
        print(
            f"W={row['width']:2d} params={row['fast_parameter_count']:2d} "
            f"query_mean={row['mean_query_mse']:.3e} "
            f"median={row['median_query_mse']:.3e} p90={row['p90_query_mse']:.3e} "
            f"context={row['median_context_mse']:.3e}"
        )


if __name__ == "__main__":
    main()
