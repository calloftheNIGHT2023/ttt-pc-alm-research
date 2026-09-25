"""Compare BP, ordinary PC, and PC-ALM credit on tent-composition episodes.

This is a small mathematical prototype. PC/PC-ALM can use either an autograd
reference backend or explicit adjacent-layer derivatives. The explicit backend
does not invoke reverse-mode autodiff for PC/PC-ALM credit; BP controls still do.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tent_depth_screen import fit_ridge, relu_features
from tent_online_screen import compose_variable, finite_difference_jacobian, fit_kernel_ridge_cv


torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class CreditConfig:
    budget: int
    state_lr: float
    rho: float
    alpha: float
    inner_steps: int = 1


def bounded_params(raw: torch.Tensor) -> torch.Tensor:
    return 0.35 + 0.30 * torch.sigmoid(raw)


def inverse_bounded(params: np.ndarray) -> torch.Tensor:
    probability = (torch.as_tensor(params) - 0.35) / 0.30
    return torch.log(probability / (1.0 - probability))


def tent(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= a, x / a, (1.0 - x) / (1.0 - a))


def tent_with_derivatives(x: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = x <= a
    value = torch.where(left, x / a, (1.0 - x) / (1.0 - a))
    derivative_x = torch.where(left, torch.ones_like(x) / a, -torch.ones_like(x) / (1.0 - a))
    derivative_a = torch.where(left, -x / a.square(), (1.0 - x) / (1.0 - a).square())
    return value, derivative_x, derivative_a


def bounded_param_derivative(raw: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(raw)
    return 0.30 * probability * (1.0 - probability)


def forward(raw: torch.Tensor, x: torch.Tensor) -> list[torch.Tensor]:
    states = []
    out = x
    for a in bounded_params(raw):
        out = tent(out, a)
        states.append(out)
    return states


def mse_half(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean((prediction - target) ** 2)


def bp_gradient(raw: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    variable = raw.detach().clone().requires_grad_(True)
    loss = mse_half(forward(variable, x)[-1], y)
    return torch.autograd.grad(loss, variable)[0].detach()


def residuals(raw: torch.Tensor, x: torch.Tensor, free: list[torch.Tensor]) -> list[torch.Tensor]:
    params = bounded_params(raw)
    result = []
    for index, activity in enumerate(free):
        previous = x if index == 0 else free[index - 1]
        result.append(activity - tent(previous, params[index]))
    return result


def local_energy(
    raw: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    free: list[torch.Tensor],
    duals: list[torch.Tensor],
    rho: float,
) -> torch.Tensor:
    params = bounded_params(raw)
    prediction = tent(free[-1], params[-1]) if free else tent(x, params[-1])
    total = mse_half(prediction, y)
    for error, dual in zip(residuals(raw, x, free), duals):
        total = total + torch.mean(dual * error + 0.5 * rho * error.square())
    return total


def initial_free(raw: torch.Tensor, x: torch.Tensor) -> list[torch.Tensor]:
    states = forward(raw.detach(), x)
    return [state.detach().clone() for state in states[:-1]]


def solve_activities(
    raw: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    free: list[torch.Tensor],
    duals: list[torch.Tensor],
    config: CreditConfig,
    steps: int,
) -> list[torch.Tensor]:
    # Energy is a batch mean; this restores a per-example activity step.
    effective_lr = config.state_lr * x.numel()
    for _ in range(steps):
        variables = [activity.detach().clone().requires_grad_(True) for activity in free]
        energy = local_energy(raw.detach(), x, y, variables, duals, config.rho)
        gradients = torch.autograd.grad(energy, variables)
        free = [(activity - effective_lr * gradient).detach() for activity, gradient in zip(variables, gradients)]
    return free


def solve_activities_explicit(
    raw: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    free: list[torch.Tensor],
    duals: list[torch.Tensor],
    config: CreditConfig,
    steps: int,
) -> list[torch.Tensor]:
    params = bounded_params(raw)
    for _ in range(steps):
        errors = residuals(raw, x, free)
        constraint_credit = [dual + config.rho * error for dual, error in zip(duals, errors)]
        gradients = []
        for index, activity in enumerate(free):
            gradient = constraint_credit[index]
            if index + 1 < len(free):
                _, derivative_next_input, _ = tent_with_derivatives(activity, params[index + 1])
                gradient = gradient - constraint_credit[index + 1] * derivative_next_input
            else:
                prediction, derivative_output_input, _ = tent_with_derivatives(activity, params[-1])
                gradient = gradient + (prediction - y) * derivative_output_input
            gradients.append(gradient)
        free = [(activity - config.state_lr * gradient).detach() for activity, gradient in zip(free, gradients)]
    return free


def explicit_weight_gradient(
    raw: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    free: list[torch.Tensor],
    duals: list[torch.Tensor],
    rho: float,
) -> torch.Tensor:
    params = bounded_params(raw)
    chain = bounded_param_derivative(raw)
    gradients = []
    errors = residuals(raw, x, free)
    for index, (error, dual) in enumerate(zip(errors, duals)):
        previous = x if index == 0 else free[index - 1]
        _, _, derivative_parameter = tent_with_derivatives(previous, params[index])
        credit = dual + rho * error
        gradients.append(torch.mean(-credit * derivative_parameter) * chain[index])
    previous = x if not free else free[-1]
    prediction, _, derivative_parameter = tent_with_derivatives(previous, params[-1])
    gradients.append(torch.mean((prediction - y) * derivative_parameter) * chain[-1])
    return torch.stack(gradients).detach()


def local_gradient(
    method: str,
    raw: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    config: CreditConfig,
    backend: str = "autograd",
) -> torch.Tensor:
    if method in {"bp", "bp_adam", "bp_normalized"}:
        return bp_gradient(raw, x, y)
    if backend not in {"autograd", "explicit"}:
        raise ValueError(f"unknown local backend: {backend}")
    free = initial_free(raw, x)
    duals = [torch.zeros_like(activity) for activity in free]
    activity_solver = solve_activities_explicit if backend == "explicit" else solve_activities
    if method == "pc":
        free = activity_solver(raw, x, y, free, duals, config, config.budget)
        weight_duals = duals
    elif method == "pcalm":
        for _ in range(max(config.budget - 1, 0)):
            free = activity_solver(raw, x, y, free, duals, config, config.inner_steps)
            current = residuals(raw.detach(), x, free)
            duals = [(dual + config.alpha * error).detach() for dual, error in zip(duals, current)]
        free = activity_solver(raw, x, y, free, duals, config, config.inner_steps)
        weight_duals = duals
    else:
        raise ValueError(f"unknown method: {method}")
    detached_free = [activity.detach() for activity in free]
    detached_duals = [dual.detach() for dual in weight_duals]
    if backend == "explicit":
        return explicit_weight_gradient(raw.detach(), x, y, detached_free, detached_duals, config.rho)
    variable = raw.detach().clone().requires_grad_(True)
    energy = local_energy(variable, x, y, detached_free, detached_duals, config.rho)
    return torch.autograd.grad(energy, variable)[0].detach()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def run_episode(
    depth: int,
    n_context: int,
    seed: int,
    config: CreditConfig,
    parameter_lrs: dict[str, float],
    parameter_steps: int,
    query_size: int,
    methods: tuple[str, ...],
    local_backend: str,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    true_params_np = rng.uniform(0.40, 0.60, size=depth)
    true_raw = inverse_bounded(true_params_np)
    context_x_np = (np.arange(n_context) + rng.random(n_context)) / n_context
    rng.shuffle(context_x_np)
    query_x_np = rng.random(query_size)
    context_x = torch.as_tensor(context_x_np)
    query_x = torch.as_tensor(query_x_np)
    context_y = forward(true_raw, context_x)[-1].detach()
    query_y = forward(true_raw, query_x)[-1].detach()

    initial = torch.zeros(depth)
    bp_initial = bp_gradient(initial, context_x, context_y)
    rows = []
    trainable_methods = tuple(
        method for method in methods if method in {"bp", "bp_adam", "bp_normalized", "pc", "pcalm"}
    )
    for method in trainable_methods:
        raw = initial.clone()
        first_gradient = local_gradient(method, raw, context_x, context_y, config, backend=local_backend)
        first_cosine = cosine(first_gradient, bp_initial)
        first_relative_norm = float(torch.linalg.vector_norm(first_gradient) / torch.linalg.vector_norm(bp_initial))
        first_moment = torch.zeros_like(raw)
        second_moment = torch.zeros_like(raw)
        for step in range(1, parameter_steps + 1):
            gradient = local_gradient(method, raw, context_x, context_y, config, backend=local_backend)
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
            raw = (raw - parameter_lrs[method] * update).detach()
        prediction = forward(raw, query_x)[-1]
        query_mse = float(torch.mean((prediction - query_y) ** 2))
        parameter_mse = float(torch.mean((bounded_params(raw) - torch.as_tensor(true_params_np)) ** 2))
        rows.append(
            {
                "depth": depth,
                "n_context": n_context,
                "seed": seed,
                "method": method,
                "query_mse": query_mse,
                "parameter_mse": parameter_mse,
                "first_gradient_cosine_to_bp": first_cosine,
                "first_gradient_relative_norm": first_relative_norm,
            }
        )

    baseline_methods = {"jacobian_ridge", "gauss_newton_1", "fixed_relu_ridge", "kernel_ridge"}
    if not baseline_methods.intersection(methods):
        return rows

    context_y_np = context_y.numpy()
    query_y_np = query_y.numpy()
    init_params_np = np.full(depth, 0.5)
    base_context = compose_variable(context_x_np, init_params_np)
    context_jacobian = finite_difference_jacobian(context_x_np, init_params_np)
    delta = fit_ridge(
        np.column_stack([np.ones(n_context), context_jacobian]),
        context_y_np - base_context,
        ridge=1e-6,
    )
    base_query = compose_variable(query_x_np, init_params_np)
    query_jacobian = finite_difference_jacobian(query_x_np, init_params_np)
    jacobian_pred = base_query + np.column_stack([np.ones(query_size), query_jacobian]) @ delta
    one_step_params = np.clip(init_params_np + delta[1:], 0.35, 0.65)
    one_step_pred = compose_variable(query_x_np, one_step_params)

    width = 3 * depth
    knots = np.linspace(0.0, 1.0, width + 2)[1:-1]
    fixed_coef = fit_ridge(relu_features(context_x_np, knots), context_y_np, ridge=1e-6)
    fixed_pred = relu_features(query_x_np, knots) @ fixed_coef
    kernel_alpha, kernel_gamma = fit_kernel_ridge_cv(context_x_np, context_y_np, ridge=1e-6)
    kernel_pred = np.exp(-kernel_gamma * (query_x_np[:, None] - context_x_np[None, :]) ** 2) @ kernel_alpha
    for method, prediction in (
        ("jacobian_ridge", jacobian_pred),
        ("gauss_newton_1", one_step_pred),
        ("fixed_relu_ridge", fixed_pred),
        ("kernel_ridge", kernel_pred),
    ):
        if method not in methods:
            continue
        rows.append(
            {
                "depth": depth,
                "n_context": n_context,
                "seed": seed,
                "method": method,
                "query_mse": float(np.mean((prediction - query_y_np) ** 2)),
                "parameter_mse": float("nan"),
                "first_gradient_cosine_to_bp": float("nan"),
                "first_gradient_relative_norm": float("nan"),
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    result = []
    for method in sorted({str(row["method"]) for row in rows}):
        group = [row for row in rows if row["method"] == method]
        summary: dict[str, float | int | str] = {"method": method, "episodes": len(group)}
        for metric in ("query_mse", "parameter_mse", "first_gradient_cosine_to_bp", "first_gradient_relative_norm"):
            values = np.asarray([float(row[metric]) for row in group])
            if np.all(np.isnan(values)):
                summary[f"median_{metric}"] = float("nan")
                summary[f"p90_{metric}"] = float("nan")
            else:
                summary[f"median_{metric}"] = float(np.nanmedian(values))
                summary[f"p90_{metric}"] = float(np.nanquantile(values, 0.9))
        result.append(summary)
    return result


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--n-context", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed-offset", type=int, default=900000)
    parser.add_argument("--query-size", type=int, default=4096)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--state-lr", type=float, default=0.01)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--bp-lr", type=float, default=0.1)
    parser.add_argument("--bp-adam-lr", type=float, default=0.01)
    parser.add_argument("--bp-normalized-lr", type=float, default=0.01)
    parser.add_argument("--pc-lr", type=float, default=1.0)
    parser.add_argument("--pcalm-lr", type=float, default=1.0)
    parser.add_argument("--parameter-steps", type=int, default=100)
    parser.add_argument("--local-backend", choices=("autograd", "explicit"), default="explicit")
    parser.add_argument(
        "--methods",
        default="bp,bp_adam,bp_normalized,pc,pcalm,jacobian_ridge,gauss_newton_1,fixed_relu_ridge,kernel_ridge",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = CreditConfig(
        budget=args.budget or 2 * args.depth,
        state_lr=args.state_lr,
        rho=args.rho,
        alpha=args.alpha,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown = set(methods) - {
        "bp", "bp_adam", "bp_normalized", "pc", "pcalm",
        "jacobian_ridge", "gauss_newton_1", "fixed_relu_ridge", "kernel_ridge"
    }
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    rows = []
    parameter_lrs = {
        "bp": args.bp_lr,
        "bp_adam": args.bp_adam_lr,
        "bp_normalized": args.bp_normalized_lr,
        "pc": args.pc_lr,
        "pcalm": args.pcalm_lr,
    }
    for episode in range(args.episodes):
        seed = args.seed_offset + episode
        rows.extend(
            run_episode(
                args.depth,
                args.n_context,
                seed,
                config,
                parameter_lrs,
                args.parameter_steps,
                args.query_size,
                methods,
                args.local_backend,
            )
        )
    summaries = summarize(rows)
    write_csv(args.output_dir / "tent_credit_episodes.csv", rows)
    write_csv(args.output_dir / "tent_credit_summary.csv", summaries)
    metadata = vars(args) | {"resolved_budget": config.budget, "implementation_note": __doc__}
    metadata["output_dir"] = str(metadata["output_dir"])
    with (args.output_dir / "tent_credit_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    for row in summaries:
        print(
            f"{row['method']:16s} query={row['median_query_mse']:.3e} "
            f"p90={row['p90_query_mse']:.3e} "
            f"cos={row['median_first_gradient_cosine_to_bp']:.3f} "
            f"norm={row['median_first_gradient_relative_norm']:.3f}"
        )


if __name__ == "__main__":
    main()
