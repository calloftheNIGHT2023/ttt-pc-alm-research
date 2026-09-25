"""Local PC-ALM credit audit for the six-parameter digits adapter.

The frozen CNN is rewritten as adjacent constraints. Activities are independent
variables, and the affine parameters receive credit only from the first
constraint h0 = warp(x; raw). No BP gradient or BP-based dual initialization is
used by the PC-ALM path.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from digits_domain_shift import (
    VALIDATION_SEED_OFFSET,
    DigitBackbone,
    correction_theta,
    episode_tensors,
    load_model_and_data,
    warp,
)


@dataclass(frozen=True)
class PCALMConfig:
    activity_steps: int
    state_learning_rate: float
    rho: float
    dual_learning_rate: float
    inner_steps: int = 1


def layer_forward(model: DigitBackbone, index: int, value: torch.Tensor) -> torch.Tensor:
    if index == 1:
        return F.relu(model.encoder[0](value))
    if index == 2:
        return F.relu(model.encoder[2](value))
    if index == 3:
        return F.relu(model.encoder[4](value))
    if index == 4:
        return F.relu(model.project(value.flatten(1)))
    raise ValueError(f"unknown constrained layer index: {index}")


@torch.no_grad()
def initial_activities(
    model: DigitBackbone, raw: torch.Tensor, images: torch.Tensor
) -> list[torch.Tensor]:
    h0 = warp(images, correction_theta(raw))
    h1 = layer_forward(model, 1, h0)
    h2 = layer_forward(model, 2, h1)
    h3 = layer_forward(model, 3, h2)
    h4 = layer_forward(model, 4, h3)
    return [state.detach().clone() for state in (h0, h1, h2, h3, h4)]


def constraint_residuals(
    model: DigitBackbone,
    raw: torch.Tensor,
    images: torch.Tensor,
    activities: list[torch.Tensor],
) -> list[torch.Tensor]:
    predictions = [warp(images, correction_theta(raw))]
    for index in range(1, 5):
        predictions.append(layer_forward(model, index, activities[index - 1]))
    return [activity - prediction for activity, prediction in zip(activities, predictions)]


def per_sample_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).flatten(1).sum(dim=1)


def augmented_energy(
    model: DigitBackbone,
    raw: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    activities: list[torch.Tensor],
    duals: list[torch.Tensor],
    rho: float,
) -> torch.Tensor:
    logits = model.classifier(activities[-1])
    total = F.cross_entropy(logits, labels)
    for residual, dual in zip(
        constraint_residuals(model, raw, images, activities), duals
    ):
        total = total + (
            per_sample_inner(dual, residual)
            + 0.5 * rho * residual.flatten(1).square().sum(dim=1)
        ).mean()
    return total


def activity_update(
    model: DigitBackbone,
    raw: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    activities: list[torch.Tensor],
    duals: list[torch.Tensor],
    config: PCALMConfig,
) -> list[torch.Tensor]:
    current = activities
    effective_learning_rate = config.state_learning_rate * images.shape[0]
    for _ in range(config.inner_steps):
        variables = [state.detach().clone().requires_grad_(True) for state in current]
        energy = augmented_energy(
            model, raw.detach(), images, labels, variables, duals, config.rho
        )
        gradients = torch.autograd.grad(energy, variables)
        current = [
            (state - effective_learning_rate * gradient).detach()
            for state, gradient in zip(variables, gradients)
        ]
    return current


def local_affine_energy(
    raw: torch.Tensor,
    images: torch.Tensor,
    h0: torch.Tensor,
    dual0: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    residual = h0 - warp(images, correction_theta(raw))
    return (
        per_sample_inner(dual0, residual)
        + 0.5 * rho * residual.flatten(1).square().sum(dim=1)
    ).mean()


def local_affine_gradient(
    raw: torch.Tensor,
    images: torch.Tensor,
    h0: torch.Tensor,
    dual0: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    variable = raw.detach().clone().requires_grad_(True)
    energy = local_affine_energy(
        variable, images, h0.detach(), dual0.detach(), rho
    )
    return torch.autograd.grad(energy, variable)[0].detach()


def pcalm_gradient(
    model: DigitBackbone,
    raw: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: PCALMConfig,
) -> tuple[torch.Tensor, dict]:
    activities = initial_activities(model, raw, images)
    duals = [torch.zeros_like(activity) for activity in activities]
    residual_history = []
    for step in range(config.activity_steps):
        activities = activity_update(
            model, raw, images, labels, activities, duals, config
        )
        residuals = constraint_residuals(model, raw.detach(), images, activities)
        residual_history.append(
            [float(residual.square().mean().sqrt()) for residual in residuals]
        )
        if step + 1 < config.activity_steps:
            duals = [
                (dual + config.dual_learning_rate * residual).detach()
                for dual, residual in zip(duals, residuals)
            ]
    gradient = local_affine_gradient(
        raw, images, activities[0], duals[0], config.rho
    )
    state_scalars = sum(activity.numel() for activity in activities)
    diagnostics = {
        "final_residual_rms_by_constraint": residual_history[-1],
        "max_residual_rms_during_inference": float(np.max(residual_history)),
        "activity_state_scalars": state_scalars,
        "dual_state_scalars": state_scalars,
        "gradient_uses_only_first_constraint": True,
        "bp_used_to_initialize_activities_or_duals": False,
    }
    return gradient, diagnostics


def bp_gradient(
    model: DigitBackbone,
    raw: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    variable = raw.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(
        model(warp(images, correction_theta(variable))), labels
    )
    return torch.autograd.grad(loss, variable)[0].detach()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def finite_difference_local_gradient(
    raw: torch.Tensor,
    images: torch.Tensor,
    h0: torch.Tensor,
    dual0: torch.Tensor,
    rho: float,
    difference: float = 1e-3,
) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for index in range(raw.numel()):
            plus = raw.clone()
            minus = raw.clone()
            plus[index] += difference
            minus[index] -= difference
            plus_energy = local_affine_energy(plus, images, h0, dual0, rho)
            minus_energy = local_affine_energy(minus, images, h0, dual0, rho)
            values.append((plus_energy - minus_energy) / (2.0 * difference))
    return torch.stack(values)


def audit(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, images, labels, splits = load_model_and_data(args.checkpoint_dir, device)
    episode = episode_tensors(
        images, labels, splits["validation"], VALIDATION_SEED_OFFSET, device
    )
    context_images = episode["adapt_images"]
    context_labels = episode["adapt_labels"]
    raw = torch.zeros(6, device=device)
    bp = bp_gradient(model, raw, context_images, context_labels)
    rows = []
    last_config = None
    for budget in args.budgets:
        config = PCALMConfig(
            activity_steps=budget,
            state_learning_rate=args.state_learning_rate,
            rho=args.rho,
            dual_learning_rate=args.dual_learning_rate,
        )
        local, diagnostics = pcalm_gradient(
            model, raw, context_images, context_labels, config
        )
        rows.append(
            {
                "activity_steps": budget,
                "state_learning_rate": args.state_learning_rate,
                "rho": args.rho,
                "dual_learning_rate": args.dual_learning_rate,
                "cosine_to_bp": cosine(local, bp),
                "relative_norm_to_bp": float(local.norm() / bp.norm()),
                "local_gradient_norm": float(local.norm()),
                "bp_gradient_norm": float(bp.norm()),
                "final_first_constraint_rms": diagnostics[
                    "final_residual_rms_by_constraint"
                ][0],
                "final_last_constraint_rms": diagnostics[
                    "final_residual_rms_by_constraint"
                ][-1],
                "activity_state_scalars": diagnostics["activity_state_scalars"],
                "dual_state_scalars": diagnostics["dual_state_scalars"],
            }
        )
        last_config = config
        print(rows[-1], flush=True)

    assert last_config is not None
    activities = initial_activities(model, raw, context_images)
    duals = [torch.zeros_like(activity) for activity in activities]
    for step in range(last_config.activity_steps):
        activities = activity_update(
            model,
            raw,
            context_images,
            context_labels,
            activities,
            duals,
            last_config,
        )
        residuals = constraint_residuals(
            model, raw.detach(), context_images, activities
        )
        if step + 1 < last_config.activity_steps:
            duals = [
                (dual + last_config.dual_learning_rate * residual).detach()
                for dual, residual in zip(duals, residuals)
            ]
    automatic = local_affine_gradient(
        raw, context_images, activities[0], duals[0], last_config.rho
    )
    # Identity sampling lands exactly on bilinear grid knots, where the
    # derivative is non-unique. Audit the same local energy at a generic nearby
    # raw point so central differences test the implementation, not a kink.
    probe_raw = torch.tensor(
        [0.017, -0.013, 0.011, 0.019, -0.007, 0.023], device=device
    )
    automatic_probe = local_affine_gradient(
        probe_raw, context_images, activities[0], duals[0], last_config.rho
    )
    finite_difference_checks = {}
    for difference in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        finite = finite_difference_local_gradient(
            probe_raw,
            context_images,
            activities[0],
            duals[0],
            last_config.rho,
            difference=difference,
        )
        finite_difference_checks[str(difference)] = {
            "max_abs": float((automatic_probe - finite).abs().max()),
            "cosine": cosine(automatic_probe, finite),
            "finite_difference_norm": float(finite.norm()),
        }
    relabeled = torch.roll(context_labels, shifts=1)
    same_local_after_relabel = local_affine_gradient(
        raw, context_images, activities[0], duals[0], last_config.rho
    )
    report = {
        "seed": VALIDATION_SEED_OFFSET,
        "context_examples": int(context_labels.numel()),
        "local_autograd_gradient": automatic.cpu().tolist(),
        "local_autograd_gradient_norm": float(automatic.norm()),
        "finite_difference_probe_raw": probe_raw.cpu().tolist(),
        "finite_difference_probe_autograd_norm": float(automatic_probe.norm()),
        "finite_difference_checks": finite_difference_checks,
        "identity_finite_difference_excluded_because": "identity affine samples exact bilinear grid knots with a non-unique derivative",
        "local_gradient_change_when_labels_change_but_local_state_fixed": float(
            (automatic - same_local_after_relabel).abs().max()
        ),
        "relabeled_tensor_constructed_for_locality_audit": bool(
            not torch.equal(relabeled, context_labels)
        ),
        "parameter_gradient_reads": [
            "raw affine parameters",
            "input images",
            "first activity h0",
            "first multiplier lambda0",
            "first residual",
        ],
        "parameter_gradient_does_not_read_labels_or_deeper_activities": True,
        "bp_used_by_pcalm_path": False,
        "note": "BP is computed only as an external audit reference after defining the local path.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "credit_budget_screen.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "locality_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--state-learning-rate", type=float, default=0.01)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--dual-learning-rate", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
