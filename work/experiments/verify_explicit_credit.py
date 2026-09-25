"""Numerically verify explicit local derivatives against the autograd reference."""

from __future__ import annotations

import torch

from tent_credit_screen import (
    CreditConfig,
    bounded_params,
    local_gradient,
    local_energy,
    solve_activities,
    solve_activities_explicit,
    initial_free,
    tent_with_derivatives,
)


torch.set_default_dtype(torch.float64)


def assert_close(left: torch.Tensor, right: torch.Tensor, name: str) -> None:
    difference = float(torch.max(torch.abs(left - right)).detach())
    if not torch.allclose(left, right, atol=1e-11, rtol=1e-10):
        raise AssertionError(f"{name} mismatch; max abs difference={difference:.3e}")
    print(f"{name}: max_abs={difference:.3e}")


def main() -> None:
    generator = torch.Generator().manual_seed(20260919)
    x = 0.02 + 0.96 * torch.rand(17, generator=generator)
    a = torch.tensor(0.47, requires_grad=True)
    x_variable = x.clone().requires_grad_(True)
    value, derivative_x, derivative_a = tent_with_derivatives(x_variable, a)
    auto_x, auto_a = torch.autograd.grad(value.sum(), (x_variable, a))
    assert_close(derivative_x, auto_x, "tent dvalue/dx")
    assert_close(derivative_a.sum(), auto_a, "tent dvalue/da")

    raw = torch.tensor([-0.35, 0.2, -0.1, 0.4])
    true_raw = torch.tensor([0.4, -0.25, 0.3, -0.5])
    y = x
    for parameter in bounded_params(true_raw):
        y = torch.where(y <= parameter, y / parameter, (1.0 - y) / (1.0 - parameter))
    config = CreditConfig(budget=8, state_lr=0.1, rho=1.0, alpha=1.0)

    free0 = initial_free(raw, x)
    duals0 = [torch.zeros_like(activity) for activity in free0]
    free_auto = solve_activities(raw, x, y, free0, duals0, config, steps=3)
    free_explicit = solve_activities_explicit(raw, x, y, free0, duals0, config, steps=3)
    for index, (auto, explicit) in enumerate(zip(free_auto, free_explicit)):
        assert_close(explicit, auto, f"activity layer {index}")

    for method in ("pc", "pcalm"):
        auto = local_gradient(method, raw, x, y, config, backend="autograd")
        explicit = local_gradient(method, raw, x, y, config, backend="explicit")
        assert_close(explicit, auto, f"{method} parameter gradient")

    # Ensure the reference objective remains finite at the compared state.
    energy = local_energy(raw, x, y, free_explicit, duals0, config.rho)
    if not torch.isfinite(energy):
        raise AssertionError("local energy must be finite")
    print(f"local energy: {float(energy):.6e}")


if __name__ == "__main__":
    main()
