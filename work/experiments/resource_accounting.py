"""Write transparent state-accounting tables for the locked L=4 experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-context", type=int, default=16)
    parser.add_argument("--parameter-steps", type=int, default=100)
    parser.add_argument("--activity-budget", type=int, default=8)
    parser.add_argument("--shallow-width", type=int, default=30)
    parser.add_argument("--shallow-steps", type=int, default=500)
    parser.add_argument("--shallow-restarts", type=int, default=3)
    parser.add_argument("--ttt-head-dim", type=int, default=8)
    parser.add_argument("--ttt-heads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    depth = args.depth
    n = args.n_context
    free_activity = n * (depth - 1)
    ttt_fast = args.ttt_heads * (8 * args.ttt_head_dim**2 + 5 * args.ttt_head_dim)
    shallow_fast = 2 * args.shallow_width + 2

    rows = [
        {
            "method": "internal_pcalm",
            "learned_episode_parameters": depth,
            "prediction_state_scalars": depth,
            "optimizer_or_credit_state_scalars": 2 * free_activity,
            "training_matrix_scalars": 0,
            "passes_or_sweeps": args.parameter_steps * args.activity_budget,
            "count_scope": "activities and multipliers are transient per parameter update",
        },
        {
            "method": "internal_pc",
            "learned_episode_parameters": depth,
            "prediction_state_scalars": depth,
            "optimizer_or_credit_state_scalars": free_activity,
            "training_matrix_scalars": 0,
            "passes_or_sweeps": args.parameter_steps * args.activity_budget,
            "count_scope": "free hidden activities; zero dual arrays are not logically required",
        },
        {
            "method": "internal_bp_sgd",
            "learned_episode_parameters": depth,
            "prediction_state_scalars": depth,
            "optimizer_or_credit_state_scalars": n * depth,
            "training_matrix_scalars": 0,
            "passes_or_sweeps": args.parameter_steps,
            "count_scope": "activation count is a lower bound; autograd implementation scratch excluded",
        },
        {
            "method": "internal_bp_adam",
            "learned_episode_parameters": depth,
            "prediction_state_scalars": depth,
            "optimizer_or_credit_state_scalars": 2 * depth + n * depth,
            "training_matrix_scalars": 0,
            "passes_or_sweeps": args.parameter_steps,
            "count_scope": "two Adam moments plus activation lower bound; scratch excluded",
        },
        {
            "method": "fixed_relu_ridge_W12",
            "learned_episode_parameters": 3 * depth + 2,
            "prediction_state_scalars": 3 * depth + 2,
            "optimizer_or_credit_state_scalars": 0,
            "training_matrix_scalars": (3 * depth + 2) ** 2,
            "passes_or_sweeps": 1,
            "count_scope": "fixed knots are shared prior; Gram matrix counted during solve",
        },
        {
            "method": "rbf_kernel_ridge",
            "learned_episode_parameters": n,
            "prediction_state_scalars": 2 * n,
            "optimizer_or_credit_state_scalars": 0,
            "training_matrix_scalars": n**2,
            "passes_or_sweeps": 21,
            "count_scope": "support keys plus alpha; 21 gamma candidates use leave-one-out score",
        },
        {
            "method": f"adaptive_shallow_W{args.shallow_width}",
            "learned_episode_parameters": shallow_fast,
            "prediction_state_scalars": shallow_fast,
            "optimizer_or_credit_state_scalars": args.shallow_restarts * 3 * shallow_fast,
            "training_matrix_scalars": 0,
            "passes_or_sweeps": args.shallow_steps * args.shallow_restarts,
            "count_scope": "three live restarts, each with parameters and two Adam moments",
        },
        {
            "method": "official_ttt_mlp_d8_h1_smoke",
            "learned_episode_parameters": ttt_fast,
            "prediction_state_scalars": ttt_fast,
            "optimizer_or_credit_state_scalars": ttt_fast,
            "training_matrix_scalars": 0,
            "passes_or_sweeps": n,
            "count_scope": "W1,b1,W2,b2 plus equal-sized gradient carry; mechanism audit only",
        },
        {
            "method": "context_cv_soft_blend_pcalm_meta_ridge32",
            "learned_episode_parameters": depth + 32 + 1,
            "prediction_state_scalars": depth + 32 + 1,
            "optimizer_or_credit_state_scalars": 2 * free_activity,
            "training_matrix_scalars": 32**2,
            "passes_or_sweeps": 5 * args.parameter_steps * args.activity_budget + 5,
            "count_scope": "four context folds plus full fit: 4000 PC-ALM sweeps and five ridge solves; sequential peak state shown",
        },
        {
            "method": "cheap_risk_blend_pcalm_meta_ridge32",
            "learned_episode_parameters": depth + 32 + 1,
            "prediction_state_scalars": depth + 32 + 1,
            "optimizer_or_credit_state_scalars": 2 * free_activity,
            "training_matrix_scalars": 32**2,
            "passes_or_sweeps": 2 * args.parameter_steps * args.activity_budget + 2,
            "count_scope": "one fixed 12-point calibration fit plus one warm 16-point fit: 1600 PC-ALM sweeps and two ridge solves",
        },
    ]

    csv_path = args.output_dir / "resource_accounting.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = vars(args) | {
        "output_dir": str(args.output_dir),
        "exclusions": "inputs, targets, predictions, temporary elementwise derivatives, framework allocator overhead",
        "warning": "state counts are not FLOP or wall-clock measurements",
    }
    with (args.output_dir / "resource_accounting_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(csv_path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
