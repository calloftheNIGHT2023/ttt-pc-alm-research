"""Read-only secondary analysis of the locked digits confirmation run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(20_000, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.confirmation_dir / "confirmation_episodes.csv")
    seeds = sorted({int(row["seed"]) for row in rows})
    methods = sorted({row["method"] for row in rows})
    metrics = ("query_brier", "query_error", "query_nll")
    arrays = {
        metric: {
            method: np.asarray(
                [
                    float(next(row[metric] for row in rows if int(row["seed"]) == seed and row["method"] == method))
                    for seed in seeds
                ]
            )
            for method in methods
        }
        for metric in metrics
    }
    rng = np.random.default_rng(20260924)
    comparisons = []
    candidate = "internal_full_equal_compute"
    for metric in metrics:
        for baseline in methods:
            if baseline == candidate:
                continue
            difference = arrays[metric][candidate] - arrays[metric][baseline]
            low, high = bootstrap_mean_ci(difference, rng)
            try:
                p_less = float(wilcoxon(arrays[metric][candidate], arrays[metric][baseline], alternative="less").pvalue)
            except ValueError:
                p_less = 1.0
            comparisons.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "episodes": difference.size,
                    "candidate_win_rate": float(np.mean(difference < 0.0)),
                    "mean_candidate_minus_baseline": float(difference.mean()),
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "wilcoxon_one_sided_p_candidate_less": p_less,
                }
            )
    write_csv(args.output_dir / "internal_full_paired_comparisons.csv", comparisons)

    representative = [
        next(row for row in rows if int(row["seed"]) == seed and row["method"] == "context_risk_blend")
        for seed in seeds
    ]
    weights = np.asarray([float(row["internal_weight"]) for row in representative])
    calibration_difference = np.asarray(
        [float(row["internal_calibration_brier"]) - float(row["ridge_calibration_brier"]) for row in representative]
    )
    query_difference = arrays["query_brier"]["internal_two_stage"] - arrays["query_brier"]["frozen_feature_ridge"]
    correlation = spearmanr(calibration_difference, query_difference)
    router = {
        "episodes": len(seeds),
        "mean_internal_weight": float(weights.mean()),
        "median_internal_weight": float(np.median(weights)),
        "p10_internal_weight": float(np.quantile(weights, 0.1)),
        "p90_internal_weight": float(np.quantile(weights, 0.9)),
        "episodes_internal_query_better_than_ridge": int(np.sum(query_difference < 0.0)),
        "episodes_calibration_says_internal_better": int(np.sum(calibration_difference < 0.0)),
        "calibration_query_advantage_spearman_r": float(correlation.statistic),
        "calibration_query_advantage_spearman_p": float(correlation.pvalue),
        "mean_blend_minus_two_stage_internal_brier": float(
            np.mean(arrays["query_brier"]["context_risk_blend"] - arrays["query_brier"]["internal_two_stage"])
        ),
        "mean_blend_minus_full_internal_brier": float(
            np.mean(arrays["query_brier"]["context_risk_blend"] - arrays["query_brier"]["internal_full_equal_compute"])
        ),
    }
    (args.output_dir / "risk_weight_diagnostics.json").write_text(
        json.dumps(router, indent=2), encoding="utf-8"
    )

    summary_rows = read_csv(args.confirmation_dir / "confirmation_summary.csv")
    selected = [
        "internal_full_equal_compute",
        "context_risk_blend",
        "shallow_residual_head",
        "closed_form_tangent",
        "rbf_kernel_ridge",
        "geometric_bank_ridge",
        "frozen_feature_ridge",
        "no_adaptation",
    ]
    summary_map = {row["method"]: row for row in summary_rows}
    labels = [
        "internal affine\n(6 parameters)",
        "risk blend",
        "shallow head\n(2,410 parameters)",
        "closed-form\ntangent",
        "RBF KRR",
        "25-view bank\nridge",
        "feature ridge",
        "no adaptation",
    ]
    means = [float(summary_map[method]["mean_query_brier"]) for method in selected]
    p90s = [float(summary_map[method]["p90_query_brier"]) for method in selected]
    errors = [100.0 * float(summary_map[method]["mean_query_error"]) for method in selected]
    x = np.arange(len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    width = 0.38
    axes[0].bar(x - width / 2, means, width, label="mean Brier", color="#2b6cb0")
    axes[0].bar(x + width / 2, p90s, width, label="90th percentile", color="#90cdf4")
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].set_ylabel("query Brier (lower is better)")
    axes[0].set_title("A. Locked 128-episode confirmation")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x, errors, color=["#2f855a"] + ["#a0aec0"] * (len(x) - 1))
    axes[1].set_xticks(x, labels, rotation=28, ha="right")
    axes[1].set_ylabel("query classification error (%)")
    axes[1].set_title("B. Accuracy agrees with Brier")
    axes[1].grid(axis="y", alpha=0.2)
    for position, value in zip(x, errors):
        axes[1].text(position, value + 0.35, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Shared affine domain shift on real handwritten digits", fontsize=14)
    fig.savefig(args.output_dir / "digits_confirmation.png", dpi=220)
    fig.savefig(args.output_dir / "digits_confirmation.pdf")
    plt.close(fig)

    headline = {
        "best_method": candidate,
        "best_mean_brier": float(np.mean(arrays["query_brier"][candidate])),
        "best_mean_error": float(np.mean(arrays["query_error"][candidate])),
        "relative_brier_reduction_vs_shallow": float(
            1.0 - np.mean(arrays["query_brier"][candidate]) / np.mean(arrays["query_brier"]["shallow_residual_head"])
        ),
        "relative_brier_reduction_vs_best_closed_form": float(
            1.0 - np.mean(arrays["query_brier"][candidate]) / np.mean(arrays["query_brier"]["closed_form_tangent"])
        ),
        "risk_mixture_transferred": False,
        "router": router,
    }
    (args.output_dir / "secondary_analysis.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
