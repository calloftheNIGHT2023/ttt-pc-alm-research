"""Paired analysis for the frozen context-CV router confirmation set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-csv", type=Path, required=True)
    parser.add_argument("--shallow-csv", type=Path, required=True)
    parser.add_argument("--closed-form-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    router_rows = read_csv(args.router_csv)
    methods: dict[str, dict[int, float]] = {}
    for method in sorted({row["method"] for row in router_rows}):
        methods[method] = {
            int(row["seed"]): float(row["query_mse"])
            for row in router_rows
            if row["split"] == "locked_test" and row["method"] == method
        }
    shallow_rows = read_csv(args.shallow_csv)
    methods["adaptive_shallow_w30"] = {
        int(row["seed"]): float(row["query_mse"])
        for row in shallow_rows
        if int(row["width"]) == 30
    }
    closed_rows = read_csv(args.closed_form_csv)
    for method in ("kernel_ridge", "fixed_relu_ridge"):
        methods[method] = {
            int(row["seed"]): float(row["query_mse"])
            for row in closed_rows
            if row["method"] == method
        }

    common_seeds = sorted(set.intersection(*(set(mapping) for mapping in methods.values())))
    arrays = {
        method: np.asarray([mapping[seed] for seed in common_seeds])
        for method, mapping in methods.items()
    }
    summaries = []
    for method, values in arrays.items():
        summaries.append(
            {
                "method": method,
                "episodes": values.size,
                "mean_query_mse": float(np.mean(values)),
                "median_query_mse": float(np.median(values)),
                "p90_query_mse": float(np.quantile(values, 0.9)),
            }
        )
    candidate = arrays["context_cv_soft_blend"]
    comparisons = []
    for baseline_name in (
        "pcalm",
        "meta_feature_ridge",
        "context_cv_hard_router",
        "adaptive_shallow_w30",
        "kernel_ridge",
        "fixed_relu_ridge",
    ):
        baseline = arrays[baseline_name]
        comparisons.append(
            {
                "candidate": "context_cv_soft_blend",
                "baseline": baseline_name,
                "episodes": candidate.size,
                "candidate_win_rate": float(np.mean(candidate < baseline)),
                "mean_candidate_minus_baseline": float(np.mean(candidate - baseline)),
                "wilcoxon_one_sided_p_candidate_less": float(
                    wilcoxon(candidate, baseline, alternative="less").pvalue
                ),
            }
        )

    base_rows = [
        row
        for row in router_rows
        if row["split"] == "locked_test" and row["method"] == "pcalm"
    ]
    pcalm_cv = np.asarray([float(row["pcalm_context_cv_mse"]) for row in base_rows])
    ridge_cv = np.asarray([float(row["ridge_context_cv_mse"]) for row in base_rows])
    pcalm_query = arrays["pcalm"]
    ridge_query = arrays["meta_feature_ridge"]
    diagnostic = {
        "confirmation_seed_start": common_seeds[0],
        "confirmation_seed_end": common_seeds[-1],
        "episodes": len(common_seeds),
        "rule_frozen_before_confirmation": True,
        "soft_weight": "(ridge_cv+1e-12)/(pcalm_cv+ridge_cv+2e-12)",
        "query_targets_available_to_weight_rule": False,
        "log_cv_difference_query_difference_correlation": float(
            np.corrcoef(
                np.log10(pcalm_cv + 1e-12) - np.log10(ridge_cv + 1e-12),
                pcalm_query - ridge_query,
            )[0, 1]
        ),
        "hard_router_oracle_choice_accuracy": float(
            np.mean((pcalm_cv <= ridge_cv) == (pcalm_query <= ridge_query))
        ),
        "multiple_comparison_warning": "one-sided paired tests are descriptive and are not multiplicity-adjusted",
    }
    write_csv(args.output_dir / "confirmation_summary.csv", summaries)
    write_csv(args.output_dir / "confirmation_paired_comparisons.csv", comparisons)
    (args.output_dir / "confirmation_diagnostics.json").write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8"
    )
    for row in summaries:
        print(row)
    for row in comparisons:
        print(row)
    print(diagnostic)


if __name__ == "__main__":
    main()
