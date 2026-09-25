"""Paired analysis for the frozen two-fit context-risk confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--shallow-csv", type=Path, required=True)
    parser.add_argument("--closed-form-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = read(args.candidate_csv)
    mappings = {
        method: {
            int(row["seed"]): float(row["query_mse"])
            for row in candidate_rows
            if row["method"] == method
        }
        for method in sorted({row["method"] for row in candidate_rows})
    }
    shallow_rows = read(args.shallow_csv)
    mappings["adaptive_shallow_w30"] = {
        int(row["seed"]): float(row["query_mse"])
        for row in shallow_rows
        if int(row["width"]) == 30
    }
    closed_rows = read(args.closed_form_csv)
    for method in ("kernel_ridge", "fixed_relu_ridge"):
        mappings[method] = {
            int(row["seed"]): float(row["query_mse"])
            for row in closed_rows
            if row["method"] == method
        }
    seeds = sorted(set.intersection(*(set(mapping) for mapping in mappings.values())))
    arrays = {
        method: np.asarray([mapping[seed] for seed in seeds])
        for method, mapping in mappings.items()
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
    candidate = arrays["cheap_context_risk_blend"]
    comparisons = []
    for baseline_name, baseline in arrays.items():
        if baseline_name == "cheap_context_risk_blend":
            continue
        comparisons.append(
            {
                "candidate": "cheap_context_risk_blend",
                "baseline": baseline_name,
                "episodes": candidate.size,
                "candidate_win_rate": float(np.mean(candidate < baseline)),
                "mean_candidate_minus_baseline": float(np.mean(candidate - baseline)),
                "wilcoxon_one_sided_p_candidate_less": float(
                    wilcoxon(candidate, baseline, alternative="less").pvalue
                ),
            }
        )
    write(args.output_dir / "cheap_confirmation_summary.csv", summaries)
    write(args.output_dir / "cheap_confirmation_paired_comparisons.csv", comparisons)
    diagnostics = {
        "seed_range": [seeds[0], seeds[-1]],
        "episodes": len(seeds),
        "candidate_pc_activity_sweeps": 1600,
        "previous_four_fold_activity_sweeps": 4000,
        "activity_sweep_reduction_fraction": 0.6,
        "selection_frozen_before_confirmation": True,
        "multiple_comparison_warning": "descriptive one-sided paired tests; no multiplicity adjustment",
    }
    (args.output_dir / "cheap_confirmation_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    for row in summaries:
        print(row)
    for row in comparisons:
        print(row)


if __name__ == "__main__":
    main()
