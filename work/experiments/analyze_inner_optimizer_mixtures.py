"""Compare PC-ALM risk mixture with matched alternative inner optimizers."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--pcalm-router-csv", type=Path, required=True)
    parser.add_argument("--alternative-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pcalm_rows = read(args.pcalm_router_csv)
    alternative_rows = read(args.alternative_csv)
    mappings: dict[tuple[str, str], dict[int, float]] = {
        ("pcalm", "inner_only"): {
            int(row["seed"]): float(row["query_mse"])
            for row in pcalm_rows
            if row["method"] == "pcalm"
        },
        ("pcalm", "context_cv_soft_blend"): {
            int(row["seed"]): float(row["query_mse"])
            for row in pcalm_rows
            if row["method"] == "context_cv_soft_blend"
        },
    }
    for method in sorted({row["inner_method"] for row in alternative_rows}):
        for result in ("inner_only", "context_cv_soft_blend"):
            mappings[(method, result)] = {
                int(row["seed"]): float(row["query_mse"])
                for row in alternative_rows
                if row["inner_method"] == method and row["result"] == result
            }
    seeds = sorted(mappings[("pcalm", "inner_only")])
    arrays = {
        key: np.asarray([mapping[seed] for seed in seeds])
        for key, mapping in mappings.items()
    }
    summaries = []
    for (method, result), values in arrays.items():
        summaries.append(
            {
                "inner_method": method,
                "result": result,
                "episodes": values.size,
                "mean_query_mse": float(np.mean(values)),
                "median_query_mse": float(np.median(values)),
                "p90_query_mse": float(np.quantile(values, 0.9)),
            }
        )
    comparisons = []
    for result in ("inner_only", "context_cv_soft_blend"):
        pcalm = arrays[("pcalm", result)]
        for method in ("bp", "bp_adam", "bp_normalized", "pc"):
            other = arrays[(method, result)]
            comparisons.append(
                {
                    "result": result,
                    "left_method": "pcalm",
                    "right_method": method,
                    "episodes": pcalm.size,
                    "pcalm_win_rate": float(np.mean(pcalm < other)),
                    "mean_pcalm_minus_other": float(np.mean(pcalm - other)),
                    "wilcoxon_one_sided_p_pcalm_less": float(
                        wilcoxon(pcalm, other, alternative="less").pvalue
                    ),
                }
            )
    write(args.output_dir / "inner_optimizer_all_summaries.csv", summaries)
    write(args.output_dir / "pcalm_vs_inner_optimizer_comparisons.csv", comparisons)
    for row in summaries:
        print(row)
    for row in comparisons:
        print(row)


if __name__ == "__main__":
    main()
