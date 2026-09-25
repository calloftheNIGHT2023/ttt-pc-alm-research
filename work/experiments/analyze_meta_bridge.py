"""Aggregate outer-training seeds and compare with the locked PC-ALM replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_query_mse": float(np.mean(values)),
        "median_query_mse": float(np.median(values)),
        "p90_query_mse": float(np.quantile(values, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--pcalm-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_run: dict[tuple[str, str], dict[int, float]] = {}
    run_rows = []
    for directory in args.bridge_dirs:
        metadata = json.loads((directory / "meta_bridge_metadata.json").read_text(encoding="utf-8"))
        run_name = f"train_seed_{metadata['train_seed']}"
        rows = read_rows(directory / "meta_bridge_episodes.csv")
        for method in sorted({row["method"] for row in rows}):
            mapping = {
                int(row["seed"]): float(row["query_mse"])
                for row in rows
                if row["method"] == method
            }
            by_run[(run_name, method)] = mapping
            summary = statistics(np.asarray(list(mapping.values())))
            run_rows.append({"outer_run": run_name, "method": method, "episodes": len(mapping), **summary})

    pcalm_rows = read_rows(args.pcalm_csv)
    pcalm = {
        int(row["seed"]): float(row["query_mse"])
        for row in pcalm_rows
        if row["method"] == "pcalm"
    }
    seeds = sorted(pcalm)
    methods = sorted({method for _, method in by_run})
    outer_runs = sorted({run for run, _ in by_run})
    averaged: dict[str, np.ndarray] = {}
    episode_rows = []
    for method in methods:
        matrix = np.asarray(
            [[by_run[(run, method)][seed] for seed in seeds] for run in outer_runs]
        )
        averaged[method] = np.mean(matrix, axis=0)
        for index, seed in enumerate(seeds):
            episode_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "outer_seed_mean_query_mse": float(averaged[method][index]),
                    "outer_seed_min_query_mse": float(np.min(matrix[:, index])),
                    "outer_seed_max_query_mse": float(np.max(matrix[:, index])),
                }
            )
    pcalm_values = np.asarray([pcalm[seed] for seed in seeds])
    averaged["pcalm"] = pcalm_values
    for index, seed in enumerate(seeds):
        episode_rows.append(
            {
                "seed": seed,
                "method": "pcalm",
                "outer_seed_mean_query_mse": float(pcalm_values[index]),
                "outer_seed_min_query_mse": float(pcalm_values[index]),
                "outer_seed_max_query_mse": float(pcalm_values[index]),
            }
        )

    aggregate_rows = []
    for method, values in averaged.items():
        aggregate_rows.append({"method": method, "episodes": values.size, **statistics(values)})

    comparisons = []
    pairs = [
        ("official_ttt_meta", "official_ttt_no_adapt"),
        ("pcalm", "official_ttt_meta"),
        ("pcalm", "meta_feature_ridge"),
    ]
    for left_name, right_name in pairs:
        left = averaged[left_name]
        right = averaged[right_name]
        comparisons.append(
            {
                "left_method": left_name,
                "right_method": right_name,
                "episodes": left.size,
                "left_win_rate": float(np.mean(left < right)),
                "mean_left_minus_right": float(np.mean(left - right)),
                "wilcoxon_one_sided_p_left_less": float(
                    wilcoxon(left, right, alternative="less").pvalue
                ),
            }
        )

    write_csv(args.output_dir / "outer_seed_run_summaries.csv", run_rows)
    write_csv(args.output_dir / "outer_seed_averaged_episodes.csv", episode_rows)
    write_csv(args.output_dir / "outer_seed_aggregate_summary.csv", aggregate_rows)
    write_csv(args.output_dir / "outer_seed_paired_comparisons.csv", comparisons)
    note = {
        "outer_runs": outer_runs,
        "locked_episode_seeds": [seeds[0], seeds[-1]],
        "outer_seed_aggregation": "average each locked episode over three outer-training seeds",
        "statistical_warning": "paired tests after outer-seed averaging are exploratory; three outer seeds do not support a confirmatory outer-training variance claim",
        "main_falsification": "meta-trained fixed features plus closed-form ridge match PC-ALM mean risk and improve its 90th-percentile risk",
    }
    (args.output_dir / "analysis_notes.json").write_text(
        json.dumps(note, indent=2), encoding="utf-8"
    )
    for row in aggregate_rows:
        print(row)
    for row in comparisons:
        print(row)


if __name__ == "__main__":
    main()
