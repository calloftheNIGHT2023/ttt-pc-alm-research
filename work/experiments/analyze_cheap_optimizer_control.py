"""Paired PC-ALM versus Adam analysis for the frozen cheap risk mechanism."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcalm-csv", type=Path, required=True)
    parser.add_argument("--adam-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pcalm_rows = read(args.pcalm_csv)
    adam_rows = read(args.adam_csv)
    pcalm = {
        int(row["seed"]): float(row["query_mse"])
        for row in pcalm_rows
        if row["method"] == "cheap_context_risk_blend"
    }
    adam = {
        int(row["seed"]): float(row["query_mse"])
        for row in adam_rows
        if row["method"] == "cheap_context_risk_blend_adam"
    }
    seeds = sorted(set(pcalm) & set(adam))
    pc = np.asarray([pcalm[seed] for seed in seeds])
    bp = np.asarray([adam[seed] for seed in seeds])

    def metrics(values: np.ndarray) -> dict[str, float]:
        p90 = np.quantile(values, 0.9)
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(p90),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "maximum": float(np.max(values)),
            "cvar90": float(np.mean(values[values >= p90])),
        }

    result = {
        "episodes": len(seeds),
        "seed_range": [seeds[0], seeds[-1]],
        "pcalm": metrics(pc),
        "adam_bp": metrics(bp),
        "pcalm_win_rate": float(np.mean(pc < bp)),
        "mean_pcalm_minus_adam": float(np.mean(pc - bp)),
        "wilcoxon_one_sided_p_pcalm_less": float(
            wilcoxon(pc, bp, alternative="less").pvalue
        ),
        "wilcoxon_one_sided_p_adam_less": float(
            wilcoxon(bp, pc, alternative="less").pvalue
        ),
        "interpretation": "PC-ALM has lower mean and CVaR90, but Adam wins most paired episodes; no independent PC-ALM advantage is established.",
    }
    (args.output_dir / "cheap_pcalm_vs_adam.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
