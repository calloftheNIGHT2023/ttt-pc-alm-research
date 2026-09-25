"""Combine the locked holdout runs and compute paired comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--extra", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(args.primary), pd.read_csv(args.controls)]
    for path in args.extra:
        extra = pd.read_csv(path)
        if "method" not in extra and "width" in extra:
            extra["method"] = "shallow_adaptive_w" + extra["width"].astype(str)
        frames.append(extra)
    frame = pd.concat(frames, ignore_index=True)
    pivot = frame.pivot(index="seed", columns="method", values="query_mse").sort_index()
    if pivot.isna().any().any():
        raise ValueError("all methods must cover exactly the same holdout seeds")
    reference = pivot["pcalm"]
    rows = []
    for method in pivot.columns:
        values = pivot[method]
        if method == "pcalm":
            p_value = float("nan")
            win_rate = float("nan")
        else:
            p_value = float(wilcoxon(reference, values, alternative="less").pvalue)
            win_rate = float((reference < values).mean())
        rows.append(
            {
                "method": method,
                "episodes": int(values.size),
                "mean_query_mse": float(values.mean()),
                "median_query_mse": float(values.median()),
                "p90_query_mse": float(values.quantile(0.9)),
                "pcalm_win_rate": win_rate,
                "wilcoxon_one_sided_p_pcalm_less": p_value,
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_query_mse")
    summary.to_csv(args.output_dir / "final_holdout_comparison.csv", index=False)
    metadata = {
        "primary": str(args.primary),
        "controls": str(args.controls),
        "extra": [str(path) for path in args.extra],
        "paired_seeds": [int(value) for value in pivot.index],
        "test": "paired one-sided Wilcoxon signed-rank; alternative PC-ALM query MSE is lower",
        "multiplicity_correction": None,
        "interpretation_warning": "exploratory controlled-task evidence, not a confirmatory paper-level result",
    }
    with (args.output_dir / "final_holdout_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
