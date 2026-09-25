"""Create a publication-style summary figure for the final synthetic results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--pcalm-csv", type=Path, required=True)
    parser.add_argument("--adam-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = {row["method"]: row for row in read(args.summary_csv)}
    order = [
        "cheap_context_risk_blend",
        "full_context_pcalm_200_steps",
        "meta_feature_ridge",
        "adaptive_shallow_w30",
        "kernel_ridge",
    ]
    labels = [
        "Risk mixture\n(2 fits)",
        "PC-ALM\n200 steps",
        "Meta ridge",
        "Shallow\nW=30",
        "RBF KRR",
    ]
    means = np.asarray([float(summary_rows[name]["mean_query_mse"]) for name in order])
    p90 = np.asarray([float(summary_rows[name]["p90_query_mse"]) for name in order])

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
    pc_values = np.asarray([pcalm[seed] for seed in seeds])
    adam_values = np.asarray([adam[seed] for seed in seeds])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.35), constrained_layout=True)
    axis = axes[0]
    x = np.arange(len(order))
    width = 0.36
    axis.bar(x - width / 2, means, width, label="Mean", color="#2463A9")
    axis.bar(x + width / 2, p90, width, label="90th percentile", color="#F28E2B")
    axis.set_xticks(x, labels, fontsize=9)
    axis.set_ylabel("Unseen-query MSE")
    axis.set_title("A. Frozen 128-episode confirmation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    for position, value in zip(x - width / 2, means):
        axis.text(position, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    axis = axes[1]
    axis.scatter(adam_values, pc_values, s=22, alpha=0.68, color="#3A7D44", edgecolors="none")
    limit = max(float(np.max(adam_values)), float(np.max(pc_values))) * 1.05
    axis.plot([0, limit], [0, limit], linestyle="--", color="#555555", linewidth=1)
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Adam-BP risk mixture MSE")
    axis.set_ylabel("PC-ALM risk mixture MSE")
    axis.set_title("B. Same mechanism, different inner credit")
    axis.grid(alpha=0.2)
    axis.text(
        0.04,
        0.96,
        "PC-ALM mean 0.0262 vs Adam 0.0303\nAdam wins 68.0% of episodes",
        transform=axis.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9, "pad": 4},
    )
    fig.suptitle(
        "Context-risk mixing reduces tail failures, but PC-ALM is not uniquely required",
        fontsize=13,
    )
    png_path = args.output_dir / "final_synthetic_results.png"
    pdf_path = args.output_dir / "final_synthetic_results.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
