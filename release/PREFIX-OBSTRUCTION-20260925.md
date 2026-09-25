# Prefix obstruction diagnosis — 25 September 2026

[中文图文结果](../results/prefix_obstruction/report_v1/report.md) · [执行前协议](../outputs/ttt-pc-alm-research/372_prefix_obstruction_protocol_v1.md)

The 48 saved frontiers contain 2,117 uniformly sampled prefix candidates. Independent exact certificates classify 2,090 as infeasible and 27 as positive-volume feasible, with none unresolved. Among the 16 budget-exhausted frontiers, 1,023 of 1,024 sampled candidates are infeasible even under the interval screener's outward-rounded observation band. This localizes a relaxation bottleneck; it is not a query-performance result or proof of PC-ALM superiority.

Cold local PDHG128 rejects 580 of those 1,023 infeasible sampled candidates; additional interval contraction100 rejects 45. Both are existing controls, and all screens were saved before the offline LP proposed any diagnostic certificates. The LP labels and weights never initialize the local screens. Correlated, previously exposed tasks are not treated as fresh independent confirmations.

The archive retains the original failed preflight assertion, the later array-serialization failure, all versioned sources, identical selections and saved-mask replays. The final algorithm itself was unchanged by these two harness repairs. Independent audit checks 426,680 exact constraint rows, 8,387 certificates, 819 outward-domain PDHG bounds, 27 positive points and all 48 selections. All research processes terminated before packaging.

This snapshot packages complete saved-input evidence for the new diagnostic. It does not republish all historical trajectories or claim that every historical experiment can be run from a clean clone.

```powershell
python scripts/prefix_obstruction_snapshot_20260925.py verify
```

Verification checks packaged bytes, links and frozen-source hashes; it does not rerun the scientific computation. Do not overwrite saved output directories. Fixed parent: `4a50252828eac8780f7665f9c5f710d359560b53`. The research goal remains active; history-guided certificate efficiency and end-to-end unseen-query benefit are not established by this diagnostic.
