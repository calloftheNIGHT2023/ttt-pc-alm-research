# Region-conditioned free credit — 25 September 2026

[Chinese illustrated results](../results/region_conditioned_credit/report_v1/report.md) · [Frozen mathematical protocol](../outputs/ttt-pc-alm-research/355_region_conditioned_credit_protocol_v1.md) · [Independent proof audit](../results/region_conditioned_credit/audit_v1/summary.json)

The new component optimizes a free separating credit for each fixed branch using a local analytic response and Euclidean halfspace projection. It is an established convex projection mechanism, not a novel ALM optimizer. The experiment tests whether existing ALM multipliers independently improve its finite-budget behavior.

On 32 exposed development tasks and 9,636 common regions, ALM initialization certifies 9,372 rejections, including 21 beyond C20 interval contraction. Zero initialization certifies 9,374, including 20 beyond C20. All new rejections are independently checked with exact rational arithmetic. Joint-constraint credits supplement C20, but these results do not establish an ALM-specific advantage or new query-risk improvement.

This snapshot includes **all 515 component-run output files**, including the support inputs, six initialization arrays, resulting credits, individual certificates and control outputs. It also includes the preserved first-attempt serialization failure and I/O-only repair. Original parent-trajectory archives, the original pool-generation inputs and old raw geometry audits are not included here. This is a complete saved-input component evidence archive, not a complete end-to-end TTT experiment archive.

Verify the archive bytes:

```text
python scripts/region_credit_snapshot_20260925.py verify
```

Replay the component from the packaged support inputs, with NumPy and SciPy installed, without accessing parent trajectories, query answers or geometry labels:

```text
python work/experiments/replay_region_conditioned_credit_v1.py --out /path/to/a/new-replay-summary.json
```

Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` for the recorded CPU configuration. Replay requires a new output file and checks exact repeat arrays on the recorded binary64 backend; cross-platform floating tie behavior is not claimed invariant. It also independently verifies every accepted credit certificate. The original global geometry audit and query-risk pipeline remain separate.

[Latest actual online-query report](../results/unvisited_online/report_v1/report.md) remains unchanged. No official TTT-MLP or LLM/VLM validation is claimed.
