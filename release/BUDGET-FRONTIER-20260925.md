# Budget frontier and exact lazy execution — 25 September 2026

[中文图文报告](../results/budget_frontier/report_v1/report.md) · [冻结协议](../outputs/ttt-pc-alm-research/360_budget_frontier_protocol_v1.md) · [独立审计](../results/budget_frontier/audit_v1/summary.json)

This supplement adds a precise finite-budget condition and an implementation-level work guarantee, not a new query-risk claim or a PC-ALM-specific superiority claim.

## Verified result

For an ordered candidate at rank `j`, strict certificates must eliminate at least `j-G` preceding candidates to put it within a `G`-candidate geometry budget. Independent per-region solvers can execute only the necessary prefix and return exactly the same ordered selection as full-pool screening.

Across the exposed 32-task development set:

- 224 lazy component calls match all original selections; 896 processed state/proof arrays match bitwise.
- ALM dual / 128 steps: 73,460 → 31,939 region-response pairs, a 56.52% reduction.
- Zero credit / 128 steps: 73,578 → 32,723 pairs, a 55.53% reduction under the same enhancement.
- 547 local proofs independently checked with Fraction arithmetic; 38,961 scalar ranking conditions verified.
- The main propagation method still reaches only 19 of 47 new positive-volume candidate regions at `G=8`. A diagnostic perfect strict-infeasibility filter can reach 47, but its archived geometry labels are not available to the candidate algorithm.

Smaller batches do **not** establish wall-clock acceleration: the new lazy-dual component totals 1.497207 seconds, while an earlier full-pool component measurement was 1.101280 seconds. They are not contemporaneous repeated timing controls. Credit/pool generation costs are excluded from these component measurements. There is no new complete online fit or query score in this supplement.

The earlier [448-fit online report](../results/cross_region_online/report_v1/report.md) remains the latest predictive evidence. Its improvement is reproduced by no-credit candidate search; PC-ALM-specific competitive task benefit and large-model validation remain unestablished.

## Reproduction and package scope

Use this Git checkout (the fixed parent is `ee0907ad75ce87299d480bbcbb050ca56a0e4556`). The package includes all new component outputs, source/protocol hashes, frontiers, independent audit and checked figure. The saved support/credit/pool inputs and their geometry classifications are already present in that parent and are verified against its Git blobs. It is not a claim of complete clean-clone end-to-end research reproduction; no historical query archive or full parent trajectory is newly packaged.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python scripts/budget_frontier_snapshot_20260925.py verify
```

The research scripts use exclusive output creation. Re-running them requires a fresh, explicitly chosen output location; do not overwrite the sealed directories.

```text
work/experiments/test_budget_frontier_v1.py
work/experiments/run_budget_frontier_v1.py
work/experiments/audit_budget_frontier_v1.py
work/experiments/report_budget_frontier_v1.py
work/experiments/audit_budget_frontier_report_v1.py
```

The first image layout placed a legend over a bar; it was visually inspected and corrected. Numeric experiments were unchanged, final numeric/visual QA passed, and the earlier rendering was retained locally.
