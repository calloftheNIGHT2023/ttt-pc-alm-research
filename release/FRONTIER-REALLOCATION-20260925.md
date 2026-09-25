# Frontier budget reallocation — 25 September 2026

[中文图文与全部对照](../results/frontier_online/report_v1/report.md) · [组件协议](../outputs/ttt-pc-alm-research/362_frontier_reallocation_protocol_v1.md) · [在线协议](../outputs/ttt-pc-alm-research/363_frontier_online_protocol_v1.md)

This package is gated on completed prediction, scoring, independent audit, attribution and figure QA. The report is the authoritative numerical conclusion; the fixed primary is `frontier_dual_frontier_g8`, never a post-hoc best method.

The completed run contains 544 new full fits, 5,376 scored predictors and zero execution failures. Primary MSE is **0.0514310967**, versus **0.0515808997** for the same-credit uniform schedule. However, the residual-credit frontier control reproduces the primary's particles, allocations and predictions bitwise on all 32 tasks. Zero-credit frontier MSE is **0.0514278705**, concurrent Adam240 is **0.0512499136**, and direct C20-all is **0.0504878776**. The result supports a finite-budget scheduling improvement, not independent multiplier superiority. The primary takes 1.277717 mean full-fit seconds versus 1.216888 for C20-all and 0.699744 for Adam240 (single interleaved measurements, not repeated timing inference).

## Mechanism and scope

Each region follows the original independent local credit trajectory. The frontier rule first preserves the original 128-step candidate prefix, then spends the remaining `128 × candidate_count` response budget on the earliest eight unresolved regions. The uniform control spends the same ceiling continuing all unresolved regions. Only strict exact infeasibility certificates prune a region. Global geometry and BP are prohibited inside screening; the explicitly labelled BP-credit control is separate, and downstream global geometry is charged in complete fit time.

On 384 saved-input component calls, the dual frontier rule reaches 25 positive-volume regions instead of 19 under uniform continuation. Zero reaches 24 and residual reaches 25 under the same frontier enhancement. This establishes a scheduling/coverage effect, not ALM-specific predictive superiority. Component screening costs increase because of small-batch continuation. The online report separates these questions using 17 current configurations plus 151 historical methods; historical timing is not treated as concurrent timing.

The primary scientific objective remains unfinished unless same-enhancement non-multiplier controls, strong regression/heads, same-parameter optimizers, actual resource costs and fresh-task evidence jointly establish independent benefit. Official TTT-MLP/LLM/VLM validation has not been performed in this supplement.

## Reproduction boundary

The fixed parent is `289800a3f948900b480aff4e647a1af6ce7e3326`. The snapshot includes all new component outputs; the online archive is compact (sealed manifests, complete risk matrix/comparisons, mechanisms and attribution), not every predictor particle and geometry log. Saved parent support/credit/candidate inputs are verified against fixed-parent Git blobs. No clean-clone complete end-to-end scientific reproduction is claimed.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python scripts/frontier_reallocation_snapshot_20260925.py verify
```

Research scripts use exclusive output creation. Preserve sealed directories; reproduction runs need separately chosen output locations.

- `test_frontier_reallocation_v1.py`, `run_frontier_reallocation_v1.py`, `audit_frontier_reallocation_v1.py`
- `run_frontier_online_v1.py --stage preflight` then `--stage development`
- `evaluate_frontier_online_v1.py`, `audit_frontier_online_v1.py`, `audit_frontier_attribution_v1.py`
- `report_frontier_online_v1.py`, `audit_frontier_report_v1.py`

All scripts are under `work/experiments`. Inspect the report's numeric and visual receipts before using its figures or claims.
