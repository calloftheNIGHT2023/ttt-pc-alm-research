# Active regional multiplier feedback — 25 September 2026

[中文图文与九组完整对照](../results/regional_alm/report_v1/report.md) · [执行前数学与实验协议](../outputs/ttt-pc-alm-research/376_regional_alm_protocol_v1.md)

This component experiment provides positive evidence for active feedback inside the currently tested region, rather than reusing a historical direction from another nonlinear trajectory. On the same previously sampled failed frontiers, the fixed primary regional-active128 certifies 958 of 1,023 infeasible candidates. Passive accumulation and instantaneous residual each certify 343; cold PDHG128 certifies 580 and identical-box-initialization PDHG128 certifies 574. Complete mean component times are 28.168, 27.703, 26.724, 26.729 and 26.313 milliseconds respectively.

All nine configurations and stronger 1,024-step controls remain visible. Checkpoint curves identify finite-cost regimes worth testing in a full join, but are not independently timed shorter runs or end-to-end prediction speedups. All methods can propose instantaneous residual certificates; no LP, global BP, suffix observations, query targets or historical-state generation enter the candidate.

The local proximal quadratic blocks are derived before execution. Preflight checks 2,484 independent scalar updates, 81 block-descent inequalities and 27 known-feasible retention cases. All 432 formal calls completed without exceptions. Independent audit verifies 12,070 exact outward-domain certificates, 1,008 identical-initial-state arrays, 144 passive/instant primal-state matches, 7,536 checkpoints, 192 long/short prefixes and 243 positive-sample retention checks.

The report and inspected figure were ready by 11:59:55 EDT; Git publication followed. This is a positive developmental mechanism/component result, not proof of query-risk benefit, novelty of generic ALM, universal superiority or completion of the paper. The next gate is a full online budget-and-readout comparison, followed by new frozen tasks if competitive.

```powershell
python scripts/regional_alm_snapshot_20260925.py verify
```

The snapshot contains complete new saved-input component evidence and checks byte/source consistency, not an automatic scientific rerun or full historical clean-clone reproduction. Fixed parent: `5a9b6eaa6052c3ede1b298fafb280fb238a2f7e0`. Research goal remains active.
