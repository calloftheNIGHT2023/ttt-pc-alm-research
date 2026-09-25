# Historical multiplier initialization — 25 September 2026

[中文图文与完整九组对照](../results/history_certificate/report_v1/report.md) · [执行前协议](../outputs/ttt-pc-alm-research/374_history_certificate_protocol_v1.md)

All 432 component calls completed and were sealed before the archived feasibility labels were read for audit. Every method saw only the same selected prefix, not the suffix of its original 24-observation task. Historical-state generation is charged independently for each call.

On 16 previously budget-exhausted frontiers, the fixed primary `alm_dual128` certifies 675 of 1,023 exact infeasible candidates. Identical ALM primal state with zero credit certifies 763; ordinary-PC state with zero credit certifies 741 at lower mean component cost. Cold128 certifies 580, while the predeclared cold1024 trajectory reaches 967 at step256 and 1,019 at step1024. Checkpoint costs are not presented as independently timed shorter fits.

This tests and rejects the claimed aggregate benefit of this particular historical initialization against its full controls. It does not invalidate all multiplier mechanisms, nor does it supply a new query-risk result. The next mechanism remains subject to independent controls and full online prediction confirmation.

The independent audit checks 28,364 scalar credit entries, 8,777 exact outward-domain certificates, 2,688 same-primal-state arrays, 3,504 checkpoints, 243 positive-sample retention checks, and 48 cold-prefix replays. The package contains complete new component states and proofs, preflight evidence, audited figures and source hashes. It does not claim full historical clean-clone scientific reproduction.

```powershell
python scripts/history_certificate_snapshot_20260925.py verify
```

This checks saved bytes and frozen-source consistency, not a scientific rerun. Fixed parent: `9c1da986ce78db4b58bf9a25f625f9f60c9a05e8`. Research goal remains active.
