# Active regional feedback in full candidate search — 25 September 2026

[中文完整图文与36组结果](../results/regional_prefix_join/report_v1/report.md) · [执行前协议](../outputs/ttt-pc-alm-research/378_regional_join_protocol_v1.md) · [状态口径核对](../outputs/ttt-pc-alm-research/378_state_accounting_note_v1.md)

The local active feedback component is now integrated into actual full candidate search. All 576 development calls completed without execution exceptions: 16 reused n=24 tasks, two support orders, BFS and batched DFS, and nine configurations. The fixed primary active128 completes 16/16 tasks in every setting. C5/C20 without extra screening completes 9/16 in observed order and 13/16 in farthest-x order; passive128 completes 13/16 throughout.

For BFS/observed, active128 averages 0.505 seconds versus 1.229 for no extra screening. However, cold PDHG512 also completes 16/16 and averages 0.419 seconds; same-box PDHG512 averages 0.433 seconds. Both stronger 512-step controls are faster than the fixed primary in every setting. This supports real integration of the active-feedback mechanism, not independent superiority over all matched strong search controls.

Independent audit passed: 13,824 exact singleton languages; 8,586,098 combination replays; 42,228 contraction masks; 85,310 exact outward-domain pruning certificates; 576 feasible-reference output-or-pending-subtree checks; and accounting for every search. Preflight reproduces all 117 saved-array outputs of the original nine solver configurations and verifies that 256/512-step controls truly execute those steps.

Interrupted searches retain unresolved subtrees and any completed candidates; an empty candidate output is not treated as an empty true posterior. Every method has the same per-schedule rules. The documented BFS/DFS state-threshold accounting difference prevents claiming a strictly identical cross-schedule peak-memory cap. Full search timings include guard setup, language construction, ordering, expansion, screening and proof-archive copies; final global geometry, posterior readout, disk compression and offline audit are excluded.

No query targets enter this experiment, and no new query-risk result or paper completion is claimed. The next candidate should test the tradeoff between stronger early certification and total tree work, with certified-row retirement available to every control. The original active128 result must remain visible. Generic ALM, PDHG and DFS are not claimed as new algorithms.

```powershell
python scripts/regional_join_snapshot_20260925.py verify
```

This release contains the full newly saved search arrays and proofs, checks source/byte consistency, and does not claim automatic end-to-end reproduction of all historical experiments from a clean clone. Fixed parent: `ce30b7cfccdf1b54c67c99b6e4d4a6870124257a`. Research goal remains active.
