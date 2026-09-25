# Strict-certificate retirement in full candidate search

Stage 380–381 investigates whether stronger local regional feedback can reduce full-search cost after **all controls** receive the same strict-certificate row-retirement optimization.

The fixed development primary is `regional_active256`. The experiment includes all 17 predeclared configurations: no additional local solver, plus active/passive regional updates and cold/box PDHG at 128/256/512/1024 steps. The 16 reused n24 tasks and two scheduling/ordering settings give 544 calls. This is not a new blind-query confirmation.

- [Complete results, all controls, actual work and figure](../results/retired_region_join/report_v1/report.md)
- [Execution-before-results protocol](../outputs/ttt-pc-alm-research/380_certified_retirement_protocol_v1.md)
- [Preflight invariants](../results/retired_region_join/preflight_v1/summary.json)
- [All sealed search calls](../results/retired_region_join/development_v1/rows.json)
- [Independent exact proof, search and live-work audit](../results/retired_region_join/audit_v1/summary.json)
- [Numeric report QA](../results/retired_region_join/report_v1/qa_numeric.json)

Retirement preserves first certificates and unresolved-row trajectories; it does not preserve a retired row's old cap-terminal state. Actual live-row coordinate steps are independently reconstructed from first-proof times and checkpoint counts. They are not FLOPs. All overlapping non-clock-limited legacy searches are compared with the previously published stage 378 arrays.

Full search time excludes final geometry, posterior sampling/readout, compressed disk output and offline audit. Named-array accounting is not process peak memory. No new query targets are used, and this release does not establish an independent PC-ALM task-level advantage, official-TTT/downstream validation or publication readiness. See the report for the actual positive evidence and the strongest controls; the original research goal remains active.

This is an after-noon-deadline supplement, not part of the pre-12:00 EDT delivery.
