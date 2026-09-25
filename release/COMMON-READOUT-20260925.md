# Common candidate validation and readout

Stage 382–383 connects every one of the 544 sealed stage-380 candidate sets to the same uncached geometry validation and 2048-particle readout. All 17 predeclared configurations and both settings remain visible. The fixed primary remains `regional_active256`.

- [Full results, cost decomposition and all controls](../results/candidate_set_readout/report_v1/report.md)
- [Execution-before-results protocol](../outputs/ttt-pc-alm-research/382_common_readout_protocol_v1.md)
- [Prototype invariants](../results/candidate_set_readout/prototype_preflight_v1/summary.json)
- [All sealed calls](../results/candidate_set_readout/development_v1/rows.json)
- [Independent exact-geometry and readout audit](../results/candidate_set_readout/audit_v1/summary.json)
- [Numerically verified report](../results/candidate_set_readout/report_v1/qa_numeric.json)

The original measured search cost is fully charged. This is a **sum of separately measured stages**, not a repeated contiguous end-to-end timing. Geometry uses global LP and numerical polytope volumes; it is explicitly shared and charged, not described as local-only PC. No method borrows modes from another method or an archived complete posterior.

Incomplete searches produce only a conditional readout from their own emitted candidates, or an explicit unavailable readout. They are not silently recovered or dropped from a query-MSE average. No query targets are accessed in this experiment. Fully resolved searches are checked for identical accepted modes, particles and predictions under the common readout.

This development-stage evidence does not establish new-task query-risk improvement, superiority to all regression/shallow-head/same-parameter optimizers, official-TTT downstream validation or paper readiness. The original research goal remains active. This release is an after-noon-deadline supplement.
