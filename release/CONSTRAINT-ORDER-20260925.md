# Constraint ordering and actual search bottlenecks — 25 September 2026

[中文结果与全部控制](../results/multiplier_constraint_order/report_v3/report.md) · [368 prefix protocol](../outputs/ttt-pc-alm-research/368_prefix_language_scaling_protocol_v1.md) · [369 theorem and claim ledger](../outputs/ttt-pc-alm-research/369_parameter_arrangement_bound_appendix_v1.md) · [370 ordering protocol](../outputs/ttt-pc-alm-research/370_multiplier_constraint_order_protocol_v1.md)

Publication of this supplement is gated on completed runs, independent signal/resource audit, symbolic checks, numeric figure QA and actual image inspection. The linked report, not the hypothesis in the protocol, is the authoritative conclusion.

## Scientific question

After removing the ALM mother trajectory without changing the four-support complete-pool prediction, the next question is where direct inference actually becomes expensive. A Cartesian-product count alone is not a computational lower bound: intersecting support paths incrementally can reject impossible prefixes earlier.

Protocol 368 tests two orderings on 32 prior four-support tasks and 16 older tasks with 4/8/16/24 supports. Its 192 component calls complete 155 joins and retain all 37 expansion-budget terminations. Both orderings reproduce the original complete candidate sets on the 32 four-support tasks. On 24 supports, original order completes 9/16 tasks and smallest marginal language first completes 0/16. This identifies intermediate-prefix congestion rather than an unavoidable full-product enumeration requirement.

Protocol 370 tests a genuinely different use of multiplier state: prioritize which already-observed constraints to join. It compares the fixed `alm_dual` score with eight controls, including spatial diversity, pairwise compatibility, residual and explicitly global BP scores. All score generation and failed work are charged. The output is search completion and cost, not query-risk superiority; any successful component result still requires a new full prediction experiment.

The theorem appendix gives an elementary fixed-dimension bound on positive-volume patterns through affine forms and a hyperplane-arrangement recurrence. It does not bound conservative false-positive prefixes, solve the inference problem, or establish PC-ALM novelty or speed.

## Evidence and known repair

The original 368 preflight passed; the first formal call stopped after saving its output because an older regression metadata field held `null` instead of an empty mode list. Original source and failure artifacts are preserved. The v2 adapter changes only that post-fit reading convention; the same failed call's 19 saved arrays replay exactly. No task, solver or budget was changed.

The snapshot includes complete saved-input component artifacts for 368 and 370, the theorem audit, protocols, result figures and QA. Older raw solver archives are not republished, and a clean-clone end-to-end rerun of all historical drivers has not been tested. Candidate functions can be inspected separately from the historical runner infrastructure.

Fixed parent: `53aba6cdacc9682cb7537dbc4552db3a1cf3c596`.

```powershell
python scripts/constraint_order_snapshot_20260925.py verify
```

This verifies the packaged bytes and frozen-source relationship; it does not itself rerun the algorithms. Preserve existing outputs, which use exclusive creation. Scientific checks were run in the current research environment with single-threaded numeric libraries. The continuing research goal remains active unless independent task benefit against all required strong controls is actually established.
