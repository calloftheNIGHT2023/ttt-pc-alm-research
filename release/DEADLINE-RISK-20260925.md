# New-task deadline risk and exact mechanism decomposition

2026-09-25. Development evidence, not independent confirmation or a completed paper claim.

## Outcome

The frozen experiment ran 32 new tasks, all 36 registered configurations and two independent online deadlines (0.5 / 1 second): **2,304 sealed predictions**, followed by a passed independent audit and only then query evaluation. The primary remained `regional_active1024_bfs_observed` at 0.5 second.

| Method | Mean query MSE, 0.5 second | Timely final packets / 32 |
| --- | ---: | ---: |
| Original primary: active1024 BFS | 0.0118358784672 | 12 |
| Passive1024 BFS | 0.0164055562994 | 3 |
| Meta-ridge128 | 0.0189512439790 | 32 |
| Adam240, 64 restarts, common union readout | 0.0000000380695525823 | 32 |
| Gauss-Newton40, 64 restarts, common union readout | 0.0000000380695525823 | 32 |

Final packets are not synonymous with a complete posterior. The primary has a positive improvement over the stated regression and passive-search controls, but **does not establish an independent advantage over strong BP**. Both predeclared all-eight-control success flags are false. At 1 second the original primary MSE is 0.00155977615315; the predeclared active DFS variant reaches 3.80666955127e-8, while the 64-restart Adam/GN controls already reach approximately the same risk at 0.5 second. This does not replace the original primary.

All 35 nonprimary controls are retained. In particular, the 256-restart union controls miss the deadline and use the paid fallback; omitting the successful 64-restart controls would incorrectly suggest superiority over BP. More restarts do not automatically make a stronger deadline-constrained control.

## What the mechanism evidence establishes

The pre-query-frozen v1 mechanism plan decomposes paired risk differences into timely fallback delivery, timely complete delivery, and differences between complete predictions. The actual analysis reconstructs all **576 per-task identities with zero error**. All **404 timely complete regional predictions are bitwise identical**. Consequently, differences among these regional methods in this experiment come entirely from whether the common complete prediction arrives before the deadline, not a different final prediction.

For primary versus passive BFS at 0.5 second, the paired primary-minus-control mean is -0.00456967783220; the 95% task-bootstrap interval is approximately [-0.00736839, -0.00207958], and the stated 35-comparison adjusted upper bound is -0.000990288. This is component-level positive evidence for the cost-to-risk channel, not proof that complete search is needed over a strong optimizer. Cold PDHG and other search orders remain competitive.

## Evidence and reproducibility

- [Frozen experiment and information/resource rules](../outputs/ttt-pc-alm-research/387_new_task_deadline_risk_protocol_v1.md)
- [All 36 configurations, both budgets: figures](../results/new_task_deadline_risk/all_controls_report_v1/report.md)
- [All numerical results, paired comparisons and resource tables](../results/new_task_deadline_risk/report_v1/report.md)
- [Pre-query frozen mechanism decomposition](../results/new_task_deadline_risk/mechanism_decomposition_v1/report.md)
- [Independent audit](../results/new_task_deadline_risk/audit_v1/summary.json)
- [Required subsequent NLP / CV / Graph real-small-model phase](../outputs/ttt-pc-alm-research/389_cross_domain_small_model_requirements_v1.md)

The audit checks 2,304 delivery/resource records, 66,746 exact geometry certificates, 58,331 exact local wide-domain cuts, 1,316,864 particle-support checks, method replays and 288 process-session lifecycles. Numerical report QA recomputes all 2,304 risks, 70 bootstrap comparisons and 1,712 table cells; the all-controls companion independently checks all 72 plotted group values. All three figures were actually viewed and passed visual QA. The failed first n24 adapter preflight is preserved alongside its successful v2 repair.

Publication is bounded by `scripts/deadline_risk_snapshot_20260925.py` and `release/deadline-risk-20260925.json`, which verify immutable parent blobs, frozen source/weight hashes, sealed result files, report hashes and exact staging. Run its `verify` action to validate this saved snapshot. This is not a claim of a separate end-to-end scientific rerun.

## Remaining work

There is no matched official-TTT result in this release, no strict common peak-RAM cap, and finite-step optimizer surrogate objectives are not identical. Official TTT matching is a separate workstream, not a retroactive change to this registry. The next mechanism question is whether genuinely ambiguous low-support contexts create predictive posterior mass missed by finite-start optimizers, and whether local certified search can resolve that mass efficiently. This is a hypothesis, not an observed benefit.

The research goal remains active. Real small models in **NLP, CV and Graph are mandatory after the theory and matched-control gates pass**; the attached plan is not a real-data result.
