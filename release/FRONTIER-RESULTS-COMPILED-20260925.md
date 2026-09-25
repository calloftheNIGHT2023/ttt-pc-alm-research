# Continuous frontier results and shared exact compilation

This snapshot contains the complete 419 development run (49 configurations, 1,568 calls), its independent audit, evaluator-only query results, and all-control figures with numerical and actual visual QA. It also includes the 422 exact compiler/component checks, 425 continuous-worker checks, and the 427 protocol and bound entry-point preflight. It does not claim an unrun 427 task result.

At the predeclared 0.5 s online deadline, the new continuous active512 reader has mean query MSE **0.0178309888721**, versus its old interface **0.0218702373174**, the strongest Adam/Gauss-Newton portfolio **0.0174640803952**, and the shared-reader PDHGcold control **0.0177561256737**. The primary candidate has not beaten the strongest portfolio, nor established an exclusive advantage over PDHG. Its 0.25 s result worsens relative to its old interface. All four tasks are reused development tasks, not independent confirmation.

The shared piecewise reader matches 24,672 archived box/query intervals exactly. Compilation plus reading takes 0.74345 s versus 2.44896 s across all eleven component cases; this excludes common search/box construction and is not a continuous speedup claim. The live compiled interface preserves 1,036 arrays and 272 packets bitwise in fixed-work checks. A real interrupted call has 67 processed query coordinates and no final archive; its delivered packets are independently replayed from the search receipt. Processed does not mean numerically changed or successful.

All methods receive the same observations and available prior. Cold setup, online cutoff, receipt hashing, compilation, serialization, cleanup and RSS sampling remain accounted for. Global geometry LPs remain in the algorithm; this is not an all-local PC-ALM claim. Same-finite-objective attribution, independent held-out confirmation, strict memory matching and the planned NLP/CV/Graph small-model suite are not complete.

- [All 49 controls, figures and actual delivery counts](../results/frontier_deadline/report_v1/report.md)
- [Run summary](../results/frontier_deadline/development_v1/summary.json)
- [Independent audit](../results/frontier_deadline/audit_v1/summary.json)
- [All evaluator results](../results/frontier_deadline/evaluation_v1/report.md)
- [Compiler mathematics and predeclared checks](../outputs/ttt-pc-alm-research/422_piecewise_frontier_readout_protocol_v1.md)
- [All eleven component results](../outputs/ttt-pc-alm-research/424_piecewise_frontier_component_results_v1.md)
- [Continuous compiled-worker checks](../outputs/ttt-pc-alm-research/426_compiled_continuous_preflight_results_v1.md)
- [Next frozen 56-configuration protocol](../outputs/ttt-pc-alm-research/427_compiled_frontier_deadline_protocol_v1.md)

Reproduce snapshot verification with `python scripts/frontier_results_compiled_snapshot_20260925.py verify`. See the manifest for exact files and digests; pre-existing dependencies remain pinned in the parent commit. Full research objective remains active.
