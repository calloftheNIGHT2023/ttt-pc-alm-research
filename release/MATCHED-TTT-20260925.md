# Matched official TTT training and low-support coverage diagnosis

2026-09-25. Completed baseline preparation and mechanism diagnostics; not independent method superiority or downstream evidence.

## Eight trained models with exactly the same outer-training data

Six official-TTT bridge configurations, one learned ridge model and one shallow nonlinear model each consumed the same stored 2,000 batches: 32,000 shared unique tasks, 2,048,000 shared unique query values, and 8,192,000 four-prefix query-loss exposures **per model**. Configuration exposures and resource usage are reported separately. These are not 256,000 independent tasks.

The official fast-weight update uses the pinned `test-time-training/ttt-lm-pytorch` submodule, commit `cd831db10c8c9a0f6340f02da5613316a8a92b67`. Four native-eta prior256 variants vary head dimension 16/32 and inner passes 1/4; native scalar and legacy-eta prior256 variants are retained. These are task bridges, not pretrained language models. Outer training uses BP; the original official TTT control retains its gradient-based inner update.

Training completed in 728.124 seconds of recorded wall time. An independent audit regenerated all 8,000 cohort tensors exactly, verified 16,000 per-model batch-consumption records, checked checkpoint selection and hashes, and replayed old-validation predictions on CPU. Maximum CPU-versus-training-device validation difference was 4.0603e-13. Concurrent light diagnostics/publication mean elapsed training times are accounting records, not isolated throughput claims.

[All eight models, training curves, prefix curves and resource table](../results/matched_official_ttt/report_v1/report.md). The plots were actually viewed. Numerical QA retains all 72 training-validation points, 32 prefix values and 64 table values. Every reported error in this training report is **old-validation raw MSE**, not new held-out research risk.

## Support-only coverage mechanism evidence

[Protocol and derivation](../outputs/ttt-pc-alm-research/393_low_support_coverage_hypothesis_v1.md) and [all diagnostic results](../results/low_support_coverage/audit_v1/report.md).

For omitted posterior mass delta and retained/omitted conditional prediction means, the exact conditional square-loss bias is delta squared times their squared predictive disagreement. Finite Monte Carlo variance and numerical geometry limitations are explicit. Rational tests cover 128 random cases, positive and zero-bias cases, and a realized-teacher counterexample.

Two fixed old tasks at four prefixes produced 32 sealed optimizer banks before offline reference geometry. All eight reference contexts resolved; exact search checks, 114,688 support-particle checks and 32 bias identities passed. An independent saved-particle replay reproduced all 28 region means exactly.

Predictively important omissions occur for individual Adam or Gauss-Newton controls, but they are complementary: a post-hoc four-bank union nearly covers the reference in these eight contexts. Therefore actual-cost optimizer portfolios must be included next. This diagnostic does not establish a unique PC-ALM benefit, and no realized query target was read.

## Matched-prefix and anytime-portfolio interfaces

[395 design and tests](../outputs/ttt-pc-alm-research/395_prefix_matched_controls_v1.md). Eight models at four prefixes passed 32 direct comparisons, 32 A-B-A isolation checks and 32 negative-stride query checks. Two portfolios at four prefixes passed 72 independently recomputed union arrays and eight task-isolation tests. Thirty-five actual worker calls, 14 repeated/recreated prediction comparisons and four exact receipt-time boundaries passed.

The portfolios progressively combine independently computed Adam/GN banks, reusing geometry only within the same invocation. Each stage can deliver a prediction before the deadline. All bank generation, new geometry, sampling and readout costs are paid; no cross-task cache or query labels are provided. An intermediate or final packet is not a claim of complete posterior coverage.

## Reproduction and remaining scope

Use `python scripts/matched_ttt_snapshot_20260925.py verify` to check this bounded saved snapshot. The official source remains the existing pinned Git submodule; initialize it before verification on a fresh clone. Checkpoint, training-cohort, source and result hashes are retained. Snapshot integrity and saved-input audits are not a full training rerun.

The next experimental comparison must jointly include prefix length, strong regressions, shallow heads, official TTT, individual and portfolio optimizers, matched local controls and full resource accounting. The complete research goal remains active. The required [NLP / CV / Graph real-small-model phase](../outputs/ttt-pc-alm-research/389_cross_domain_small_model_requirements_v1.md) has not started; its theory gates are not yet all satisfied.
