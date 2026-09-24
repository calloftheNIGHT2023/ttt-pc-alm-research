# 287｜最终证据总审计的接续计划

本文不改变冻结方法、数据、统计指标或判据。新增 `audit_round_287.py` 为待验证草稿；目前已通过语法检查和活跃锁拒绝测试，未执行任何实际总审计。正式预测计时期间不得运行审计工作，即使默认旧任务模式也会拒绝活跃锁；已核对拒绝测试没有创建输出。

## 用途

独立前向、轨迹/几何、查询风险和全重抽统计已经有各自检查器。总审计不冒充额外一次独立数值重现；它核对这些证据是否指向同一源版本、同一批完整预测、同一固定 27 方法与 25 项主比较，并检查全部原始文件清单和统计/图文的交叉引用。

必须同时保留技术通过与科学判据的区别：总审计 passed 表示证据链满足检查要求；main_adjusted_negative_upper_bounds 表示冻结近似统计阈值通过数量。无论数量如何，core_research_goal_complete 仍为 false；一般模型、官方 TTT、资源公平性和最终实际任务收益要另行审查。

## 执行位置

1. 原 287_execution_handoff.md 的五个旧任务前检全部完成后，实际打开两张旧任务图并阅读完整报告。
2. 在 `results/probe_credit_confirmation/figures_preflight/visual_review.json` 写入真实视检记录。不能仅因为画图脚本运行成功就预填通过。
3. 运行 `python work/experiments/audit_round_287.py --project .`，只对两个旧任务做功能前检；成功生成 `results/round_287_preflight_audit.json`。
4. 原确认阶段的预测核验、固定首 64 轨迹、评价、独立统计核验与图文完成后，实际打开两张确认图并阅读报告，在 `figures/visual_review.json` 保存独立视检记录。
5. 运行 `python work/experiments/audit_round_287.py --project . --stage confirmation`。要求同源旧任务总审计已经通过，成功后才生成 `results/round_287_audit.json`。

现有成功输出不可覆盖；发现失败应保留证据、诊断并以新版本修复，不回写冻结候选，也不按新查询结果调方法。

## 视检记录格式

人工/代理实际打开图像及报告后记录：stage、passed、reviewer、reviewed_utc；reviewed_files 包含 `287_all_methods.png`、`287_all_contrasts.png`、`report.md` 的实际 SHA-256。checks 必须逐项确认 layout_readable、no_clipping、all_methods_retained、interval_legend_correct、preflight_or_confirmation_label_correct、resource_and_evidence_limits_present。仅语法分析或脚本自报不满足此门禁。

原正式预测源未因新增此审计改动。上传阶段快照期间普通 CPU/磁盘竞争的限制在 release/SCOPE.md 中保留；最终统计报告解释实际费用时必须一并说明，不能宣称整段运行完全独占资源。
