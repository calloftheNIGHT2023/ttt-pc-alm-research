# 统一预加载后的限时结果与共同目标求解器

2026-09-25。本轮405已完成1344次预测、独立审计、评估及图文检查：1016次交付final，328次使用备用或中间预测，无常数退回。全部42方法、4个支持前缀、两个预算与逐任务数据均保留。四个任务是已经使用过的开发任务，不是独立确认。

## 实际任务结果

表中为每任务4/8/16/24支持前缀的MSE等权平均，再对四任务平均，越低越好。新主为active512 DFS；主预算仍为预先规定的0.5秒，没有事后改成更有利的0.25秒。

| 配置 | 0.25秒 | 0.5秒（主预算） |
| --- | ---: | ---: |
| 本轮主active512 DFS | 0.028631345605 | 0.021870237317 |
| BP组合Adam/GN 64+256起点 | 0.030931532925 | 0.017464080395 |
| BP组合Adam/GN 64起点 | 0.032416666940 | 0.018789712348 |
| GN40，64起点，后验并集 | 0.033117488570 | 0.020517449883 |
| 同训练cohort元岭回归128 | 0.043720797769 | 0.043720797769 |
| 本表均值最低的匹配官方TTT配置，h32/p4 | 0.046826584146 | 0.046826584146 |
| 同训练cohort浅头64/20 | 0.094077351480 | 0.094077351480 |

主预算下，当前主方法没有超过修正计时后的强BP组合；0.25秒仍有正向开发信号，但不据此替换主预算或宣布统计确认。全部控制见[406图文](../results/runtime_matched_prefix/report_v1/report.md)、[逐任务配对](../results/runtime_matched_prefix/evaluation_v1/comparisons.json)。

组合模块在不接收支持数据的setup阶段预加载，setup费用仍计入记录，没有提前进行支持更新。原397表因冷导入边界不一致而不能支持公平组合优势；其[原始结果和披露](PREFIX-DEADLINE-20260925.md)完整保留。本次是修正后的实际截止时间比较，不是通过调整归档时间来重算旧预测。

[405冻结设计](../outputs/ttt-pc-alm-research/405_runtime_matched_prefix_protocol_v1.md) · [1344次审计](../results/runtime_matched_prefix/audit_v1/summary.json) · [数值QA](../results/runtime_matched_prefix/report_v1/qa_numeric.json) · [实际视觉QA](../results/runtime_matched_prefix/report_v1/visual_qa.json)

## 共同目标接口已通过实际检查

有限tau带状目标、精确输出近端、相邻层偏导和驻点信用经过有理数检查；同目标BP Adam／GN、梯度PC／ALM、精确块PC／ALM六种实现通过组件预检。局部四族从零乘子开始，禁用全链Jacobian仍通过。完整数量、误差和范围见[410测试报告](../outputs/ttt-pc-alm-research/410_common_band_preflight_results_v1.md)。

这使下一轮能够分清乘子与块求解的作用，不表示这六种新实现已经取得任务收益，也没有将其混入405结果。现有候选还包含收费的全局几何步骤，RSS为采样下界，尚非同峰值RAM或严格同有限步目标的最终对照。

## 下一验证点

检查限时搜索在未完成时能否将已有证书转成有信息的部分预测，并与共享同样读出规则的强BP组合及更便宜的支持包络比较。已有[400风险推导](../outputs/ttt-pc-alm-research/400_certified_partial_readout_v1.md)提供条件性非增风险结论，但实际区间宽度、生成成本及PC-ALM独立收益仍须实验。

随后按[389三域计划](../outputs/ttt-pc-alm-research/389_cross_domain_small_model_requirements_v1.md)开展NLP、CV、Graph真实小模型：先各一个主任务，再各一个扩展任务，包含真正内部多层适应。当前没有这些真实模型训练或收益结果；完整研究目标保持进行中。

此快照由`scripts/runtime_matched_snapshot_20260925.py`限定发布范围并核验冻结源、逐调用文件、评估与两组组件结果。没有覆盖或删除旧实验。
