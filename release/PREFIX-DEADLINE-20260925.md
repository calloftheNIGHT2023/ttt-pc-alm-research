# 四前缀限时开发、精确读出原语与组合运行时修正

2026-09-25。四个旧任务，42 方法，4/8/16/24 支持前缀，每次 .25/.5 秒独立预算，共 1344 次调用。全部预测封存、独立审计、评估及图文数值 QA 通过；997 final、347 备用／中间包，0 常数退回。两张完整热图已实际查看。

**必须先读计时补充：** 原组合 worker 在在线阶段才导入优化器运行模块，单优化器却在无任务 setup 阶段预加载。因此原表的两条组合结果不能支持“超过公平预热的强组合”。[问题、原始证据与修正规则](../outputs/ttt-pc-alm-research/402_portfolio_runtime_setup_correction_v1.md)和[新旧 worker 复测](../outputs/ttt-pc-alm-research/403_portfolio_runtime_setup_results_v1.md)一并公开。原表和冻结代码不回填、不覆盖。

## 本轮开发信号

下表为 .5 秒下每任务四前缀 MSE 等权平均，再对四旧任务平均。不是独立确认、显著性结果或真实模型结果；预列 active512 的排名不追溯替换原主 active1024。

| 配置 | 四前缀平均 MSE |
| --- | ---: |
| 预列 active512 DFS | 0.01878577846485 |
| GN40，64 起点，后验并集 | 0.02051744988340 |
| Adam240，64 起点，后验并集 | 0.02081725880269 |
| 原主 active1024 DFS | 0.02187023731738 |
| 同训练 cohort 元岭回归128 | 0.04372079776860 |
| 匹配官方 TTT 中本表最低均值配置：native prior256 h32/p4 | 0.04682658414637 |
| 匹配官方 TTT 原主：native prior256 h32/p1 | 0.05203657483845 |
| 同 cohort 浅头64/20 | 0.09407735148009 |

active512 提供值得独立确认的正向开发信号，但不是每个前缀都更好：n24 的 active512 MSE 为 .00552767245，而 Adam/GN64 后验并集约为 7.87e-8。原主 active1024 在 .5 秒下没有超过强单优化器。全部方法、前缀、预算与逐任务差异均保留，不能只选此表里较弱的控制。

[完整42方法图文](../results/prefix_deadline_pilot/report_v1/report.md) · [冻结协议](../outputs/ttt-pc-alm-research/397_prefix_deadline_pilot_v1.md) · [1344次独立审计](../results/prefix_deadline_pilot/audit_v1/summary.json) · [全部配对结果](../results/prefix_deadline_pilot/evaluation_v1/comparisons.json)

## 新的可检验数学接口

已知可行内质量与积分界、并能控制包括未搜索区域在内的总质量上界时，可以给出后验均值区间。将任何基线投影到有效区间不会增加条件期望平方风险；区间外时严格下降。该结论适用于所有求解器，不专属 PC-ALM；独立贡献仍须来自更便宜地产生有信息的证书。

[400完整推导与条件](../outputs/ttt-pc-alm-research/400_certified_partial_readout_v1.md) · [401精确原语测试结果](../outputs/ttt-pc-alm-research/401_certified_partial_readout_preflight_v1.md)。没有在线搜索接入或实际任务收益声明；既有单纯形积分代码明确复用。

## 仍需完成

修正预加载后的限时组合比较、独立新任务确认、同目标同参数归因、证书区间的实际宽度与完整费用，以及真实内部适应接口。随后执行用户要求的 [NLP / CV / Graph 三域真实小模型计划](../outputs/ttt-pc-alm-research/389_cross_domain_small_model_requirements_v1.md)，不能以合成实验替代。当前没有三域真实任务结果，完整研究目标保持进行中。

本快照由 `scripts/prefix_deadline_snapshot_20260925.py` 校验冻结哈希、原始逐调用归档、数值与视觉 QA，并限定 Git 暂存范围。无原实验删除或覆盖。
