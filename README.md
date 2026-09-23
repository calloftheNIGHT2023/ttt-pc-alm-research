# TTT × PC-ALM：少样本上下文适应中的局部信用与分支探索

阶段性研究快照，2026-09-23。目标是在相同观察、先验和完整资源预算下，找到局部乘子相对强回归与同参数优化器的独立任务收益。

**目前已得到精确的机制见证；尚未确认新任务分布上的独立平均优势。** 这不是已完成的 LLM/VLM 方法，也不是官方 TTT 或 PC-ALM 的完整效果复现。

## 最重要的阶段结果

同一个四参数非线性模型、相同初始参数和活动、零初始化乘子：无乘子方法处于精确块不动点，真实参数目标的梯度因共享参数信用抵消而为零。累积乘子改变跨分支能量，能够离开该状态。

对固定原始状态和一个合法竞争点，

\[
u_t=t\eta r_0,\qquad E_t(z')-E_t(z_0)=\Delta-t\eta\kappa.
\]

若 \(\kappa>0\)，超过 \(\Delta/(\eta\kappa)\) 次累积后，旧点不能继续保持块最优。具体支持侧选定见证在第 7 轮发生真实参数分支改变，第 62 轮进入观察噪声带；独立查询的精确积分风险由 **0.1879342658 降至 0.1214341867，下降 35.38%**。

这是条件性存在见证，不代表所有 BP、回归或浅层头都无法解决任务。共同准备成本不能免费，也不能将该单例当作总体提升。

![精确停滞与乘子逃逸](results/multiplier_fixed_point/analysis/multiplier_fixed_point.png)

- [完整数学、控制与成功/失败边界：报告 246](outputs/ttt-pc-alm-research/246_multiplier_fixed_point_escape_results.md)
- [全部精确证书](results/multiplier_fixed_point/exact/certificates.json)、[独立核验](results/multiplier_fixed_point/audit/summary.json)、[精确风险积分](results/multiplier_fixed_point/witness/query_risk.json)

## 冷启动与强基线

后续候选使用支持侧分支试探与局部 ALM 续接，再通过共同几何/多解释读出预测。候选内部不使用全局 BP 或 BP 信用初始化；部分强对照和外层元训练使用 BP，明确分列。

在重复使用的 64 个开发任务上，候选查询 MSE 为 0.0501401。对增强预算后的 PC、强 Adam 等控制，**查询误差差异尚未获得确认**。全部控制及费用保留，不能从更有利的条件风险指标替代查询结论。

![强预算开发比较](results/probe_credit_budget/figures/286_probe_credit_budget_development.png)

- [报告 286](outputs/ttt-pc-alm-research/286_probe_credit_budget_results.md)：77 种方法、完整描述性比较与边界。
- [全部方法表](results/probe_credit_budget/figures/all_methods.md)、[资源表](results/probe_credit_resources/figures/resource_table.md)：包含计算、求解器状态和额外费用。
- [冻结确认协议 287](outputs/ttt-pc-alm-research/287_probe_credit_confirmation_protocol.md)：8192 个新任务 × 27 方法；先封存全部 221184 份预测，再审计和开放查询答案。

确认预测在首次上传时仍在本地运行。本快照不包含在途新任务预测，不提供其质量结论。后续审计/统计脚本已准备但尚未通过数值前检，不能把“代码存在”当作验证完成。

## 仓库内容与复核

|位置|内容|
|---|---|
|`work/experiments/`|研究源码及历史实验脚本；包括明确标记的未验证后续脚本|
|`outputs/ttt-pc-alm-research/`|按编号保留的机制推导、实验协议和阶段报告|
|`results/multiplier_fixed_point/`|246 阶段全部原始数据、证书、轨迹与图文|
|其他已跟踪 `results/`|选定汇总、协议和图表；不是所有历史原始档案|
|`release/`|发布范围、环境记录、SHA-256 文件清单|
|`work/third_party/`|两个官方仓库的固定提交子模块，保留上游许可证|

```bash
git clone --recurse-submodules https://github.com/calloftheNIGHT2023/ttt-pc-alm-research.git
cd ttt-pc-alm-research
python scripts/snapshot.py verify
```

上面的检查只验证快照字节、Python 语法和冻结源哈希，不运行研究算法、不等于数值重现。具体说明见 [复核指南](release/REPRODUCIBILITY.md) 和 [发布范围](release/SCOPE.md)。历史报告指向未打包档案的链接可能不完整；不要在已有封存输出上直接重跑带写入的原实验脚本。

## 来源与定位

原始工作：[TTT 论文](https://arxiv.org/abs/2407.04620)、[PC-ALM 论文](https://arxiv.org/abs/2605.31022)、[R2-D2](https://www.robots.ox.ac.uk/~vgg/publications/2019/bertinetto19/bertinetto19.pdf)、[Test-time regression](https://www.jmlr.org/papers/v27/25-0903.html)。官方代码提交与许可证见 [来源记录](release/UPSTREAM.md)。

本仓库不把“无 backward”“开放更多参数”或“信用等于 BP”作为贡献。研究正在检验：未满足的层间约束如何产生可用的跨分支动力学，以及这能否转化为强基线之外的查询风险收益。
