# 287｜独立确认的执行与接续说明

本文为执行状态与代码交接，不修改已封存的[287研究协议](287_probe_credit_confirmation_protocol.md)，也不是新的质量结果。研究目标保持进行中。

## 已完成的门禁

额外八个强回归/核/元学习头：32旧任务预测核对、192完整计时、16独立内存测量及独立审计通过。固定方法总数27，固定新任务8192，总预测221184。

长运行v2经过三次旧任务进程测试：两次调用后退出；重新计算未提交任务，完成一个任务后退出；跳过已提交任务并完成第二任务。54预测的348金标数组相同，已提交164文件保留；中断前两次额外调用0.6623822秒单独记账。独立恢复审计通过。此为受控进程退出测试，不声称能抵御任意操作系统或文件系统崩溃。

旧v1及其54预测前检全部保留。v2只改持久化与恢复，不改算法、观察、先验、步数、方法选择、样本量或判据。

## 正在运行

进程命令：`run_probe_credit_confirmation_v2.py --project . --stage confirmation`。

初始工具session为70031，PID31324，进程创建时间1790157256.106642。正确定位以`results/probe_credit_confirmation/predictions/RUNNING.lock`内的PID与创建时间二者为准，防止PID复用。工具等待返回超时不等于进程终止，不能因此再启动一份。

每个任务按冻结置换顺序运行全部27方法。每方法写入start记录、完成call记录和预测；27方法齐备再提交commit。不能覆盖已提交任务。部分attempt保留并单列已知完成费用；缺少完成记录的在途调用费用标为unknown，不能记0。

只可查看进度、失败、费用、状态计数及磁盘。全部8192任务提交前不读查询教师。预估核心计算14.86小时，不含全部I/O等费用；这是运行计划，不是优势保证。

## 新准备但尚未数值运行的代码

以下八个新文件仅经过语法解析，尚未执行旧任务数值前检；当前正式预测计时结束前不要运行它们的重计算。

|文件|用途|评价前要求|
|---|---|---|
|`probe_confirmation_independent_heads.py`|NumPy重算回归/元岭/浅头；浅头用显式带状损失梯度|旧预测数值核对|
|`audit_probe_confirmation_predictions.py`|全部文件、观察、状态、独立前向、支持粒子及头读出|全部221184通过|
|`probe_confirmation_path_reference.py`|原始Local块和逐起点BP/全试探轨迹参照|固定首64任务|
|`audit_probe_confirmation_paths.py`|从原始输入重算、全原子点、路径、几何、抽样；不求全空间后验|固定首64×27通过|
|`probe_confirmation_statistics.py`|以任务为单位、共享重抽索引的分块bootstrap|显式索引小例一致|
|`evaluate_probe_credit_confirmation.py`|25项校正上界、全26对照描述区间及资源表|两类审计先通过|
|`audit_probe_credit_confirmation_evaluation.py`|独立分段教师前向、全部实际风险和100000次重抽、全部区间与资源复核|评价后通过|
|`plot_probe_credit_confirmation.py`|27方法、104比较、完整费用和状态小计的两图及中文报告|独立统计复核后生成并视检|

新统计实现默认先用两个旧任务作功能前检，不是拿它们作确认检验；确认入口必须核对全8192清单和审计哈希。独立统计复核和完整图文生成已准备但未运行；不能把评价脚本自身的passed当成研究成功。

静态复查修正：`probe_confirmation_path_reference.py`只对branch_probe断言零初始乘子；普通ALM的alm1原子可以有合法非零乘子，并核验effective_initial_u与其原值相等。没有修改正式候选或对照。几何审计另要求精确面片边界闭合，若不闭合先保留预测并诊断，不开放查询质量。独立资源复核补齐每个命名状态小计、旧/实际预算类及所有启动加载费用。

[无正则OLS矩边界补充](287_ols_moment_addendum.md)在新查询未开放时由连续分布理想模型推导：四带噪支持点的无正则线性查询MSE均值有限而方差可无限；有限浮点/固定样本并非无限方差的实际数组。此为统计解释边界，不是方法贡献；主判据、任务、算法及所有强基线保持不变，不能据此删异常或弱化岭/RLS等控制。

## 预测全部结束后的顺序

在项目根目录使用现有Python314，设置`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。先核实正式runner的exit0、summary、完整manifest及无活跃锁。不要在计时进程仍活跃时开展以下步骤。

1. `audit_probe_confirmation_predictions.py --project .`：旧54预测数值前检。
2. `audit_probe_confirmation_paths.py --project .`：旧两任务原子/轨迹/几何前检。
3. `evaluate_probe_credit_confirmation.py --project .`：旧任务评价和统计功能前检。
4. `audit_probe_credit_confirmation_evaluation.py --project .`：旧任务风险与完整重抽复核。
5. `plot_probe_credit_confirmation.py --project .`：旧任务图文生成并打开两图视检，不能当科学结果。
6. 对上述第一项加`--stage confirmation`，核验全部新预测。
7. 对上述第二项加`--stage confirmation`，核验冻结的首64新任务，不按结果挑选。
8. 对上述第三项加`--stage confirmation`，这时才允许新查询教师评价。
9. 对上述第四项加`--stage confirmation`，独立重算实际风险与重抽区间。
10. 对上述第五项加`--stage confirmation`，完整报告所有预定对照、失败与资源差距，打开图片视检后再交付。另作总封存，不能以图文生成passed代替研究目标完成。

一旦数值前检不一致，保留原证据，先诊断基础设施；不要改候选、剔除任务或提前打开新质量。已冻结成功源不能直接修补，应使用新版本，并保持方法与判据不变。

## 什么才算结果

本次主指标是257网格独立查询MSE。候选减控制的单侧Bonferroni上界小于零才支持对应预定比较。只有全部25项均满足才可使用全族统一优于的表述；129网格、点预测、条件风险或子组不能替代主指标。100000次bootstrap是近似推断，不是有限样本严格数学保证。

即使本次确认有正收益，仍须区分局部信用的增量、共同几何/多解释读出的作用，以及更便宜控制的预算空间。官方TTT同任务、一般MLP和LLM/VLM桥接仍未完成。不得把本次合成确认缩写成整个研究目标已完成。
