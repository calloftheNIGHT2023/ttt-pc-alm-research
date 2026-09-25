# 324–325 新结果：支持梯度为零时，持久乘子的独立分支价值实例

封存时间：2026-09-25T04:58:02+00:00。这是9月25日中午交付材料的追加版；此前321–323报告保留。整体研究目标未完成。

## 本轮最重要的结果

**拿到了一个经独立核验的正向机制实例。** 在seed=5910054的首次支持拟合状态，BP支持损失信用为零，历史局部乘子仍非零。相同状态、相同链式选择器、相同每距离层K=8预算下，乘子将一个有效区域排到第3位，而BP、纯残差、随机符号与零信用均未进入前8。该区域占完整支持后验质量 **4.5925%**，并非近似零体积碎片。

将旧强控制并集和本轮全部非乘子控制记为C，加入该乘子独有区域后，64任务平均条件超额风险下降 **1.45762129e-05**。两对粒子批与129/257网格方向一致。**全部独立边际收益来自这一个任务；去掉它，剩余63任务增益为0。** 因此这证明了当前有限预算选择器下的独立机制作用，但未证明新任务上的稳定方法优势，更未证明所有闭式回归/浅层网络都做不到。

![完整风险与边际价值](C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/results/support_consistency_trigger/report_v1/credit_case.png)

左图保留更强的早期Adam-8结果，不能只比较较弱信用控制。右图的C是昂贵的离线多方法并集，不能假装为单一在线算法；候选生成过程中未使用其中BP控制来初始化状态或信用。

## 数学机制与适用条件

支持误差带损失为 L(b)=sum_i max(|f_b(x_i)-v_i|-epsilon,0)^2/(2n)。当所有支持样本都落在误差带内，L=0且参数梯度为0；但局部乘子u累计历史层间残差，不要求在前向支持损失刚归零时同步归零。因此，当前BP信用无法区分的分支，u仍可能提供不同排序。这是**计算状态的差别，不是额外标签或额外观察信息**。

321的链式证书对真实可行模式满足D_chain≤0，并支配不保留跨层分支像的旧松弛。324将同一选择器移动到支持一致状态，不改K、不增加适应参数。更负下界仍不代表更高概率；本轮只有通过完整区域证书和后验风险评价后，才确认该排序变化有实际条件预测价值。

若乘子已衰减到0、排序未改变，或新区域预测与完整后验均值偏离更大，则不一定改善。随机位置＋乘子残差在本轮边际读出中更强，说明“首次拟合触发必然最佳”并未成立。不能从本实例推导PC-ALM对所有任务不可替代。

## 固定实验与结果完整性

沿原纯局部ALM轨迹的1088个步后状态，每任务固定取一个first_fit状态；无支持一致状态则按最低支持误差带损失回退，平局取最早。另用固定随机种子均匀取一个轨迹状态。全部64旧开发任务都保留，两策略各64位置、六信用来源，共768选择器调用；18203提议先封存，再进行几何评价。

first_fit有25个真正支持一致状态、39个回退；25个成功状态的BP信用全为0且乘子全非零。随机位置的64状态均未支持一致。独立原轨迹核验遍历69632状态；首达性、状态数组和768信用数组全部对齐。

共同几何10034个不同任务-模式中：226正体积、9799不可行、9零体积或空；没有未解决分类。所有完整64任务统计如下，重复区域不能跨方法相加：

| 触发/信用 | 正体积区域 | 超出旧强池S | 超出S及本轮非乘子通道 |
|---|---:|---:|---:|
| first_fit/dual | 65 | 2 | 1 |
| first_fit/dual_plus_residual | 66 | 2 | 1 |
| first_fit/residual | 69 | 10 | 0 |
| first_fit/bp | 49 | 9 | 0 |
| first_fit/random_sign | 80 | 6 | 0 |
| first_fit/zero | 41 | 7 | 0 |
| uniform_state/dual | 41 | 4 | 1 |
| uniform_state/dual_plus_residual | 42 | 5 | 2 |
| uniform_state/residual | 41 | 4 | 0 |
| uniform_state/bp | 41 | 3 | 0 |
| uniform_state/random_sign | 42 | 2 | 0 |
| uniform_state/zero | 29 | 3 | 0 |

旧强池S在319–321累计区域之外，还显式补入297全部20种ALM/PC/nodual/Adam续接控制的区域，补入了71个此前并集遗漏的任务-模式。S/C不是免费先验，也不是候选线上使用的BP模块；它们只用于检验是否有旧控制解释不了的边际价值。

## 全64任务条件风险：不删强对照

将各通道新区域并入同一个原ALM池O后的结果如下，越低越好。这是已知先验下离线理想后验读出的**条件超额风险**，不是实际未见查询MSE；数值体积和Monte Carlo区域矩仍有估计误差。

| 信用 | 首次拟合/最优回退：条件风险 | 随机位置：条件风险 |
|---|---:|---:|
| 乘子 | 0.0012763188 | 0.0012815748 |
| 乘子＋残差 | 0.0012763188 | 0.0012622105 |
| 残差 | 0.0013049543 | 0.0013122473 |
| BP信用 | 0.0012992689 | 0.0012813347 |
| 随机符号 | 0.0013080967 | 0.0013064740 |
| 零信用 | 0.0013000570 | 0.0013067642 |

原O：0.0013067236。首次拟合＋乘子：0.0012763188，相对O减少2.327%的这项条件超额风险。早期Adam-8：**0.0010429252**，仍优于该完整候选池。因此不能声称当前纯局部候选已优于强BP。

旧强池S为0.0010248792；加入首次拟合乘子为0.0010101745。再把本轮全部非乘子控制补入C之后，独有增益仍是上述1.45762129e-05，但只发生在seed5910054。全部46池、257/129网格、25拟合/39回退及全64三个层次的276汇总均保存在原始表，不只报告有利子集。

## 为什么该区域进入预算，而对照没有

目标与当前模式的Hamming距离为2。用独立反向搜索器与另一份固定下界实现交叉核验，所有信用下D≤0，没有把一个可行区域错误标为不可行；区别在有限K排序，而非额外标签：

| 同一状态信用 | 目标所在距离层的排名/排除原因 |
|---|---|
| 乘子 | 第 3 位 |
| 乘子＋残差 | 第 3 位 |
| 残差 | 前8名之外（精确排序不等式核验） |
| BP信用 | 前8名之外（精确排序不等式核验） |
| 随机符号 | 前8名之外（精确排序不等式核验） |
| 零信用 | 前8名之外（精确排序不等式核验） |

[精确有理下界、前8截止值及模式编码见证](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/results/support_consistency_trigger/exclusive_value_v1/witnesses.json>)保留了全部五个候选-区域见证。相同区域被两个乘子通道找到，不算两份独立收益。

## 核验、资源与下一门槛

独立搜索审计覆盖2304个实际K最佳距离层、全部18203提议及固定下界；独立几何核验39684证书、768方法池、3072比较集合，并重建1280个旧297控制池。324风险独立标量重算5888行、48760字段，最大差6.939e-18。325独立复算512风险行和30个精确信用下界，最大差1.735e-18。

选择器主实验约213.57秒、几何约64.09秒、独立K序审计约201.71秒。它们是诊断计时（部分审计并行），不是匹配资源下净加速。first_fit额外计算55267次精确支持前向；原33起点准备、1088局部步、活动/乘子/搜索表和几何读出也须计入完整方法成本。不能把复用归档状态算免费。

**下一步门槛是实际在线实现与新冻结任务复现。** 保持纯局部前缀，不运行BP给候选生成信用；实现first_fit及随机位置触发的真实状态缓存和读出，先与本轮归档逐位兼容，再计完整状态/时间，保留强回归、浅层头和同参数强优化器。只有新任务中的平均查询收益及资源证据支持，才可把此单任务机制实例升级为方法主张。当前尚无该新确认、官方TTT完整对照或LLM/VLM结果。

## 复现入口

- [324冻结协议](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/outputs/ttt-pc-alm-research/324_support_consistency_trigger_protocol_v1.md>)；[325归因协议](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/outputs/ttt-pc-alm-research/325_credit_exclusive_value_protocol_v1.md>)。
- [完整46池风险表](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/results/support_consistency_trigger/risk_v1/aggregate.json>)；[每任务风险](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/results/support_consistency_trigger/risk_v1/scores.json>)；[独有区域边际价值](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/results/support_consistency_trigger/exclusive_value_v1/summary.json>)。
- [独立K序审计](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/results/support_consistency_trigger/search_audit_v1/summary.json>)；[独立几何审计](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/results/support_consistency_trigger/geometry_audit_v1/summary.json>)；[独立风险审计](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/results/support_consistency_trigger/risk_audit_v1/summary.json>)；[独立归因审计](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/results/support_consistency_trigger/exclusive_audit_v1/summary.json>)。
- [状态触发实现](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/work/experiments/prepare_support_consistency_states_v1.py>)；[提议实现](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/work/experiments/run_support_consistency_search_v1.py>)；[本报告生成器](<C:/Users/callofthenight/Documents/Codex/2026-09-18/ttt-pc-alm-ttt-pc-alm/work/experiments/report_support_consistency_v1.py>)。

在项目根目录使用已有Python314，运行各脚本的--out并指定新目录；不要覆盖封存输出。全部实现和输入哈希在协议/summary中，报告与图哈希在manifest.json。当前是本地研究成果，未据此执行Git推送或投稿。
