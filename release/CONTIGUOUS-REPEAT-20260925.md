# 384–385｜连续支持到预测的重复费用验证

全部960连续调用和独立审计完成：16旧开发任务、十种配置、两种调度/排序、三次技术重复。每次重新搜索、验证几何和读取2048粒子，没有加载旧搜索状态或查询答案。

主active1024的每任务三次中位数跨任务均值：

| 设置 | active1024 | 本轮最低PDHG | 均时变化 | 主减控制的任务配对95%区间 |
| --- | ---: | ---: | ---: | --- |
| BFS / observed | 0.498811秒 | cold1024：0.528265秒 | -5.58% | [-0.058140, -0.004850]秒 |
| DFS / farthest_x | 0.464455秒 | box1024：0.520498秒 | -10.77% | [-0.132035, 0.008807]秒 |

BFS提供旧任务上连续完整链成本改善的证据；DFS均值有利但区间跨零，不能称两个设置均确认稳定加速。全部十配置960调用完整解析，三次输出逐位一致；完整读出跨方法和调度相同。没有新查询MSE，不构成优于强回归/浅头/TTT的完整论文结果。

独立审计：47,412精确几何证书、47,218精确局部不可行证书、655,360粒子支持检查、320预测重放、105,930重复数组摘要、640重复非时间元数据核验。报告额外核对304跨方法预测摘要；364个表格格及18个配对bootstrap结果从960原始行重算，图像已实际查看。

- [冻结协议](../outputs/ttt-pc-alm-research/384_contiguous_repeat_protocol_v1.md)
- [完整图文、全部控制和资源](../results/contiguous_regional_repeat/report_v1/report.md)
- [960次原始记录](../results/contiguous_regional_repeat/development_v1/rows.json)
- [独立审计](../results/contiguous_regional_repeat/audit_v1/summary.json)
- [三次原值及每任务统计](../results/contiguous_regional_repeat/audit_v1/task_groups.json)

搜索受协作预算限制，但当前完整几何阶段未设置全链期限；不是等总时间anytime竞赛。共享GPU没有用于本轮。后续截止时间原型和强基线接口检查不包含在本快照的正式结果中。
