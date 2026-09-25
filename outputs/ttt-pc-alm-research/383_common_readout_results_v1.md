# 383｜共同验证与读出阶段结果

BFS/observed：固定主active256完整解析16/16、均拼接费用0.5648秒；全解析PDHG配置中最低均时为pdhg_cold1024、0.5266秒。

DFS/farthest_x：固定主active256完整解析16/16、均拼接费用0.6262秒；全解析PDHG配置中最低均时为pdhg_box1024、0.5199秒。

以下预列active配置在两个设置均完整解析16/16，且本次拼接平均费用低于各设置全部PDHG配置中的最低值：regional_active512, regional_active1024。这支持进入连续整链重复计时，不等于已经确认稳定加速或未见查询收益。

[完整图文与17配置](../../results/candidate_set_readout/report_v1/report.md)

没有新query风险结果，目标保持ACTIVE。
