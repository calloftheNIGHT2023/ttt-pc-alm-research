# 328 v2：只修日志序列化，不改研究问题、算法或任务

2026-09-25 01:46 EDT，v1预检在旧seed5910000的第二个方法online_uniform_state_dual_plus_residual写call.json时失败：TypeError: Object of type Fraction is not JSON serializable。已完成一个旧控制的保存，没有任何完整任务commit，没有query seal，没有评分，也没有新512任务运行。原v1所有源、通过的单元测试、一个旧控制输出、截断的call.json及failure.json原样保留。failure SHA210fe6251dbe9bf248d192bf37e609c30d3f0db6cc68c5ed68c40e1478c0b240。

原因是新持久化函数直接json.dump原始metadata；实际显式几何证书包含Fraction。已成功完成的326/327保存器使用exact_quadratic_events_v1.encode递归编码有理数，原主算法没有错误。本修订使用同一个已验证编码器，并在创建输出文件前完成encode和严格allow_nan=False的json.dumps，避免类型/非有限值错误留下新的半写文件；仍采用exclusive创建、flush和fsync，不覆盖已有文件。

六个328管线源另存_v2，模块引用及输出目录改到tests_v2/preflight_*_v2/pilot_*_v2。v1不修改。328_fresh_online_credit_protocol_v1.md保持原SHA和全部科学选择：同51方法、同512种子、同M=2048、同输入/初始化/目标/超参、同两个主候选、同预算、同统计及分组。326候选和327资源/强对照源一字未改。

新增单元覆盖嵌套Fraction及元组证书的精确往返、拒绝覆盖已存在文件、拒绝NaN且不创建半写文件。v2重新跑这些单元及全部153旧预测/评分/独立审计，逐位比同一327归档。只有新预检全部通过才能进入尚未使用的新任务。v1失败属于序列化实现问题，不是科研的负向或正向结果。
