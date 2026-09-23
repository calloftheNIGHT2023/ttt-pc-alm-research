# 复核层级与已知限制

## 1. 不进行研究计算的快照检查

在仓库根目录运行 `python scripts/snapshot.py verify`。仅依赖 Python 标准库，检查发布清单中的每个文件、Python 语法状态，以及 246/286/287 源码哈希。不会加载新任务答案、执行算法、写入原结果目录或依赖本机绝对路径。

已知历史例外：`analyze_dual_jump_query.py` 第 17 行有未闭合括号，属于 257 阶段保留的失败 v1；实际成功版本为 `analyze_dual_jump_query_v2.py`。旧审计刻意同时封存两份文件，失败 v1 的字节仍与冻结哈希相符。本快照如实保留它，不宣称它可执行。检查器只接受这一确切已知例外，其他语法错误均拒绝；八个后续草稿仍必须通过语法检查。语法通过也不等于数值测试通过。

它回答“代码与所上传的证据是否一致”，不回答“方法是否有效”。仓库中 passed 的历史审计摘要是历史证据，不是假定本次已重新运行全部实验。

## 2. 246 机制阶段的独立数值/有理重放

本仓库包含该阶段全部 screen/exact/continuation/audit/witness/analysis 文件。原脚本拒绝覆盖已有结果，因此应在另一个新建目录中复制 `work/experiments` 和 `results/multiplier_fixed_point` 的 `screen`、`exact`、`continuation` 三个输入子目录，然后依次执行：

```bash
python work/experiments/audit_multiplier_fixed_point.py --project NEW_REPLAY_DIRECTORY
python work/experiments/prove_multiplier_escape_witness.py --project NEW_REPLAY_DIRECTORY
python work/experiments/prove_escape_stationary_basin.py --project NEW_REPLAY_DIRECTORY
python work/experiments/exact_escape_query_risk.py --project NEW_REPLAY_DIRECTORY
```

以上从包含的原始数据复核证书、完整有理轨迹、局部极小邻域和精确风险，而不是从零复跑此前 245 个研究阶段。本次发布期间没有运行这些计算，以免与正式预测计时竞争。请使用实际新目录路径，保留所有原始档案；不要删除原 audit/witness 目录来强制覆盖。

## 3. 开发比较与 287 确认

286 已包含全部方法与配对汇总，但不是其全部前置原始预测/后验数组。完整端到端历史重现需要本机完整研究档案，不能声称干净克隆后直接运行任意历史脚本都会成功。五个历史元学习检查点等依赖也未在此首版打包。

287 冻结协议和源代码已上传，新的运行产物还未完成，后续统计代码尚未数值前检。不要在克隆中直接启动完整确认来“补齐缺失”，这不是轻量复核且可能需要长时间与额外数据。预测、审计、统计、图文各阶段状态独立报告。

## 环境

原运行环境：Windows，CPython 3.14.2；NumPy 2.4.2、SciPy 1.17.1、PyTorch 2.10.0+cu128、Matplotlib 3.10.8、SymPy 1.14.0、psutil 7.2.2。数值运行设置 OPENBLAS_NUM_THREADS=1 与 OMP_NUM_THREADS=1。此为实测版本记录，不是跨平台精确相同的保证。

`requirements-research.txt` 列出主要 Python 依赖；未在干净新环境中验证全历史依赖闭包。标准库快照检查不需要安装这些研究依赖。官方仓库有其独立环境要求，本项目没有宣称两者可在此环境直接完整训练。
