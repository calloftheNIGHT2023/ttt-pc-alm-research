# 439｜Claude Code 接手：R3 局部回归强基线结果与 RunPod 直连

2026-09-25 19:05 EDT。Codex 线程在 22:53 UTC 因 401 认证错误中断，三个子 Agent 同样以 401 终止。用户要求由 Claude Code 接手同一研究目标。整体 Goal 仍 ACTIVE，截稿提醒保持 PAUSED。

## 子 Agent 实际留下的产物

| 线 | 438 分配任务 | 中断时实际产物 |
| --- | --- | --- |
| R（regression_ttt_related_work） | 8 局部回归 × 4 旧任务 × 4 前缀 × 2 预算 = 256 调用 | `run_local_linear_ml_v1.py`、`local_linear_ml_protocol_v1.md` 已写完，未执行 |
| L（local_credit_related_work） | 完整 support-cover | 无新文件 |
| T（anytime_transfer_related_work） | 原生 ALM 乘子→查询方向证书 | 无新文件 |

## R3 256 调用：已执行并通过

根端确认无并发 Python 实验进程后执行：

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'
& 'C:/Users/callofthenight/AppData/Local/Programs/Python/Python314/python.exe' work/parallel_research/run_local_linear_ml_v1.py all --exclusive-numerical-run
```

EXIT=0。run 106.5 秒，256/256 timely final；audit 16.8 秒 PASS（32896 独立 query WLS、1152 真 LOO 分数、512 包重放、真实 checkpoint bank 重绑）；evaluate 在审计后才读取 query truth。输出：`results/local_linear_ml/{development_v1,audit_v1,evaluation_v1}`。

| 方法 | .25 秒 MSE | .5 秒 MSE |
| --- | ---: | ---: |
| local_linear_raw_fixed | 0.169732 | 0.169732 |
| local_linear_raw_loo | 0.146904 | 0.146904 |
| local_linear_prior_fixed | 0.173656 | 0.173656 |
| local_linear_prior_loo | 0.148572 | 0.148572 |
| local_constant_raw_fixed | 0.129360 | 0.129360 |
| local_constant_raw_loo | 0.130910 | 0.130910 |
| local_constant_prior_fixed | 0.129075 | 0.129075 |
| local_constant_prior_loo | 0.131931 | 0.131931 |

解读边界：这是补充开发轮，不与 427/436 同场随机顺序。仅作跨运行参照：同四任务 .5 秒下 427 主方法 0.017505、强 BP 组合 0.017464、cohort_meta_ridge128 0.043721、官方 TTT prior256 0.046827。局部回归（TTR 式 p=0/p=1，含 LOO 选参与 prior 残差）远差于已有闭式强基线，更差于分支搜索类方法，**不能解释主方法相对闭式基线的收益**。p=1 比 p=0 更差，符合支持稀疏、跨折点时局部斜率估计方差大的预判。

## RunPod 直连

`ssh.runpod.io` 网关必须 `-tt` 交互；pod 同时暴露直连 TCP，已验证免密、支持非交互命令和 scp：

```bash
ssh -i ~/.ssh/runpod_verifier_ed25519 -p 11804 root@69.8.146.195
```

`runpod_verifier_ed25519.pub` 已在 pod 的 `/root/.ssh/authorized_keys`。不使用 RunPod API（用户要求）。pod 重启后 IP/端口可能变化，需从 pod 内 `env | grep RUNPOD_` 或控制台重新获取。

## 接续

R3 完成后，438 队列剩余：L3 完整 support-cover、T3 原生 ALM 乘子证书、独立新任务确认、389 三域六数据集真实小模型。方向选择待用户确认（见本轮给用户的建议）。
