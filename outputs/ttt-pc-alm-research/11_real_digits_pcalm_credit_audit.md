# 真实数字任务上的 PC-ALM 局部信用审计

## 1. 目的与状态

本阶段只回答：能否在不先运行 BP、不用 BP 初始化乘子、且六参数只读取相邻约束的条件下，把标签信用传到输入仿射适配器。所有优化试验只用验证 split 的4个 episode；没有为 PC-ALM 再次访问128-episode确认集。

结论是：**局部信用实现正确，瞬时方向可以接近 BP，但有限步任务效果仍明显差于匹配 BP，第二关未通过。**

## 2. 约束化的冻结 CNN

把冻结计算路径拆为

\[
\begin{aligned}
h_0 &= \operatorname{warp}(x;u),\\
h_1 &= \operatorname{ReLU}(\operatorname{Conv}_1(h_0)),\\
h_2 &= \operatorname{ReLU}(\operatorname{Conv}_2(h_1)),\\
h_3 &= \operatorname{ReLU}(\operatorname{Conv}_3(h_2)),\\
h_4 &= \operatorname{ReLU}(W_ph_3+b_p),\\
z &= W_ch_4+b_c.
\end{aligned}
\]

其中 (u\in\mathbb R^6) 是唯一在线参数，卷积、投影和分类器全部冻结。令

\[
r_0=h_0-\operatorname{warp}(x;u),\qquad
r_l=h_l-f_l(h_{l-1}),\quad l=1,\ldots,4,
\]

增广拉格朗日能量为

\[
\mathcal L_{\mathrm{AL}}
=\operatorname{CE}(W_ch_4+b_c,y)
+\frac1n\sum_{i,l}
\left[\lambda_{l,i}^{\top}r_{l,i}
+\frac\rho2\lVert r_{l,i}\rVert_2^2\right].
\]

记 (c_l=\lambda_l+\rho r_l)。中间活动的信用只读取相邻约束，形式为

\[
\nabla_{h_l}\mathcal L_{\mathrm{AL}}
=c_l-J_{f_{l+1}}(h_l)^\top c_{l+1},
\]

最后活动再加分类损失的局部梯度。六参数更新只来自第一条约束：

\[
g_u=-\frac1n\sum_i
J_{\operatorname{warp}}(x_i;u)^\top c_{0,i}.
\]

因此计算 (g_u) 时不读取标签、分类器或更深活动；标签只通过交替活动/乘子更新逐层传播到 (h_0,\lambda_0)。

## 3. 正确性与局部性审计

在验证 seed 4100000、每类2个上下文的20点批次上：

- 1、2、4次活动传播后，信用尚未到达第一条约束，六参数梯度严格为0；
- 8次传播后，局部信用与全局 BP 梯度余弦为0.993；16次为0.980；32次为0.933；
- 固定 (h_0,\lambda_0) 后任意更改标签，六参数局部梯度变化严格为0；
- 在避开恒等仿射恰好落于双线性采样网格折点的普通位置，局部 autograd 与中心有限差分余弦为1.000，步长 (10^{-4}) 时最大绝对差 (3.07\times10^{-4})；
- PC-ALM 活动和乘子从冻结前向与全零状态初始化，不读取 BP 信号。BP 仅在审计结束后作为外部方向参考。

恒等仿射处的中心差分不能用作严格参考，因为采样点正好位于分段双线性插值的折点，导数不唯一；这不影响 autograd 路径与实际 BP 对照使用同一个规定子梯度。

## 4. 状态与传播成本

20个上下文时，各活动标量数为：

| 活动 | 形状（省略 batch） | 标量数 |
|---|---:|---:|
| (h_0) | 1×16×16 | 5,120 |
| (h_1) | 32×16×16 | 163,840 |
| (h_2) | 48×8×8 | 61,440 |
| (h_3) | 64×4×4 | 20,480 |
| (h_4) | 64 | 1,280 |
| 合计 | — | 252,160 |

乘子另需252,160个标量。30上下文正式适应时二者合计756,480个 float32，约3.03 MB，尚未计临时图和梯度。每做一次六参数更新还需8–32次全层局部活动 sweep；这远大于六参数 Adam 的12个矩状态。

## 5. 验证集优化试验

先固定活动 (ho=\alpha=1)。状态步长0.1在16次传播时发散；0.03在8次传播方向好但32次开始不稳；0.01可稳定到32次。因此只做如下有限探索，不继续按查询结果无界调参。

同一4个验证 episode、40次六参数更新：

| 局部信用/优化 | 活动设置 | PC-ALM平均 Brier | 匹配 BP平均 Brier |
|---|---|---:|---:|
| Adam，参数步长0.01 | 8步，状态步长0.03 | 0.267 | 0.0805 |
| Adam，参数步长0.02 | 8步，状态步长0.03 | 0.280 | 0.0479 |
| Adam，参数步长0.05 | 8步，状态步长0.03 | 0.283 | 0.0543 |
| 归一化 SGD，参数步长0.02 | 8步，状态步长0.03 | 0.278 | 0.0367 |
| 归一化 SGD，参数步长0.02 | 16步，状态步长0.01 | 0.235 | 0.0367 |
| 归一化 SGD，参数步长0.02 | 32步，状态步长0.01 | **0.156** | **0.0370** |

增加活动传播改善了 PC-ALM，但最好设置仍比匹配 BP 差0.119 Brier。瞬时梯度余弦高并不保证非凸轨迹一致：早期的小角度误差会进入不同区域，之后“局部信用与当前位置 BP 相似”也不能追回 BP 的轨迹。

## 6. 决策

- 不把“局部信用方向接近 BP”写成任务收益。
- 不在确认集上继续调 PC-ALM；当前真实任务的已确认赢家仍是六参数 Adam-BP。
- PC-ALM 若要继续，必须先在验证集上通过明确门槛，例如在固定预算下达到匹配 BP Brier 的110%以内，并同时报告活动/乘子状态和同步延迟。
- 合理的下一项数学工作是给不同层按局部 Jacobian/Lipschitz 尺度设置可推导的预条件，而不是扩大盲目网格；否则高维活动的统一步长同时面临“信用传不到输入”和“深层状态发散”。

## 7. 文件

- `work/experiments/digits_pcalm_credit.py`
- `work/experiments/digits_pcalm_optimizer_pilot.py`
- `results/digits_domain_shift/pcalm_credit_audit/`
- `results/digits_domain_shift/pcalm_credit_lr_0.03/`
- `results/digits_domain_shift/pcalm_credit_lr_0.1/`
- `results/digits_domain_shift/pcalm_optimizer_pilot*/`
