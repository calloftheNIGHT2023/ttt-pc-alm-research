# 显式局部信用实现与重放验证

## 局部公式

令隐藏约束

$$r_l=h_l-T_{a_l}(h_{l-1}),\qquad l=1,\ldots,L-1,$$

并令局部约束信用

$$c_l=\lambda_l+\rho r_l.$$

tent map 的分支导数为

$$
\frac{\partial T_a(x)}{\partial x}=
\begin{cases}1/a,&x\le a,\\-1/(1-a),&x>a,\end{cases}
$$

$$
\frac{\partial T_a(x)}{\partial a}=
\begin{cases}-x/a^2,&x\le a,\\(1-x)/(1-a)^2,&x>a.\end{cases}
$$

活动更新只依赖相邻层。对中间活动，

$$
\nabla_{h_l}\mathcal L_{\mathrm{AL}}
=c_l-c_{l+1}\frac{\partial T_{a_{l+1}}(h_l)}{\partial h_l};
$$

对最后一个自由活动，

$$
\nabla_{h_{L-1}}\mathcal L_{\mathrm{AL}}
=c_{L-1}+(\hat y-y)\frac{\partial T_{a_L}(h_{L-1})}{\partial h_{L-1}}.
$$

隐藏层参数信用为

$$
g_l=-\mathbb E[c_l\,\partial T_{a_l}(h_{l-1})/\partial a_l]\,\frac{da_l}{du_l},
$$

输出层参数信用为

$$
g_L=\mathbb E[(\hat y-y)\,\partial T_{a_L}(h_{L-1})/\partial a_L]\,\frac{da_L}{du_L}.
$$

其中 $a_l=0.35+0.30\,\sigma(u_l)$。每项只读取本层输入、输出、残差和乘子，不需要把输出误差沿整网链式传播。乘子仍按 $\lambda_l\leftarrow\lambda_l+\alpha r_l$ 更新。

## 数值等价验证

显式公式与停止梯度的 autograd 参考实现逐项比较：

| 项目 | 最大绝对差异 |
|---|---:|
| $\partial T/\partial x$ | 0 |
| $\partial T/\partial a$ | 0 |
| 三层活动更新 | $5.55\times10^{-17}$ |
| 普通 PC 参数信用 | $8.67\times10^{-19}$ |
| PC-ALM 参数信用 | $2.39\times10^{-18}$ |

随后用显式后端重放 seeds 1000000–1000063 的 64 个锁定 episode。PC 与 PC-ALM 的中位和 90 分位查询误差在报告精度下与参考实现相同；100 次迭代后，单 episode 查询 MSE 的最大绝对差异为 $1.17\times10^{-5}$。

这证明当前 PC/PC-ALM 数值结果不依赖全局 autograd 信用。它仍不是高性能系统结论：Python 原型没有证明墙钟时间、显存或硬件并行优势，BP 对照仍正常使用反向传播。

## 当前解释

PC-ALM 已经实现了本项目要求的“在不先运行 BP、也不用 BP 初始化乘子的情况下进行内部多层适应”。在当前合成任务上，它优于闭式固定特征头、核岭和普通 PC；但尚未显著优于 Adam-BP，因此合理主张是“在禁止全局 BP 的局部信用约束下仍能实现有用的深层在线适应”，而不是“普遍优于 BP”。

