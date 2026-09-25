"""Render a frozen Chinese research delivery from audited 321-323 artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(root, out):
    base = root / 'results/branch_image_chain'
    sources = {name: base / name / 'summary.json' for name in [
        'development_v1', 'geometry_v1', 'audit_v1', 'geometry_audit_v1',
        'conditional_risk_v1', 'opportunity_v1', 'delivery_audit_v1']}
    data = {name: read(p) for name, p in sources.items()}
    assert all(s['passed'] for s in data.values())
    for rel, digest in data['delivery_audit_v1']['input_sha256'].items():
        assert sha(root / rel) == digest
    geo = data['geometry_v1']['aggregate']
    risk = data['conditional_risk_v1']
    opportunity = data['opportunity_v1']
    fraction = data['delivery_audit_v1']['no_trigger_risk_fraction']['257']
    mass = opportunity['aggregate']['dual']
    categories = ['no_trigger', 'outside_window', 'ranked_out']
    fractions = [mass[k]['mean_mass'] / opportunity['mean_missing_mass'] for k in categories]
    colors = ['#C95643', '#D79B38', '#477EAC']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), gridspec_kw={'width_ratios': [1.12, 1]})
    fig.patch.set_facecolor('#FAFBFD')
    ax = axes[0]
    values = [100 * fraction, 100 * (1 - fraction)]
    bars = ax.barh([1, 0], values, color=[colors[0], colors[2]], height=.48)
    ax.set_yticks([1, 0], ['No trigger\n45 tasks', 'Triggered\n19 tasks'])
    ax.set_xlim(0, 106)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Share of remaining conditional excess risk (%)')
    ax.set_title('A  Most remaining risk is never touched', loc='left', pad=18, fontsize=12, weight='bold')
    for b, v in zip(bars, values):
        ax.text(v + 1, b.get_y() + b.get_height()/2, f'{v:.2f}%', va='center', weight='bold')
    ax = axes[1]
    bars = ax.barh([2, 1, 0], [100*x for x in fractions], color=colors, height=.5)
    ax.set_yticks([2, 1, 0], ['No trigger', 'Outside 3-shell\nsearch window', 'Outside top 8\nwithin window'])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Share of missing posterior mass (%)')
    ax.set_title('B  Expanding top-K alone has little reach', loc='left', pad=18, fontsize=12, weight='bold')
    for b, v in zip(bars, fractions):
        ax.text(100*v + 1, b.get_y() + b.get_height()/2, f'{100*v:.2f}%', va='center', weight='bold')
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E2E6EA')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    fig.suptitle('Where should the next unit of search budget go?', x=.04, ha='left', y=.99, fontsize=17, weight='bold')
    fig.text(.04, .035, '64 reused development tasks | offline posterior diagnostic | numerical volumes + Monte Carlo moments\nNot held-out query MSE; not an independent PC-ALM performance win.', fontsize=10, color='#555D68')
    fig.subplots_adjust(left=.13, right=.965, bottom=.23, top=.81, wspace=.55)
    fig.savefig(out / 'bottleneck.png', dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    def link(path, label=None):
        return f'[{label or path.name}](<{path.resolve().as_posix()}>)'

    labels = {'dual': '乘子信用', 'dual_plus_residual': '乘子＋残差',
              'residual': '纯残差', 'bp': 'BP 信用对照', 'random_sign': '随机符号', 'zero': '零信用'}
    rows = ['| 信用来源 | 不同任务-模式提议 | 正体积区域 | 相对旧强控制并集新增 |',
            '|---|---:|---:|---:|']
    for name, values in geo.items():
        rows.append(f"| {labels[name]} | {values['proposal_pairs']} | {values['positive_pairs']} | {values['new_vs_prior']} |")
    novel = risk['novel_regions'][0]
    residual = next(r for r in risk['aggregate'] if r['channel'] == 'residual' and r['grid'] == 257)
    opp = opportunity['risk_opportunity']['257']
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    report = f'''# TTT × PC-ALM：2026-09-25 截稿前交付版 v1

封存时间：{stamp}。硬截止：2026-09-25 12:00 America/New_York（EDT）。本版已完成，不依赖后续实验结束；11:30 的同任务检查只补入已核验的新结果。

## 先给结论

当前得到的是一个**经过实现和独立核验的链式局部证书机制，以及明确的搜索预算瓶颈**。尚未得到“PC-ALM 在匹配资源下独立优于闭式回归、浅层网络或 BP”的任务结果，不能把这一版写成已经胜出的完整论文。

关键发现不是“再多跑几步就可能好”：在固定的 64 个旧开发任务上，现有停滞触发只覆盖 19 个任务。其余 45 个任务贡献了剩余理想池条件超额风险的 **{fraction*100:.2f}%**。如果保持它们的预测不变，即便将有触发任务全部替换为理想后验预测，最多也只能消除约 {(1-fraction)*100:.2f}% 的这项风险。这决定了下一步应先改变上下文驱动的触发与预算分配，而不是继续只扩当前位置的 top-K。

![搜索预算瓶颈]({(out/'bottleneck.png').resolve().as_posix()})

图 A 是条件超额风险的分解；图 B 是尚未进入旧控制并集的后验质量分解，两者不是同一个指标。均为离线开发诊断，不是独立未见查询 MSE，也不是新方法收益。

## 1. 本轮到底做了什么

研究对象为小规模四层 tent 非线性复合模型、在线可适应偏置与局部活动/乘子原型；并非官方 TTT 或官方 PC-ALM 的完整大模型复现。固定观察、先验、参数盒和观察误差带（B=0.12，epsilon=0.001），保留全部 64 个旧任务（5910000–5910063），不删除无触发任务。

阶段 320 将固定信用下的分解证书用于整网分支排序。阶段 321 补入相邻层的分支输出像，在同样 131 个触发位置运行六种信用来源，统一 K=8、从最小未排空 Hamming 距离起的三个距离层。阶段 322 核算新区域的后验质量和理想池风险，阶段 323 将遗漏分解为触发不足、搜索距离不足、排序预算不足。

所有六通道共享结构增强。BP 信用仅用于明确标注的 BP 对照，没有用于初始化乘子候选；近期生成与诊断没有读取真实查询答案。完整后验参考仅用于离线评价，不能免费提供给在线方法。

## 2. 已成立的数学机制

对固定分支模式 R 和信用 a，旧松弛允许各内部活动独立处于 [0,1]。但 tent 的平坦分支输出必为 0。把上一层分支的真实输出像保留到下一层，使新松弛域包含于旧域，因此：

    D_chain(R,a) ≥ D_row(R,a)
    真实支持可行 ⇒ D_chain(R,a) ≤ 0

所以 D_chain>0 或结构域为空可排除该模式；D_chain≤0 仍只是必要条件，不能解释为可行概率或预测质量。严格增强例子是连续选择“平坦分支→要求预激活至少 0.5 的分支”：前层输出 0，而下一层偏置最多 0.12，故不可能。零信用也能得到这个结构改进，因此不能把它归为乘子的独立贡献。全为非平坦分支、输出域相同的模式给出两种下界相等的对照。

精确 K 最佳动态规划状态为（层、Hamming 距离、上一层平坦像掩码）。保留的是跨层约束，不只是增加参数。掩码数 2^n、分支行数 4^n，仍有对观察数指数增长的限制，不能声称已可用于长上下文。

完整设计与证明条件：{link(root/'outputs/ttt-pc-alm-research/321_branch_image_chain_design_v1.md')}。其中“尚未实现”是冻结前的历史表述，当前执行结果以下文为准。

## 3. 实际找到的区域及其价值

主实验完成 131 个位置 × 6 通道，共 18,292 次提议。去重后 3,911 个任务-模式中，3,902 个不可行、9 个有正体积。各通道会重叠，表中区域数不能相加。

{chr(10).join(rows)}

旧强控制并集 S 是阶段 319/320 等已累计的已知正体积区域，来自多个方法，是昂贵的离线参照，不是单一可部署基线。新增 1 个区域来自**纯残差**，不是乘子通道；乘子和乘子＋残差均无相对 S 的新区域。

新增区域位于 seed={novel['seed']}，在该任务完整支持后验中的质量仅 {novel['posterior_mass']:.6e}。把它加入 S 后，64 任务平均条件超额风险变化为 **{residual['mean_delta']:+.6e}**（正数是变差），不是可用的任务改善。两组独立粒子批配对和 129/257 两种网格均得到同方向、同量级的结果；这不构成统计置信区间。

若新增区域归一化混合权重为 alpha，则 mu_new=(1-alpha)mu_S+alpha mu_added。输出位于 [0,1] 时，预测最大改变量≤alpha，平方损失改变量≤2 alpha。该恒等式说明“发现新区域”还必须转化为足够的后验质量或预测价值；本次数值体积没有区间认证，不能把代入数值后的界称为严格浮点区间证书。

## 4. 为什么下一步必须先检查触发机制

设上下文为 C，完整后验均值为 mu(C,q)，旧区域池理想均值为 m_S(C,q)。平方损失的条件超额风险为 E_q[(m_S-mu)^2]。新方法若仅在触发集合 T 内改变预测，则逐上下文的 Bayes 分解给出：

    新方法的平均条件超额风险
        ≥ (1/N) Σ_{{i 不属于 T}} E_q[(m_S(C_i,q)-mu(C_i,q))²]

原因是未触发项不变，触发项即便达到后验均值，其超额风险也只能降到 0。这是关于“固定未触发预测”这一方法类的约束，不是宣称任何方法都不能提升。

在 257 点网格的四批 Monte Carlo 区域矩估计下：

| 项目 | 任务数 | 对全部 64 任务平均值的贡献 |
|---|---:|---:|
| 旧池总条件超额风险 | 64 | {opp['all']['mean_contribution']:.12g} |
| 无触发任务 | 45 | {opp['no_trigger']['mean_contribution']:.12g} |
| 有触发任务 | 19 | {opp['with_trigger']['mean_contribution']:.12g} |

因此无触发部分占 {fraction*100:.2f}%。129 点复算为 {data['delivery_audit_v1']['no_trigger_risk_fraction']['129']*100:.2f}%，但数值来自旧开发集与数值体积/MC 矩，不是新任务上的严格概率界。使用两个独立批的残差乘积减小直接平方的 MC 自噪声，不等于消除所有估计误差。

遗漏后验质量共 0.1265989942（按 64 任务平均）：{fractions[0]*100:.2f}% 来自无触发任务，{fractions[1]*100:.2f}% 在现有三个距离层之外，仅 {fractions[2]*100:.2f}% 在距离层内但排在 top-8 外。该百分比针对乘子通道；纯残差选中的极小区域单独记账。不能根据这些离线归因把重要区域标签泄漏给下一版在线触发器。

## 5. 核验与成本

独立反向下界＋最佳优先搜索核验了实际数据上的 2,358 个 K 最佳距离层、全部 18,292 个提议顺序和下界支配关系。独立几何核验覆盖 15,626 个证书、3,902 个不可行区域、9 个正体积区域及 1,152 组集合比较。

本交付另写独立标量汇总程序，从归档区域矩重新计算 64 任务、768 个风险行、2,574 个覆盖标签和 7,544 个数值字段；最大数值差 {data['delivery_audit_v1']['max_numeric_gap']:.3e}。它复用已审核的区域矩，不重复声称完成一次独立粒子采样或新的查询实验。

主选择实验约 {data['development_v1']['seconds']:.2f} 秒，几何约 {data['geometry_v1']['seconds']:.2f} 秒，独立选择器审计约 {data['audit_v1']['seconds']:.2f} 秒，独立几何审计约 {data['geometry_audit_v1']['seconds']:.2f} 秒。这些是本机诊断计时，不能作为净加速结论。缓存前缀、表项、乘子、活动、求解器状态和离线后验参考均不能算免费；本轮尚无完整的公平在线成本优势证明。

## 6. 中午前剩余工作的明确边界

下一实验问题应是：**仅利用已观察上下文，能否在相同总搜索预算下识别并覆盖当前从不触发、但后验仍有多分支不确定性的情形？** 先冻结上下文触发规则与预算，不用上述离线“重要任务”清单挑例子；六信用通道共享新触发器，先区分触发收益和乘子收益，再与强回归/同参数优化器比较。

这只是下一步假设，当前没有实现或收益结果。中午前不为追求正向结论而扩大为 LLM/VLM 实验、不改旧测试集、不把报告截止当成科学结论的门槛。到 11:30 汇总当时已完成且已核验的内容；未完成实验列出实际进度。本版随时可交付，整体研究目标仍未完成。

## 7. 复现和证据入口

- {link(base/'development_v1/summary.json', '主实验摘要')}；{link(base/'geometry_v1/summary.json', '区域几何摘要')}。
- {link(base/'audit_v1/summary.json', '独立选择器审计')}；{link(base/'geometry_audit_v1/summary.json', '独立几何审计')}。
- {link(base/'conditional_risk_v1/summary.json', '新增区域风险核算')}；{link(base/'opportunity_v1/summary.json', '触发与搜索机会分解')}。
- {link(base/'delivery_audit_v1/summary.json', '交付版独立标量复算')}；{link(root/'work/experiments/audit_chain_delivery_v1.py', '复算代码')}；{link(Path(__file__), '图文生成代码')}。

在项目根目录使用现有 Python 环境，给新的输出目录运行（不覆盖已封存结果）：

```powershell
& 'C:/Users/callofthenight/AppData/Local/Programs/Python/Python314/python.exe' work/experiments/audit_chain_delivery_v1.py --out results/branch_image_chain/delivery_audit_replay
& 'C:/Users/callofthenight/AppData/Local/Programs/Python/Python314/python.exe' work/experiments/report_chain_delivery_v1.py --out results/branch_image_chain/delivery_report_replay
```

复现依赖本项目已归档的观察、轨迹、几何和区域矩，不是脱离这些输入的独立安装包。输入哈希见审计文件和本目录 manifest.json。当前交付为本地文件；没有据此声称已推送 GitHub或完成投稿。
'''
    (out / 'report.md').write_text(report, encoding='utf-8')
    manifest = dict(created_utc=stamp, deadline_utc='2026-09-25T16:00:00Z',
                    source_sha256=sha(Path(__file__)), inputs={str(p.relative_to(root)): sha(p) for p in sources.values()},
                    outputs={name: sha(out/name) for name in ['report.md', 'bottleneck.png']},
                    figure_values=dict(risk_percent=[100*fraction, 100*(1-fraction)], missing_mass_percent=[100*x for x in fractions]),
                    independent_task_gain_established=False, visual_qa_required=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(dict(report=str(out/'report.md'), figure=str(out/'bottleneck.png')), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    run(Path(__file__).resolve().parents[2], args.out)
