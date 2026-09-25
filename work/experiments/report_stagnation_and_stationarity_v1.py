"""300-301 full old-development report, exact KKT finding, no success relabeling."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_counterfactual_conditional_risk_v1 import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve()
    inp=root/'results/stagnant_trigger_exploration/evaluation_v2'
    census=root/'results/projected_stationarity/development_v1'
    audit=root/'results/projected_stationarity/audit_v1'
    out=root/'results/stagnant_trigger_exploration/report_v1';out.mkdir(parents=True,exist_ok=False)
    for folder in [inp,census,audit]:
        ss=read(folder/'summary.json');assert ss['passed']
        for name,digest in ss['outputs_sha256'].items():assert sha(folder/name)==digest,name
    rows=read(inp/'aggregates.json');lookup={(r['method'],r['grid']):r for r in rows}
    cs=read(census/'summary.json');au=read(audit/'summary.json')
    certs=read(audit/'certificates.json')
    witness=next(r for r in certs if r['smooth_box_local_minimum'])
    with np.load(census/'metrics.npz',allow_pickle=False) as z:
        values=z['values'][witness['seed']-5910000]
    phase,step,origin=witness['location']
    selected=values[(values[:,0]==phase)&(values[:,2]==origin)]
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.6),layout='constrained')
    names=['background__original_alm','union__dual_union','union__residual_union','union__random_sign_union',
           'union__alm_reset_8','union__nodual_8','union__pc_8','union__adam_8']
    labels=['Original pool','Dual / 7 amplitudes','Residual / 7 amplitudes','Random signs / 7 amplitudes',
            'ALM reset / 8 steps','No dual / 8 steps','Ordinary PC / 8 steps','Adam / 8 steps']
    axes[0].barh(labels,[1000*lookup[n,257]['policy_excess'] for n in names],
        color=['#64748b','#2563eb','#f59e0b','#10b981','#0ea5e9','#94a3b8','#94a3b8','#8b5cf6'])
    axes[0].invert_yaxis();axes[0].set_xlim(0,1.42)
    axes[0].set_xlabel('Ideal conditional excess risk (x 0.001; lower is better)')
    axes[0].set_title('300: same-location union, OLD64\nNot matched complete-call resources',fontsize=11)
    axes[1].semilogy(selected[:,1],np.maximum(selected[:,4],1e-16),'.-',label='Ordinary gradient max')
    axes[1].semilogy(selected[:,1],np.maximum(selected[:,5],1e-16),'.-',label='Projected gradient max')
    states=[r['location'][1] for r in certs if r['seed']==witness['seed'] and r['location'][0]==phase
            and r['location'][2]==origin and r['smooth_box_local_minimum']]
    axes[1].scatter(states,[1e-16]*len(states),marker='s',facecolors='none',edgecolors='#ef4444',s=90,label='Exact box local minimum')
    axes[1].set_xlabel('Original ALM step-before index (1-based)')
    axes[1].set_ylabel('Gradient norm; display floor = 1e-16')
    axes[1].set_title(f'301: first support-ordered certificate\nSeed {witness["seed"]}, phase {phase}, origin {origin}',fontsize=11)
    axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.18))
    axes[1].grid(alpha=.2)
    fig.suptitle('Trigger diagnosis: real constrained traps exist, but the tested triggers missed them',fontsize=12)
    fig.savefig(out/'stagnation_and_stationarity.png',dpi=170);plt.close(fig)
    text=['# 300–301｜同位置探索全量结果与受约束局部极小点证据','',
      '结论：300没有建立局部信用的独立收益；301则确认原轨迹确实包含此前触发遗漏的盒约束局部极小点。这个数学发现定位了更具体的优化障碍，不等于已经找到相对强BP、回归或浅层头的在线任务优势。',
      '', '![同位置结果与精确驻点见证](stagnation_and_stationarity.png)', '',
      '## 300：改换触发位置后的全部对照', '',
      '64个旧开发任务；2068个模式停留位置、38个参数近停滞位置，重合1个，并集2105个。参数近停滞组在47个任务中为空，按协议保留原池。相同位置复制相同b/h/u/best；候选从未使用BP初始化，Adam仅为对照。',
      '', '三种七幅度方向并集与五类1/2/4/8步续接，共23配置×3位置组=69个新池。298全部46背景配置原样保留并加background前缀；它们不是同触发成本对照。63个单幅度分组提议也已存档，但未事后挑幅度进入主要评分。所有支持提议先封存，才读旧后验区域矩。',
      '', '并集位置的257网格理想超额风险：原池 .00130672362865，乘子七幅度 .00130643389918，等范数残差 .00129017476064，随机符号 .00127474594463，ALMreset8 .00122644793164，Adam8 .00116739139025。原乘子方向仅增加2个正体积区域，而Adam8增加30个；范围是这64个旧任务，不是普遍结论。',
      '', '端点自检：192组α=0的b/h、64组α=1的原ALM下一步、64组keep1、64组reset1/nodual1、576组幅度集合并、1472组选择集并均通过。风险内积另用math.fsum核验，最大误差3.47e−18；46背景配置重放误差为0。',
      '', '首版脚本在提议封存后错误读取q257字段而中止；实际存档名为q。完整失败记录与源保留，evaluation_v2只修复评分读取，直接复用原封存提议，未重跑适应或改变配置。',
      '', '## 301：不能把普通梯度非零等同于没有局部极小点', '',
      '参数盒为C=[−.12,.12]^4，令g=∇L(b)，G=b−Π_C(b−g)。可微点满足G=0当且仅当内部坐标g_j=0、下界坐标g_j≥0、上界坐标g_j≤0。故普通梯度非零仍可在盒约束下驻定。',
      '', '当所有前向预激活远离折点，存在一个邻域使f_b(x)关于b仿射。平方噪声带损失是该仿射映射的凸函数，因此L(b+d)≥L(b)+gᵀd；盒KKT保证对所有合法小d有gᵀd≥0。这证明相对于盒约束的局部极小性，允许平坦方向，不证明全局最优。',
      '', '独立精确核验还给出正邻域半径：每个预激活的折点距离除以其参数敏感度一范数的两倍，取全层全样本最小值。由逐层归纳，在此半径内所有激活分支不变。',
      '', '| 支持侧检查 | 全部69632位置 | 本轮2105触发并集 |','|---|---:|---:|',
      '| 浮点投影梯度为零且支持非可行 | 23 | 0 |',
      '| 投影梯度≤1e−12且支持非可行 | 35 | 0 |',
      '| 投影梯度≤1e−8且支持非可行 | 62 | 0 |',
      '| 完整原始b/h变化≤1e−8且残差非零、支持非可行 | 31 | 0 |',
      '| 精确光滑盒约束局部极小点 | 24 | 0 |','',
      f'对全部62个近驻点位置做有理检查，24个精确通过，来自{au["certified_tasks"]}个任务、{au["unique_certified_parameter_states"]}个不同参数状态。任务分别是5910029(8位置)、5910037(6)、5910039(9)、5910058(1)。不能将24位置当成24个独立任务。其余38个未获精确KKT证书；本轮未逐个有理枚举所有69632状态，不能声称找全所有驻点。',
      '', f'独立前向敏感度有理实现核对全部62位置、248梯度坐标，与反向计算逐个分数完全一致。24个证书均无折点，最小证明邻域半径为{au["minimum_positive_radius"]:.12g}。这比浮点近零的说法严格，也没有读查询或后验参照。',
      '', '图右仅取按seed/phase/step/origin顺序出现的第一个证书作说明，不按风险或效果选例。为能画对数轴，零值显示在1e−16；证书本身是有理等式而非这个显示阈值。',
      '', '## 这对机制意味着什么', '',
      '“参数变化小”“模式停留”“盒投影驻点”和“完整原始块不动点”不是同一个条件。当前触发常在更早的位置用尽一次预算，且沿平坦方向的参数移动不意味着预测或可优化方向还在变化。因此300不能验证246定理所需的真正坏局部分支情形。',
      '', '不过301的KKT诊断使用全链梯度，只能作为离线机制证据；不能把它偷换成无BP候选的线上触发。下一步先检查仅由边界活动集与支持预测变化构成的事件能否识别这些位置，再用同状态扰动/重启BP和局部对照测试逃逸。即便局部信用能逃逸，也还要计入原始轨迹、检测、几何及读出成本，重新冻结新任务与强回归/浅层头比较。',
      '', '246关于乘子压力打破完整原始块不动点的条件仍不能直接由参数KKT推出。本轮没有新官方TTT、一般GELU MLP或LLM/VLM实验；294新128任务及旧8192强BP结论保持不变。',
      '', '## 完整结果：115方法×2网格', '',
      '两对批次是区域矩Monte Carlo的独立内积估计，不是任务采样置信区间；数值体积不是精确实数体积。额外状态步不是完整墙钟时间。所有方法均保留。','',
      '| 方法 | 网格 | 理想超额风险 | 相对原池差 | 对01差 | 对23差 | 覆盖质量 | 状态步/任务 | 新正区域/任务数 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        text.append(f'| {r["method"]} | {r["grid"]} | {r["policy_excess"]:.12g} | {r["delta"]:.12g} | {r["pair_delta_means"][0]:.12g} | {r["pair_delta_means"][1]:.12g} | {r["mass"]:.10g} | {r["shadow_steps"]:.5f} | {r["new_positive_modes"]} / {r["tasks_with_new_positive_modes"]} |')
    text += ['',f'300评分摘要SHA256：`{sha(inp/"summary.json")}`。',
             f'301普查摘要SHA256：`{sha(census/"summary.json")}`。',
             f'301独立核验摘要SHA256：`{sha(audit/"summary.json")}`。',
             '研究目标继续active；本报告是开发与数学证据，不将其标作核心收益已完成。']
    with (out/'report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(text)+'\n')
    save(out/'summary.json',dict(passed=True,source_sha256=sha(Path(__file__)),aggregate_rows=len(rows),
        input_sha256={str(p.relative_to(root)):sha(p) for p in [inp/'summary.json',census/'summary.json',audit/'summary.json']},
        visual_review_pending=True,outputs_sha256={n:sha(out/n) for n in ['report.md','stagnation_and_stationarity.png']}))
    print(dict(report=str(out/'report.md'),figure=str(out/'stagnation_and_stationarity.png'),aggregate_rows=len(rows)),flush=True)


if __name__=='__main__':main()
