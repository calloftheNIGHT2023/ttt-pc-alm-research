"""Report 295 without changing predictions, metrics, or the frozen diagnosis."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_counterfactual_conditional_risk_v1 import read,save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    root=parser.parse_args().project.resolve()
    inp=root/'results/counterfactual_conditional_risk/development_v2'
    out=root/'results/counterfactual_conditional_risk/report_v1'
    out.mkdir(parents=True,exist_ok=False)
    summary=read(inp/'summary.json')
    assert summary['passed']
    for name,digest in summary['outputs_sha256'].items():
        assert sha(inp/name)==digest
    agg=read(inp/'aggregates.json')
    actual={r['method']:r for r in agg['actual'] if r['grid']==257}
    ideal={r['pool']:r for r in agg['ideal'] if r['grid']==257}
    delta={r['method']:r for r in agg['contrasts'] if r['grid']==257}
    proposal=read(root/'results/counterfactual_branch_proposals/development_v1/summary.json')
    names=['original_alm','counterfactual','bp240']
    labels=['Original ALM','Mode-change proposal','Same-probe Adam240']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    colors=['#2563eb','#f59e0b','#10b981']
    bottom=np.zeros(3)
    for metric,label,color in zip(['policy_excess','read_error','cross_term'],
                                   ['Missing-pool bias','Finite readout error','Cross term'],colors):
        values=np.array([actual[n][metric]['mean'] for n in names])
        axes[0].bar(labels,values*1e3,bottom=bottom*1e3,label=label,color=color)
        bottom+=values
    axes[0].set_ylabel('Conditional excess risk (x 0.001)')
    axes[0].set_title('Most remaining error is missing-pool bias')
    axes[0].tick_params(axis='x',labelrotation=15)
    axes[0].legend(loc='upper left',fontsize=8)
    axes[0].set_ylim(0,.00265*1e3)
    rules=['all_steps','changed_forward_mode','first_global_forward_mode']
    rulelabels=['All steps','Mode changes','First global mode']
    changes=np.array([ideal[n]['delta_vs_original']['mean'] for n in rules])
    axes[1].barh(rulelabels,changes*1e6,color=['#64748b','#2563eb','#10b981'])
    for y,n in enumerate(rules):
        pair=np.array(ideal[n]['delta_vs_original']['pair_means'])*1e6
        axes[1].plot(pair,[y,y],'k.-',linewidth=1)
        count=proposal['rules'][n]['mean_proposals']
        axes[1].text(-42,y,f'{count:.1f} shadow state-steps/task',va='center',fontsize=9)
    axes[1].axvline(0,color='black',linewidth=.7)
    axes[1].set_xlim(-92,3)
    axes[1].set_xlabel('Ideal pool-risk change vs original ALM (x 0.000001)')
    axes[1].set_title('All three pre-existing rules retained')
    axes[1].invert_yaxis()
    fig.suptitle('OLD64 development diagnosis | 257-point grid | no query targets',fontsize=13)
    fig.savefig(out/'risk_decomposition.png',dpi=170)
    plt.close(fig)
    old,new=actual['original_alm'],actual['counterfactual']
    policy_share=new['policy_excess']['mean']/new['actual_excess']['mean']
    read_share=new['read_error']['mean']/new['actual_excess']['mean']
    rel=-delta['counterfactual']['actual_excess']['mean']/old['actual_excess']['mean']
    text=['# 295｜定位到了主要瓶颈：遗漏分支，而不是粒子均值读出',
      '', '这是旧 64 个开发任务的事后机制诊断，不是新确认实验。使用已审计的完整数值后验参照；没有读取真实教师或实际查询答案。',
      '',f'候选方法当前条件超额风险的约 {policy_share:.2%} 来自已发现池的截断偏差，有限读出误差约占 {read_share:.2%}。这两个比例是带交叉项的诊断比值，不是独立方差分量。下一步优先改进支持可观测的分支搜索。',
      '', '![条件风险分解](risk_decomposition.png)', '',
      '## 1．主要数值', '',
      '| 已保存的实际方法 | 条件超额风险 | 池截断偏差 | 有限读出误差 | 交叉项 |',
      '|---|---:|---:|---:|---:|']
    for n,label in zip(names,labels):
        text.append('| '+label+' | '+' | '.join(f"{actual[n][m]['mean']:.10g}" for m in ['actual_excess','policy_excess','read_error','cross_term'])+' |')
    text+=['',f'新候选相对原 ALM 的条件超额风险差为 {delta["counterfactual"]["actual_excess"]["mean"]:.10g}，约减少 {rel:.2%}。池截断项差为 {delta["counterfactual"]["policy_excess"]["mean"]:.10g}，解释了几乎全部变化。这里的百分比仅针对旧64条件超额风险，不是总 MSE，也不是294新128任务的提升。',
      '', '294 的新128任务总 MSE 改善仍只有约 0.038%，原先跨零的区间没有因本诊断而改变。Adam240 仅是本诊断的同探测起点对照，不能替代更长强 Adam 或原8192任务结论。',
      '', '## 2．三个已存在的提议规则：不能只数新分支', '',
      '| 理想池规则 | 平均后验质量覆盖 | 理想条件超额风险 | 相对原池差 | 额外影子状态步/任务 |',
      '|---|---:|---:|---:|---:|']
    for n,label in zip(rules,rulelabels):
        r=ideal[n]
        text.append(f'| {label} | {r["mass"]["mean"]:.8f} | {r["policy_excess"]["mean"]:.10g} | {r["delta_vs_original"]["mean"]:.10g} | {proposal["rules"][n]["mean_proposals"]:.5f} |')
    text+=['','First-global 的影子步数约为 mode-change 的14%，但在这个旧开发集上理想池风险下降并不更小。这是减少重复搜索、将预算投入新分支的动机，不是上线速度优势；该规则尚无实际完整调用和新任务结果。所有规则都必须支付检测、几何和读出成本。',
      '', '## 3．数学解释', '',
      '令当前已发现池的均值为 μ，新增池均值为 ν，新增池在两者并集中的后验质量比例为 α，完整后验均值为 m。并集均值为 μ′=μ+α(ν−μ)。平方损失下，理想条件超额风险变化严格满足：',
      '', r'\[\|\mu\prime-m\|_Q^2-\|\mu-m\|_Q^2=2\alpha\langle\mu-m,\nu-\mu\rangle_Q+\alpha^2\|\nu-\mu\|_Q^2.\]',
      '', '因此，增大覆盖质量本身不保证每个任务的风险下降；新增区域的函数均值方向同样重要。完整 m 只能作昂贵离线诊断，不能成为在线选择器的免费输入。下一步的触发规则只能使用支持样本、局部状态、模式历史和已计费计算。',
      '', '固定实际预测 g 的分解使用两批独立区域矩：(g−mᵃ)(g−mᵇ) 的网格平均，严格拆成池偏差、读出误差及交叉项。两对批次固定为(0,1)、(2,3)，小负估计保留。有限 Monte Carlo 与数值体积仍是近似，不是精确实数积分。',
      '', '## 4．全量网格与批次敏感性', '',
      '下列数值完整保留所有方法、两种网格、两对批次。批次差反映参照的 Monte Carlo 敏感性，不是任务采样置信区间；图中黑线同理。']
    for group,key,metrics in [('actual','method',['actual_excess','policy_excess','read_error','cross_term']),
                              ('ideal','pool',['policy_excess','delta_vs_original','mass']),
                              ('contrasts','method',['actual_excess','policy_excess','read_error','cross_term'])]:
        text+=['',f'### {group}', '', '| 方法/池 | 网格 | 指标 | 两对均值 | 对(0,1) | 对(2,3) | 对间差 |',
               '|---|---:|---|---:|---:|---:|---:|']
        for r in agg[group]:
            for metric in metrics:
                v=r[metric]
                text.append(f'| {r[key]} | {r["grid"]} | {metric} | {v["mean"]:.12g} | {v["pair_means"][0]:.12g} | {v["pair_means"][1]:.12g} | {v["pair_difference"]:.12g} |')
    text+=['','## 5．执行与复核', '',
      f'64任务、768分解、1280理想池估计、512直接风险对比均完成。恒等式最大残差 {summary["max_identity_gap"]:.4g}，独立标量求和差 {summary["max_independent_scalar_gap"]:.4g}，直接风险差核对误差 {summary["max_direct_contrast_gap"]:.4g}。',
      '', '初次 development_v1 在结果计算前遇到旧协议 source_sha256 的格式兼容错误并停止；失败记录保留。修复只兼容标量哈希和相对路径哈希，完整结果在 development_v2；未改任务、规则、指标或容差。',
      '', '这些结果支持搜索改进的优先级，但尚不能证明 PC-ALM 相对强 BP 的独立优势，也不是通用 MLP、官方 TTT 或 LLM/VLM 验证。总目标未完成。',
      '', f'结果摘要 SHA256：`{sha(inp/"summary.json")}`。逐任务风险、批次矩、全部输入哈希和源代码哈希见同级 development_v2。']
    with (out/'report.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(text)+'\n')
    save(out/'summary.json',dict(passed=True,diagnosis_summary_sha256=sha(inp/'summary.json'),
        source_sha256=sha(Path(__file__)),visual_review_pending=True,
        outputs_sha256={n:sha(out/n) for n in ['report.md','risk_decomposition.png']}))
    print(json.dumps(dict(report=str(out/'report.md'),policy_share=policy_share,read_share=read_share,
                          conditional_excess_relative_drop=rel)),flush=True)


if __name__=='__main__':
    main()
