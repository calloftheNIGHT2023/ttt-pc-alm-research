"""375 complete matched-state warm-credit experiment with charged generation."""
from pathlib import Path
import math
import numpy as np
from run_history_certificate_v1 import read,sha,save,complete,BASE,model
from report_search_radius_development_v1 import table,write


def main():
    root=Path(__file__).resolve().parents[2];base=root/BASE;summary=complete(base/'development_v1');audit=complete(base/'audit_v1')
    assert audit['development_summary_sha256']==sha(base/'development_v1/summary.json')
    rows=read(base/'development_v1/rows.json');groups=read(base/'audit_v1/groups.json');pairs=read(base/'audit_v1/paired.json');traces=read(base/'audit_v1/checkpoints.json')
    for path,h in read(base/'development_v1/protocol.json')['source_sha256'].items():assert sha(root/path)==h
    indexed={(g['method'],g['source_status']):g for g in groups};data=[]
    for name in model.METHODS:
        failed=indexed[name,'budget_exhausted'];done=indexed[name,'complete'];pp=[r for r in pairs if not r['source_completed'] and r['control']==name]
        data.append([name,failed['certified'],f"{failed['mean_total_seconds']:.6f}",done['certified'],f"{done['mean_total_seconds']:.6f}",
            sum(r['primary_only'] for r in pp),sum(r['control_only'] for r in pp)])
    curve=[]
    for step in [0,32,64,128,256,512,1024]:
        rr=[r for r in traces if not r['source_completed'] and r['method']=='cold_zero1024' and r['step']==step]
        assert len(rr)==16
        curve.append(dict(step=step,certified=sum(r['certified'] for r in rr),mean_component_checkpoint_seconds=math.fsum(r['generation_plus_checkpoint_seconds'] for r in rr)/16))
    primary=indexed[model.PRIMARY,'budget_exhausted'];zero=indexed['alm_zero128','budget_exhausted'];cold=indexed['cold_zero128','budget_exhausted']
    intro=(f'固定主历史乘子初始化在失败前沿的1023个已证不可行样本中排除 **{primary["certified"]}** 个，完整生成＋筛选均时 **{primary["mean_total_seconds"]:.6f}秒/前沿**。'
        f'同一ALM原始状态但零乘子排除{zero["certified"]}个；冷启动128步排除{cold["certified"]}个。'
        '因此，相对冷128的改善不能证明历史乘子的独立价值。')
    out=base/'report_v1';out.mkdir(parents=True,exist_ok=False);save(out/'table_data.json',data);save(out/'figure_data.json',dict(groups=groups,cold_curve=curve))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12.5,6.1));yy=np.arange(len(model.METHODS))
    axes[0].barh(yy,[indexed[n,'budget_exhausted']['certified'] for n in model.METHODS],
        color=['#157f78' if n==model.PRIMARY else '#6884a6' if n=='alm_zero128' else '#abb7c4' for n in model.METHODS])
    axes[0].set_yticks(yy,[n.replace('_',' ') for n in model.METHODS]);axes[0].invert_yaxis();axes[0].set_xlim(0,1130)
    for i,n in enumerate(model.METHODS):
        val=indexed[n,'budget_exhausted']['certified'];axes[0].text(val+10,i,str(val),va='center',fontsize=9)
    axes[0].set_title('Certified exclusions on failed frontiers',loc='left',pad=12);axes[0].set_xlabel('Out of 1,023 exact infeasible sampled candidates')
    axes[1].plot([r['mean_component_checkpoint_seconds'] for r in curve],[r['certified'] for r in curve],'-o',label='Cold PDHG checkpoints',color='#566f91')
    for r in curve:
        if r['step'] in [128,256,512,1024]:axes[1].annotate(str(r['step']),xy=(r['mean_component_checkpoint_seconds'],r['certified']),xytext=(4,-15 if r['step']==1024 else 7),textcoords='offset points',fontsize=9)
    for name,label,color,marker in [('alm_dual128','ALM state + history','#157f78','D'),('alm_zero128','Same ALM state + zero','#d8903b','s'),('pc_zero128','PC state + zero','#95537f','^')]:
        g=indexed[name,'budget_exhausted'];axes[1].scatter(g['mean_total_seconds'],g['certified'],label=label,color=color,marker=marker,s=58,zorder=3)
    axes[1].set_ylim(-35,1120);axes[1].set_xlim(0,.105);axes[1].set_title('Historical generation is not free',loc='left',pad=12)
    axes[1].set_xlabel('Mean generation + screening seconds per frontier');axes[1].set_ylabel('Certified exclusions')
    axes[1].legend(loc='lower right',frameon=False,fontsize=8.5)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.14)
    fig.subplots_adjust(left=.17,right=.98,top=.86,bottom=.20,wspace=.33)
    fig.text(.17,.08,'Frozen 48 old frontiers, 432 calls; figure shows 16 budget-exhausted frontiers.',fontsize=9)
    fig.text(.17,.03,'No query targets. Cold curve is one 1,024-step trajectory, not separately timed shorter runs.',fontsize=9)
    fig.savefig(out/'history_certificate.png',dpi=165);plt.close(fig)
    curve_table=[[r['step'],r['certified'],f"{r['mean_component_checkpoint_seconds']:.6f}"] for r in curve]
    text=['# 375｜历史乘子热启动：同状态对照与完整生成费用','',intro,'',
        '## 1. 本轮真正改变的变量','',
        '固定372已经封存的48前沿、2117样本；每个方法只看其前缀支持，不看剩余观察或查询答案。候选从原17起点运行64步ALM，按当前支持最大误差、移动距离和索引选出一个原始状态。所有ALM方法逐位共享同一b,h的数值，但各自实际重新生成并收费；只改变归一化初始信用。PC两个控制同理。','',
        '活动约束信用a来自历史乘子、零、当前残差、独立随机符号或明确全局BP负伴随。局部映射p=s*a消去初始z系数，不需要完整链式BP。随后均运行同一个既有PDHG；新意假设只在历史方向是否节省证书发现费用，不把PDHG本身称新方法。','',
        '## 2. 全九方法与两个来源终态','',
        '失败前沿16个，1024样本中1023不可行；完成前沿32个，1093样本中1067不可行。后两列是失败前沿上固定主独有/该控制独有的证书数量，不是独立任务数。','',
        table(['方法','失败前沿排除','均全组件秒','完成前沿排除','均全组件秒','主独有','控制独有'],data),'',
        '![同状态证书覆盖与计费](history_certificate.png)','',
        '全部配置在0步都没有直接得到正证书。主在失败前沿675，高于同状态残差480、随机499、BP信用458，但低于同状态零信用763；普通PC原始状态零信用741也更便宜。成功前沿主263高于同ALM零220，但普通PC零266仍可达到，不能单挑该子集宣布独立优势。','',
        '## 3. 强冷启动的固定工作曲线','',table(['冷PDHG步数','失败前沿排除','生成＋该检查点均秒'],curve_table),'',
        '冷1024保留预列128/256/512/1024工作前缀；256步已排除967个、512步1010个、1024步1019个。检查点时间是同一长轨迹的累计费用，包含初始分配、守卫进入和信用生成；不冒充独立短运行的完整最终退出/归档时间。128前缀在48个前沿逐位复现独立冷128的证明mask。母状态生成时间不可免费删去或假设任意摊薄。','',
        '这些结果未通过374预设机制门槛，因此不直接进入新任务确认或声称完整预测改善。更多局部成功不能替代相对零信用和强冷启动的独立比较。','',
        '## 4. 数学解释与下一可检验问题','',
        '固定区域的局部分离函数D(p,a)=inf_y[pᵀ(z-h_prev-b)+aᵀ(h-sz-c)]是凹、正齐次且超可加的。D>0证明不可能有一致原始状态，但这些性质不保证用原非线性求解轨迹生成的历史乘子适合另一个待检区域。相同原始状态第一步的增量相同，只有D(history+increment)>0而D(increment)≤0时才能证明该步提前；本轮没有出现0步正证书。','',
        '特别地，不能把不可行问题当成有可行鞍点的优化问题，套用到最优鞍点距离的热启动收敛界。历史乘子来自原非线性轨迹，待检证书却针对不同的固定分支约束；二者方向一致尚未证明。以上是结果支持的归因边界，不是证明所有历史信用无用。','',
        '下一候选若继续，应直接针对当前固定区域的层间约束累积原始/乘子状态，检验增广惩罚与局部精确块更新是否比既有PDHG更经济地产生分离方向；普通无乘子、PC、同参数强优化和冷PDHG仍需保留。不能仅继续调当前历史信用尺度或重挑初始重启。该后续算法本轮尚未实现，最终仍须接完整在线预测与新任务确认。','',
        '## 5. 审计与范围','',
        f'432正式调用全部完成、0执行异常；预检12个旧证明数组逐位复现、213落盘数组回读、27次已知可行保留通过。独立审计{audit["checks"]["scalar_credit_entries"]}标量信用（最大差0）、{audit["checks"]["exact_wide_domain_proofs"]}实际宽域Fraction正证明、{audit["checks"]["same_primal_state_arrays"]}同原始状态数组、{audit["checks"]["checkpoints"]}检查点及{audit["checks"]["positive_retained"]}次正体积样本保留通过。','',
        '初始信用来自本前缀支持而非LP标签；全部432输出封存后才读取旧几何分类作审计。已暴露任务和跨排序重复样本，不宣称新分布显著性。组件时间包括生成与筛选，不含原建树、最终几何和查询读出；命名数组统计不等于进程峰值。没有新增query MSE，也没有完成研究目标。','',
        '[执行前协议](../../../outputs/ttt-pc-alm-research/374_history_certificate_protocol_v1.md) · [全部调用](../development_v1/rows.json) · [独立审计](../audit_v1/summary.json) · [配对覆盖](../audit_v1/paired.json) · [全部检查点](../audit_v1/checkpoints.json)','']
    write(out/'report.md','\n'.join(text));entry='outputs/ttt-pc-alm-research/375_history_certificate_results_v1.md'
    write(root/entry,'# 375｜历史乘子热启动结果\n\n'+intro+'\n\n[完整图文](../../results/history_certificate/report_v1/report.md)\n')
    save(out/'manifest.json',dict(source_sha256=sha(Path(__file__)),entry_file=entry,entry_sha256=sha(root/entry),
        input_summary_sha256={p:sha(base/p) for p in ['development_v1/summary.json','audit_v1/summary.json']},
        outputs_sha256={f:sha(out/f) for f in ['table_data.json','figure_data.json','report.md','history_certificate.png']}))
    content=(out/'report.md').read_text(encoding='utf-8');cells=0
    for r in data+curve_table:assert '| '+' | '.join(map(str,r))+' |' in content;cells+=len(r)
    assert (primary['certified'],zero['certified'],cold['certified'])==(675,763,580)
    assert [(r['step'],r['certified']) for r in curve if r['step']>=256]==[(256,967),(512,1010),(1024,1019)]
    save(out/'qa_numeric.json',dict(passed=True,table_cells=cells,methods=9,curve_points=7,manifest_sha256=sha(out/'manifest.json')))
    print(dict(passed=True,report=str(out/'report.md')),flush=True)


if __name__=='__main__':main()
