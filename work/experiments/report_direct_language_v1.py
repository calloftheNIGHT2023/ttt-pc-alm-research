"""366 completed direct-baseline evidence; display is separate from frozen fitting."""
from pathlib import Path
import math
import numpy as np
from report_search_radius_development_v1 import read,sha,save,write,complete,table

BASE='results/direct_language';PRIMARY='frontier_dual_frontier_g8'
ENTRY='outputs/ttt-pc-alm-research/366_direct_language_results_v1.md'


def main():
    root=Path(__file__).resolve().parents[2];base=root/BASE
    prediction=complete(base/'development_predictions_v1');evaluation=complete(base/'development_evaluation_v1')
    protocol=read(base/'development_evaluation_v1/protocol.json')
    assert protocol['prediction_summary_sha256']==sha(base/'development_predictions_v1/summary.json')
    for n,h in protocol['source_sha256'].items():assert sha(root/n)==h
    methods=read(base/'development_evaluation_v1/methods.json');by={m['method']:m for m in methods}
    comparisons={(c['candidate'],c['control'],c['metric']):c for c in read(base/'development_evaluation_v1/comparisons.json')}
    mechanisms=read(base/'development_evaluation_v1/mechanisms.json');rows=[]
    for m in methods:
        c=None if m['method']==PRIMARY else comparisons[PRIMARY,m['method'],'mse257']
        rows.append([m['method'],f"{m['metrics']['mse257']:.10f}",f"{m['mean_current_seconds']:.6f}",m['failures'],
            '—' if c is None else f"{c['mean_difference']:+.10f}",
            '—' if c is None else f"{c['improved']}/{c['equal']}/{c['worse']}"])
    out=base/'report_v1';out.mkdir(parents=True,exist_ok=False)
    plot=[dict(method=m['method'],mse=m['metrics']['mse257'],seconds=m['mean_current_seconds']) for m in methods]
    save(out/'table_data.json',rows);save(out/'figure_data.json',plot)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6));yy=np.arange(4)
    labels=['Direct language; no mother','ALM credit frontier (primary)','ALM mother + all C20','Adam240 + shared readout']
    colors=['#bc7d32','#157f78','#7896ad','#9dabc3']
    for ax,field,title in [(axes[0],'mse','Unseen-query MSE'),(axes[1],'seconds','Complete fit seconds')]:
        values=[p[field] for p in plot];ax.barh(yy,values,color=colors,height=.63)
        ax.set_yticks(yy,labels if ax is axes[0] else []);ax.invert_yaxis()
        ax.set_xlim(0,max(values)*1.25);ax.set_title(title,loc='left',pad=12)
        for y,v in enumerate(values):ax.text(v+max(values)*.018,y,f'{v:.6f}',va='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    fig.subplots_adjust(left=.27,right=.985,top=.85,bottom=.23,wspace=.13)
    fig.text(.27,.11,'32 exposed tasks; same observations, prior, geometry and 2048-particle readout.',fontsize=9)
    fig.text(.27,.05,'One interleaved full-fit timing per task; global geometry explicitly charged.',fontsize=9)
    fig.savefig(out/'direct_risk_cost.png',dpi=170);plt.close(fig)
    direct=by['direct_language_all'];mother=by['frontier_c20_all'];main=by[PRIMARY]
    total_enum=sum(m['enumerated'] for m in mechanisms);total_remaining=sum(m['remaining'] for m in mechanisms)
    direct_only=sum(len(m['direct_only']) for m in mechanisms);mother_only=sum(len(m['mother_only']) for m in mechanisms)
    exact=sum(m['bitwise_same_prediction'] for m in mechanisms);maxgap=max(m['max_prediction_gap'] for m in mechanisms)
    cc=comparisons['direct_language_all','frontier_c20_all','mse257']
    msg=(f"去掉全部母轨迹的直接必要语言基线：MSE **{direct['metrics']['mse257']:.10f}**，完整平均 **{direct['mean_current_seconds']:.6f} 秒/任务**。"
         f"母轨迹加全池：MSE {mother['metrics']['mse257']:.10f}、{mother['mean_current_seconds']:.6f} 秒；"
         f"研究主配置：MSE {main['metrics']['mse257']:.10f}、{main['mean_current_seconds']:.6f} 秒。")
    text=['# 366｜把母轨迹也移除：完整候选推断的强对照','',msg,'',
        f"本轮共128次正式完整fit，另4次预检，执行失败0。32任务此前已经暴露；预测全部封存后才评分。本页不是一次新的512任务确认，也不把这个直接推断基线改名为PC-ALM。",'',
        '## 1. 为什么这项控制必须补','',
        '之前的残差/零信用对照共享ALM生成的629路母轨迹，只能说明新增信用不必要。现在直接从支持观察出发：单观测可行路径的笛卡尔积→C5/C20区间收缩→逐区域联合几何→同一体积加权粒子读出。Local、branch_probe、母fit及BP入口在直接法中全部禁止；联合几何LP明确使用并计入时间。','',
        '```text\n相同4个支持观测 + 相同先验/噪声范围\n  ├─ ALM母轨迹 → 有限信用筛选 ─┐\n  ├─ ALM母轨迹 → 剩余全池 ────┤\n  ├─ Adam母轨迹 ──────────────┤→ 联合几何 → 同一2048粒子读出 → 未见查询\n  └─ 必要语言全池（无母轨迹）──┘\n```','',
        '## 2. 数学上检验的是哪个瓶颈','',
        r'记第i个观察允许的单观测路径集合为$L_i$。每个联合支持可行参数的模式都在$\prod_iL_i$内，因为联合可行必然逐观察可行。这个必要包含不需要乘子或反向梯度。直接法的候选枚举量为$\prod_i|L_i|$，而不是把模式缓存或教师参数当免费输入。','',
        '这是关于候选覆盖的包含关系，不是精确Bayes计算的实现保证。保守收缩、浮点几何、体积、零体积边界与粒子读出仍各有条件；报告保留几何分类与精确证书。模式一致也不自动保证粒子逐位一致。','',
        '## 3. 全部同期完整成本和查询结果','',
        table(['方法','MSE257','完整均秒','失败','主−该方法','主改善/同/差'],rows),'',
        '![同期查询误差和完整成本](direct_risk_cost.png)','',
        f"直接法相对母轨迹全池：均差{cc['mean_difference']:+.10f}，改善/同/差{cc['improved']}/{cc['equal']}/{cc['worse']}。所有48项有向配对比较和四指标均保留，不按查询结果改预算或挑任务。",'',
        f"直接法共枚举{total_enum:,}个单观测语言积候选，收缩后{total_remaining:,}个进入几何分类。直接法独有正体积模式{direct_only}个任务—模式对；母全池独有{mother_only}个。与母全池预测逐位相同{exact}/32，最大逐查询绝对差{maxgap:.10g}。",'',
        '这里比较的是相同观察信息和明确测量的完整成本，不是严格相同FLOPs或峰值内存。直接法返回数组/几何数组字节和建池/几何时间见逐任务机制表；这些只是命名数组subtotal，未测进程峰值，不能作全面省内存结论。','',
        '## 4. 这怎样影响论文主张','',
        '如果移除母轨迹仍得到相同或更低误差，就不能再以这批四参数、四观察任务证明ALM母轨迹必要；也不能把更好的直接法作为PC-ALM新方法。已有严格乘子逃逸见证仍成立，但它回答的是特定优化状态如何逃逸，不是任意推断算法必须使用乘子。','',
        '若要继续证明乘子有独立任务价值，靶点应是实际受预算限制的发现过程：在必要语言完整展开不可承受时，历史约束信用能否比残差、随机、零信用和同参数强优化器更可靠地找到影响预测的区域。必须先限定任务族、候选/状态预算和查询风险条件，再做新冻结测试；不能从指数枚举量直接推出ALM会胜。','',
        '这项研究目标尚未完成。本实验不宣称任意闭式解/任意浅层网络都不如PC-ALM，也没有新增官方TTT-MLP、LLM/VLM或真实下游结果。','',
        '## 5. 核验与复核入口','',
        f"独立标量前向核验{evaluation['checks']['scalar_risks']}项风险，最大差{evaluation['maximum_scalar_risk_gap']:.3g}；"
        f"核验{evaluation['checks']['geometry_certificates']:,}个几何证书、{evaluation['checks']['support_particles']:,}个支持粒子，"
        f"32次直接读出独立重算最大差{evaluation['maximum_direct_readout_gap']:.3g}。原控制{prediction['counts']['control_arrays']}个数组逐位重放，首任务{prediction['counts']['preflight_arrays']}个数组预检复现。",'',
        '[执行前协议](../../../outputs/ttt-pc-alm-research/365_direct_language_protocol_v1.md) · [全部方法](../development_evaluation_v1/methods.json) · [配对结果](../development_evaluation_v1/comparisons.json) · [逐任务机制](../development_evaluation_v1/mechanisms.json) · [审计](../development_evaluation_v1/summary.json)','']
    write(out/'report.md','\n'.join(text));write(root/ENTRY,'# 366｜直接必要语言强对照\n\n'+msg+'\n\n[完整图文](../../results/direct_language/report_v1/report.md)\n')
    save(out/'manifest.json',dict(entry_file=ENTRY,entry_sha256=sha(root/ENTRY),
        evaluation_summary_sha256=sha(base/'development_evaluation_v1/summary.json'),source_sha256=sha(Path(__file__)),
        outputs_sha256={n:sha(out/n) for n in ['report.md','table_data.json','figure_data.json','direct_risk_cost.png']}))
    # Check rendered numeric cells and graph source against the sealed risk matrix/rows.
    with np.load(base/'development_evaluation_v1/task_metrics.npz',allow_pickle=False) as z:
        names=list(z['methods']);risks=z['risk'].copy()
    raw=read(base/'development_predictions_v1/rows.json');html=(out/'report.md').read_text(encoding='utf-8')
    cells=0
    for row,item in zip(rows,plot):
        j=names.index(row[0]);mse=math.fsum(map(float,risks[:,j,0]))/32
        seconds=math.fsum(r['seconds'] for r in raw if r['method']==row[0])/32
        assert row[1]==f'{mse:.10f}' and row[2]==f'{seconds:.6f}'
        assert '| '+' | '.join(map(str,row))+' |' in html
        assert item['mse']==mse and item['seconds']==seconds
        if row[0]!=PRIMARY:
            delta=risks[:,names.index(PRIMARY),0]-risks[:,j,0]
            assert row[4]==f'{math.fsum(map(float,delta))/32:+.10f}'
            assert row[5]==f'{int((delta<0).sum())}/{int((delta==0).sum())}/{int((delta>0).sum())}'
        cells+=6
    save(out/'qa_numeric.json',dict(passed=True,table_cells=cells,figure_values=8,manifest_sha256=sha(out/'manifest.json')))
    print(dict(passed=True,report=str(out/'report.md'),message=msg,positive_direct_only=direct_only,positive_mother_only=mother_only,
               bitwise_same=exact,enumerated=total_enum,remaining=total_remaining),flush=True)


if __name__=='__main__':main()
