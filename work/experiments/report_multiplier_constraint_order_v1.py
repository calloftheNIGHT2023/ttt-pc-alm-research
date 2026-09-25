"""371 actual constraint-order result, resource bottleneck and exact theorem scope."""
from pathlib import Path
import math
from collections import Counter
import numpy as np
from report_search_radius_development_v1 import read,sha,save,write,complete,table

BASE='results/multiplier_constraint_order';ENTRY='outputs/ttt-pc-alm-research/371_constraint_order_results_v1.md'


def main():
    root=Path(__file__).resolve().parents[2];base=root/BASE
    s=complete(base/'development_v1');audit=complete(base/'audit_v1');symbolic=complete(root/'results/parameter_arrangement_bound/audit_v1')
    ps=complete(root/'results/prefix_language_join_v2/development_v1')
    assert audit['development_summary_sha256']==sha(base/'development_v1/summary.json')
    rows=read(base/'development_v1/rows.json');p=read(base/'development_v1/protocol.json');names=p['methods']
    for file,h in p['source_sha256'].items():assert sha(root/file)==h
    prefix=read(root/'results/prefix_language_join_v2/development_v1/rows.json')
    index={(r['seed'],r['n'],r['method']):r for r in rows}
    current={};raw=[]
    for n in [16,24]:
        current[n]=[]
        for name in names:
            rr=[r for r in rows if r['n']==n and r['method']==name]
            mean=math.fsum(r['seconds'] for r in rr)/16;ordering=math.fsum(r['ordering_seconds'] for r in rr)/16
            expanded=sum(r['expanded'] for r in rr);count=sum(r['completed'] for r in rr)
            gained=sum(r['completed'] and not index[r['seed'],n,'observed']['completed'] for r in rr)
            lost=sum(not r['completed'] and index[r['seed'],n,'observed']['completed'] for r in rr)
            current[n].append([name,f'{count}/16',f'{mean:.6f}',f'{ordering:.6f}',expanded,sum(r['pair_candidate_rows'] for r in rr),f'{gained}/{lost}'])
            raw.append(dict(n=n,method=name,completed=count,mean_seconds=mean,mean_ordering_seconds=ordering,expanded=expanded))
    scaling=[]
    for n in [4,8,16,24]:
        for order in ['observed','small_first']:
            rr=[r for r in prefix if r['group']=='context_scaling' and r['n']==n and r['ordering']==order]
            scaling.append([n,order,f"{sum(r['completed'] for r in rr)}/16",f"{math.fsum(r['seconds'] for r in rr)/16:.6f}",
                sum(r['expanded'] for r in rr),str(max(r['full_product'] for r in rr))])
    totals={name:sum(r['completed'] for r in rows if r['method']==name) for name in names}
    times={name:math.fsum(r['seconds'] for r in rows if r['method']==name)/32 for name in names}
    otherbest=max(totals[name] for name in names if name!='alm_dual')
    description=(f"固定主排序alm_dual完成 **{totals['alm_dual']}/32** 个任务—支持块，完整组件均时 **{times['alm_dual']:.6f}秒**；"
        f"原顺序完成{totals['observed']}/32，成对兼容完成{totals['pair_compatibility']}/32，空间分散完成{totals['farthest_x']}/32。"
        f"全部预列非主控制中最高完成数为{otherbest}/32（描述性汇总，不替换固定主）。")
    out=base/'report_v1';out.mkdir(parents=True,exist_ok=False)
    save(out/'table_data.json',dict(scaling=scaling,current=current));save(out/'figure_data.json',raw)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,6.6));yy=np.arange(len(names))
    for j,(n,color) in enumerate([(16,'#32698d'),(24,'#ba8a3c')]):
        values=[next(r['completed'] for r in raw if r['n']==n and r['method']==name) for name in names]
        axes[0].barh(yy+(j-.5)*.34,values,height=.32,label=f'{n} supports',color=color)
    axes[0].set_yticks(yy,[n.replace('_',' ') for n in names]);axes[0].set_xlim(0,17);axes[0].set_xticks([0,4,8,12,16]);axes[0].legend(loc='lower right')
    axes[0].set_title('Complete joins out of 16 tasks',loc='left',pad=12)
    axes[1].barh(yy,[times[n] for n in names],color=['#157f78' if n=='alm_dual' else '#9aaebe' for n in names])
    axes[1].set_yticks(yy,[]);axes[1].set_title('Mean total component seconds (all 32 cases)',loc='left',pad=12)
    for ax in axes:ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.16)
    axes[0].set_xlabel('Completion, not prediction quality');axes[1].set_xlabel('Includes ordering generation and failed-budget work')
    fig.subplots_adjust(left=.19,right=.985,top=.90,bottom=.20,wspace=.12)
    fig.text(.19,.09,'Fixed 65,536 join-expansion ceiling; generation overhead is separately charged, not equal FLOPs.',fontsize=9)
    fig.text(.19,.04,'Old development tasks. No query targets, geometry integration or new query-risk evaluation.',fontsize=9)
    fig.savefig(out/'ordering_completion_cost.png',dpi=165);plt.close(fig)
    bp=sum(r['completed'] for r in rows if r['method']=='alm_bp_score')
    residual=sum(r['completed'] for r in rows if r['method']=='alm_residual')
    same=read(base/'audit_v1/pools.json');completed_groups=[g for g in same if g['complete_methods']]
    body=['# 371｜历史乘子引导约束求交顺序：机制与完整资源测试','',description,'',
        '本轮288次正式调用，另9方法预检和重复；全部预检/正式调用的未完成终态均保留。这里的“完成”仅指预算内遍历完必要候选求交，不是论文完成、查询风险改善或精确后验积分。','',
        '## 1. 先确认真正的计算瓶颈','',
        '368以单观察必要路径构造增量联合候选。每次只扩展上一阶段保留的前缀，再执行原C5/C20。完整支持可行参数是所有前缀的共同见证，因而保守剪枝不会丢失它；实际展开量是各阶段保留前缀数乘以下一观察路径数之和，不是完整笛卡尔积。','',
        table(['支持数','顺序','完成','组件均秒','展开总数','最大完整语言积'],scaling),'',
        '四观察的另外32个365任务中，两排序均保留与原完整笛卡尔积逐位相同的模式集合；原全积每顺序均计277604展开，增量原顺序29548、小语言优先27391。到了24观察，原顺序9/16完成，小语言优先0/16，全部37个预算终止均为展开上限。不能把一个局部路径更少的顺序当成全局求交更便宜的保证。','',
        '## 2. 新假设：用历史压力决定先绑定哪些支持','',
        r'对观察i，乘子线性项对残差的敏感度上包络满足$\max_{\|\delta r_i\|_\infty\le1}|u_i^T\delta r_i|=\|u_i\|_1$。候选将17起点、64步原ALM的该值平均，按降序加入约束。它利用历史约束信息，但上包络方向未必是实际参数可达方向，所以这不是候选删除数或收益保证。','',
        '所有方法只改变支持求交顺序，最终仍由相同保守收缩决定能否删除。控制包括原顺序、单语言最小、输入空间分散、固定随机、成对兼容、同ALM末步残差、同ALM参数的BP梯度分数、普通PC残差。三个ALM评分各自现场重算母状态，没有免费复用。成对兼容的额外枚举也计费。','',
        '## 3. 所有配置与费用','',
        '### 16个支持','',table(['排序','完成','总组件均秒','生成均秒','join展开总数','额外成对候选','相对原顺序解锁/失去'],current[16]),'',
        '### 24个支持','',table(['排序','完成','总组件均秒','生成均秒','join展开总数','额外成对候选','相对原顺序解锁/失去'],current[24]),'',
        '![完整组件费用和求交完成数](ordering_completion_cost.png)','',
        '时间包括未完成调用与生成排序费用，不报成功子集速度。共同65536上限只约束join展开，不等于相同总FLOPs；成对预处理、局部迭代、状态数组都另列。总组件8秒门槛在批边界合作检查，可越过一个批，不称硬实时保证。数值库单线程，本轮未另起研究重计算，但没有连续记录全系统负载，不宣称完全独占环境。未测进程峰值；未执行最终LP/体积/粒子，不能与完整在线预测时间混比。','',
        '## 4. 独立性与数学边界','',
        f"主完成{totals['alm_dual']}/32，同ALM残差{residual}/32，BP分数{bp}/32；原顺序{totals['observed']}/32。具体是否存在独立竞争力必须看全部表格及成本，不能只对小语言优先这一较弱顺序报告提升。",'',
        f"至少一种方法完成的{len(completed_groups)}个任务—块中，{sum(g['all_complete_pools_equal'] for g in completed_groups)}/{len(completed_groups)}的各完整方法候选池相同；预算未完成的空regions数组只是终态标记，不解释为真实可行域为空。",'',
        '369附录还证明：固定d个偏置时，正体积支持模式数≤Σ_{j=0}^d binom(H,j)，H=3n(2^(d+1)−d−2)，因此固定d下是关于n的多项式上界。其系数可以很大，且未证可行的区间放松前缀不受这个界约束。20个有限符号案例与排列递推核验通过，只是通用书面证明的辅助检查，不是运行时间或PC优势定理。','',
        '本轮定位和检验的是中间前缀拥塞。它未新增查询MSE、未使用未见任务、未替代强回归/浅层头/同参数优化器的完整终端比较；不完成持续研究目标。只有新机制在强排序控制和生成成本下有优势，才应进入完整预测确认，不能用组件完成数直接发布成任务收益。','',
        '## 5. 审计、失败与复核','',
        f"368正式192调用，155完成、37预算终止；独立1920单观察语言、495339前缀行、7704有效模式前缀保留核验。370独立重算{audit['checks']['scalar_observation_signals']}个观察分数，最大差{audit['maximum_scalar_signal_gap']:.3g}；{audit['checks']['same_alm_state_arrays']}个同ALM母状态数组逐位一致，288排序与资源条目核验。",'',
        '368 v1正式第一调用之后，旧回归元数据positive_mode_keys:null造成后置检查器异常。原源与失败快照保留；v2仅把此旧元数据字段解释为空列表，算法/任务/预算未改，重新冻结前检与正式。原失败调用19个数组在v2逐位复现。370无此异常，不隐藏预算终止。','',
        '[368运行前协议](../../../outputs/ttt-pc-alm-research/368_prefix_language_scaling_protocol_v1.md) · [兼容修复](../../../outputs/ttt-pc-alm-research/368_null_metadata_repair_v2.md) · [369数学与主张台账](../../../outputs/ttt-pc-alm-research/369_parameter_arrangement_bound_appendix_v1.md) · [370冻结协议](../../../outputs/ttt-pc-alm-research/370_multiplier_constraint_order_protocol_v1.md) · [全部288调用](../development_v1/rows.json) · [独立核验](../audit_v1/summary.json)','',
        '369台账里的UNRESOLVED是该附录写作当时的实验状态；本页是完成后的结果，不回改执行前假设。附录与报告由AI协作推导、编码、核验；未作会场提交或新颖性宣称。','']
    write(out/'report.md','\n'.join(body));write(root/ENTRY,'# 371｜约束求交排序结果\n\n'+description+'\n\n[完整图文](../../results/multiplier_constraint_order/report_v1/report.md)\n')
    save(out/'manifest.json',dict(source_sha256=sha(Path(__file__)),entry_file=ENTRY,entry_sha256=sha(root/ENTRY),
        input_summary_sha256={BASE+'/audit_v1/summary.json':sha(base/'audit_v1/summary.json'),
            'results/parameter_arrangement_bound/audit_v1/summary.json':sha(root/'results/parameter_arrangement_bound/audit_v1/summary.json')},
        outputs_sha256={n:sha(out/n) for n in ['report.md','table_data.json','figure_data.json','ordering_completion_cost.png']}))
    text=(out/'report.md').read_text(encoding='utf-8');cells=0
    for n,rr in current.items():
        for row in rr:
            source=[r for r in rows if r['n']==n and r['method']==row[0]]
            assert row[1]==str(sum(r['completed'] for r in source))+'/16'
            assert row[2]==f"{sum(r['seconds'] for r in source)/16:.6f}" and row[4]==sum(r['expanded'] for r in source)
            assert '| '+' | '.join(map(str,row))+' |' in text;cells+=7
    prefix_cells=0
    for row in scaling:
        source=[r for r in prefix if r['group']=='context_scaling' and r['n']==row[0] and r['ordering']==row[1]]
        assert row[2]==str(sum(r['completed'] for r in source))+'/16'
        assert row[3]==f"{sum(r['seconds'] for r in source)/16:.6f}" and row[4]==sum(r['expanded'] for r in source)
        assert row[5]==str(max(r['full_product'] for r in source))
        assert '| '+' | '.join(map(str,row))+' |' in text;prefix_cells+=6
    for item in raw:
        source=[r for r in rows if r['n']==item['n'] and r['method']==item['method']]
        assert item['completed']==sum(r['completed'] for r in source)
        assert abs(item['mean_seconds']-sum(r['seconds'] for r in source)/16)<1e-12
    save(out/'qa_numeric.json',dict(passed=True,current_table_cells=cells,prefix_table_cells=prefix_cells,
        figure_source_values=len(raw)*2,manifest_sha256=sha(out/'manifest.json')))
    print(dict(passed=True,report=str(out/'report.md'),completed=totals,mean_seconds=times),flush=True)


if __name__=='__main__':main()
