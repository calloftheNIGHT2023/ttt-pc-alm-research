"""373 completed prefix obstruction diagnosis; no query-performance claims."""
from pathlib import Path
import math
import numpy as np
from budget_reinvestment_suite_v1 import read,sha,save,complete
from report_search_radius_development_v1 import write,table


def main():
    root=Path(__file__).resolve().parents[2];base=root/'results/prefix_obstruction'
    summary=complete(base/'development_v3');audit=complete(base/'audit_v3');groups=read(base/'audit_v3/groups.json')
    assert audit['development_summary_sha256']==sha(base/'development_v3/summary.json')
    assert len(groups)==6 and sum(g['samples'] for g in groups)==summary['samples']
    for path,h in read(base/'development_v3/protocol.json')['source_sha256'].items():assert sha(root/path)==h
    failed=[g for g in groups if g['source_status']=='budget_exhausted'];done=[g for g in groups if g['source_status']=='complete']
    def total(gg,k):return sum(g[k] for g in gg)
    bad,total_n=total(failed,'infeasible'),total(failed,'samples');cp,pp=total(failed,'c100'),total(failed,'pdhg')
    intro=(f'16个预算耗尽前沿的{total_n}个均匀抽样分支中，**{bad}个精确不可行、{total(failed,"positive")}个正体积可行**。'
        f'冷启动局部PDHG128排除{pp}个，额外C100仅排除{cp}个。'
        '这定位了一个具体计算瓶颈：区间放松留下大量并不存在共同参数的前缀。它不构成PC-ALM独立收益。')
    out=base/'report_v1';out.mkdir(parents=True,exist_ok=False)
    data=[]
    for g in groups:
        data.append([g['method'],g['source_status'],g['cases'],g['samples'],g['infeasible'],g['positive'],g['zero'],g['unresolved'],g['c100'],g['pdhg'],
            f"{g['c100_seconds']/g['cases']:.6f}",f"{g['pdhg_seconds']/g['cases']:.6f}"])
    save(out/'table_data.json',data)
    plot=dict(failed=[dict(method=g['method'],infeasible=g['infeasible'],positive=g['positive'],samples=g['samples'],c100=g['c100'],pdhg=g['pdhg']) for g in failed])
    save(out/'figure_data.json',plot)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8));yy=np.arange(3)
    labels=[g['method'].replace('_',' ') for g in failed]
    badrate=[100*g['infeasible']/g['samples'] for g in failed]
    goodrate=[100*g['positive']/g['samples'] for g in failed]
    axes[0].barh(yy,badrate,color='#a95c46',label='Exact infeasible')
    axes[0].barh(yy,goodrate,left=badrate,color='#258478',label='Exact positive volume')
    axes[0].set_yticks(yy,labels);axes[0].set_xlim(0,104);axes[0].set_xticks([0,25,50,75,100]);axes[0].set_xlabel('Percent of sampled prefix candidates')
    for i,g in enumerate(failed):axes[0].text(3,i,f"{g['infeasible']}/{g['samples']} infeasible",va='center',color='white',fontsize=10)
    axes[0].set_title('Why the join budget was exhausted',loc='left',pad=13)
    for offset,key,label,color in [(-.18,'c100','Additional interval C100','#a4afbf'),(.18,'pdhg','Cold local PDHG128','#32698d')]:
        values=[100*g[key]/g['infeasible'] for g in failed];axes[1].barh(yy+offset,values,height=.33,label=label,color=color)
        for i,(val,g) in enumerate(zip(values,failed)):axes[1].text(val+1,i+offset,str(g[key]),va='center',fontsize=9)
    axes[1].set_yticks(yy,labels);axes[1].set_xlim(0,100);axes[1].set_xlabel('Percent of exact infeasible sample removed')
    axes[1].set_title('Simple local screens still leave a gap',loc='left',pad=13)
    for ax in axes:ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=1,frameon=False,fontsize=9)
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=1,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.12,right=.985,top=.84,bottom=.34,wspace=.37)
    fig.text(.12,.035,'16 failed frontiers; 1,024 sampled candidates. Correlated old tasks, not unseen-query evaluation.',fontsize=9)
    fig.savefig(out/'prefix_obstruction.png',dpi=170);plt.close(fig)
    body=['# 373｜预算拥塞来自哪里：固定抽样与精确证书诊断','',intro,'',
        '## 1. 数学问题已经具体化','',
        '每个固定前缀对应共享参数约束A b≤r、b∈[-B,B]^4。区间传播只保留坐标范围，可能丢掉层间与观察间相关性，因而每个局部区间都不空，仍可能不存在共同b。对任意非负权重w，若L=-wᵀr-B||Aᵀw||₁>0，则所有盒内b都不可能满足这些约束。这是既有线性分离证书，不宣称新定理。','',
        '主要诊断采用原筛选器的EPS+TOL外舍入宽域。不可行不是由偷偷收窄到.001支持带产生。LP仅提议点/权重，Fraction精确计算决定结论。可行前缀也不保证可以扩展到剩余观察；本轮没有体积或后验读出。','',
        '## 2. 全部抽样结果','',
        table(['排序','原调用终态','前沿数','抽样数','不可行','正体积','零体积或空','未判定','C100排除','PDHG128排除','C100秒/前沿','PDHG秒/前沿'],data),'',
        '![无效前缀与简单控制覆盖](prefix_obstruction.png)','',
        f'全部48前沿共{summary["samples"]}样本：2090不可行、27正体积，0未判定。预算失败子集的逐前沿等权不可行比例为{audit["mean_budget_frontier_infeasible_fraction"]:.8%}，通过执行前≥75%的诊断分流阈值。每个失败前沿本轮均抽64个，因此这里与汇总样本比例相同。完成子集有些池不足64，不能混用按前沿与按样本平均。','',
        f'失败子集PDHG拒绝{pp}/{bad}（{100*pp/bad:.2f}%），仍有{bad-pp}个精确不可行抽样未被其128步证明。C100拒绝{cp}/{bad}（{100*cp/bad:.2f}%）。两种控制工作不同，表中仅为本机一次附加组件计时，不是公平端到端速度结论。','',
        '## 3. 这改变下一步，而不是完成目标','',
        '长支持的主要障碍已经从“可能存在太多真实解释”收窄到“便宜收缩未处理跨观察/层间矛盾”。因此下一候选应直接检验历史乘子能否降低这些不可行证书的发现成本，而不是再换一种支持排序。冷PDHG是必需的强简单控制，不能把它的580次删除改名PC-ALM贡献。','',
        '候选需要在同一原始状态下比较历史乘子、零乘子、残差、随机信用及明确标记的BP信用，并将生成历史状态的成本完整计入。仅证书覆盖更高还不够：只有完整求交/发现能在预算内留下有预测价值的区域，再在新冻结任务中降低查询误差，才达到原研究目标。本轮没有实现该后续候选，也没有查询效果新结论。','',
        '## 4. 公平性与审计','',
        '48输入来自旧16任务、24支持的原顺序/空间分散/乘子排序；选最后完整前缀，所有抽样先于任何筛选/LP封存。两局部控制全部封存后才执行LP。没有查询答案、没有用LP权重热启动任何候选。跨排序和同任务样本相关，不做独立任务显著性检验。正式输出无算法异常。','',
        f'独立审计：{audit["checks"]["independent_constraint_rows"]}条约束的另一种组装核验，{audit["checks"]["exact_certificates"]}个有理证书重算，{audit["checks"]["pdhg_wide_domain_exact"]}个PDHG宽域正下界核验，27个独立可行点，以及全部48抽样重放。','',
        '预检v1错误要求区间收缩必须检出矛盾支持，v2改为记录并保留原失败；v2首次正式筛选保存时旧JSON编码器不支持NumPy证明数组，v3仅修落盘并增加回读测试。算法文件一直未改，v1选择240数组、v2已保存2个mask逐位复现。两次失败和原源码全保留；不能把预检/落盘修复掩盖成一次完美运行。','',
        '[执行前协议](../../../outputs/ttt-pc-alm-research/372_prefix_obstruction_protocol_v1.md) · [断言修复](../../../outputs/ttt-pc-alm-research/372_preflight_assertion_repair_v2.md) · [序列化修复](../../../outputs/ttt-pc-alm-research/372_serialization_repair_v3.md) · [正式结果](../development_v3/rows.json) · [独立审计](../audit_v3/summary.json)','']
    write(out/'report.md','\n'.join(body));entry='outputs/ttt-pc-alm-research/373_prefix_obstruction_results_v1.md'
    write(root/entry,'# 373｜精确定位前缀拥塞\n\n'+intro+'\n\n[完整图文](../../results/prefix_obstruction/report_v1/report.md)\n')
    save(out/'manifest.json',dict(source_sha256=sha(Path(__file__)),entry_file=entry,entry_sha256=sha(root/entry),
        input_summary_sha256={'development_v3/summary.json':sha(base/'development_v3/summary.json'),'audit_v3/summary.json':sha(base/'audit_v3/summary.json')},
        outputs_sha256={n:sha(out/n) for n in ['table_data.json','figure_data.json','report.md','prefix_obstruction.png']}))
    text=(out/'report.md').read_text(encoding='utf-8');cells=0
    for row,g in zip(data,groups):
        assert row[2:10]==[g[k] for k in ['cases','samples','infeasible','positive','zero','unresolved','c100','pdhg']]
        assert '| '+' | '.join(map(str,row))+' |' in text;cells+=len(row)
    assert (bad,total_n,cp,pp)==(1023,1024,45,580)
    save(out/'qa_numeric.json',dict(passed=True,table_cells=cells,figure_groups=3,headline_values=[bad,total_n,cp,pp],manifest_sha256=sha(out/'manifest.json')))
    print(dict(passed=True,report=str(out/'report.md'),budget_infeasible=bad,budget_samples=total_n),flush=True)


if __name__=='__main__':main()
