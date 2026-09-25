"""359 reporting only after complete 358 scoring and independent audit."""
from pathlib import Path
from report_search_radius_development_v1 import read,sha,save,write,complete,table

BASE='results/cross_region_online';PRIMARY='cross_dual_reuse_g8'
ENTRY='outputs/ttt-pc-alm-research/359_cross_region_online_results_v1.md'
CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


def main():
    root=Path(__file__).resolve().parents[2];base=root/BASE
    ps=complete(base/'development_predictions_v1');es=complete(base/'development_evaluation_v1');audit=complete(base/'development_audit_v1')
    assert audit['prediction_summary_sha256']==sha(base/'development_predictions_v1/summary.json')
    assert audit['evaluation_summary_sha256']==sha(base/'development_evaluation_v1/summary.json')
    protocol=read(base/'development_predictions_v1/protocol.json');assert protocol['primary']==PRIMARY
    for p,h in protocol['source_sha256'].items():assert sha(root/p)==h
    methods=read(base/'development_evaluation_v1/methods.json');by={m['method']:m for m in methods}
    comparisons={(c['candidate'],c['control'],c['metric']):c for c in read(base/'development_evaluation_v1/comparisons.json')}
    component=complete(root/'results/cross_region_credit/audit_v1')
    attribution=complete(base/'attribution_v1')
    mechanisms=read(base/'development_audit_v1/mechanisms.json')
    out=base/'report_v1';out.mkdir(parents=True,exist_ok=False)
    rows=[];figure_methods=[]
    for cfg in protocol['configs']:
        name=cfg['name'];m=by[name];c=None if name==PRIMARY else comparisons[PRIMARY,name,'mse257']
        mm=[r for r in mechanisms if r['method']==name]
        row=[name,f"{m['metrics']['mse257']:.10f}",f"{m['mean_current_seconds']:.6f}",str(m['failures']),
             str(sum(r['geometry_calls'] for r in mm)) if mm else '—',
             '—' if c is None else f"{c['mean_difference']:+.10f}",
             '—' if c is None else f"{c['improved']}/{c['equal']}/{c['worse']}"]
        rows.append(row);figure_methods.append(dict(method=name,mse=m['metrics']['mse257'],seconds=m['mean_current_seconds']))
    references=['unvisited_dual','unvisited_zero','strong_native_alm128','strong_native_alm256',
        'strong_native_pc64','strong_native_pc256','strong_native_nodual256','strong_native_adam3840',
        'cold__prior16384_ridge','cold__meta_ridge128','cold__meta_shallow64_20']
    refrows=[]
    for name in references:
        c=comparisons[PRIMARY,name,'mse257']
        refrows.append([name,f"{by[name]['metrics']['mse257']:.10f}",f"{c['mean_difference']:+.10f}",
                        f"{c['improved']}/{c['equal']}/{c['worse']}",f"{c['worst_leave_one_out_mean']:+.10f}"])
    component_rows=[]
    for ch in CHANNELS:
        a=component['totals'][ch+'_independent'];b=component['totals'][ch+'_reuse']
        component_rows.append([ch,a['rejected'],b['rejected'],b['additional_rejections'],b['transfer_rejections'],
                               a['total_response_pairs'],b['total_response_pairs'],f"{a['seconds']:.6f}",f"{b['seconds']:.6f}"])
    main=by[PRIMARY];old=comparisons[PRIMARY,'unvisited_dual','mse257'];zero=comparisons[PRIMARY,'cross_zero_reuse_g8','mse257']
    off=comparisons[PRIMARY,'cross_dual_independent_g8','mse257'];adam=comparisons[PRIMARY,'cross_native_adam240','mse257']
    title=f"主候选MSE257 **{main['metrics']['mse257']:.10f}**；相对不传播同初始化 **{off['mean_difference']:+.10f}**，相对同增强zero **{zero['mean_difference']:+.10f}**，相对同期Adam240 **{adam['mean_difference']:+.10f}**。负值表示主方法更低。"
    save(out/'table_data.json',dict(current=rows,references=refrows,component=component_rows))
    save(out/'figure_data.json',dict(methods=figure_methods,component={c:{k:component['totals'][c+'_'+k]['rejected'] for k in ['independent','reuse']} for c in CHANNELS}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(13,7));xx=np.arange(14)
    labels=[m['method'].replace('cross_','').replace('_',' ') for m in figure_methods]
    axes[0].barh(xx,[m['mse'] for m in figure_methods],color=['#147d75' if m['method']==PRIMARY else '#8ea7bc' for m in figure_methods])
    axes[1].barh(xx,[m['seconds'] for m in figure_methods],color=['#147d75' if m['method']==PRIMARY else '#8ea7bc' for m in figure_methods])
    for ax in axes:
        ax.set_yticks(xx,labels if ax is axes[0] else []);ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18)
    axes[0].set_xlabel('Mean unseen-query MSE (lower is better)');axes[1].set_xlabel('Mean complete fit seconds (one repetition)')
    axes[0].set_title('Fixed primary and all 13 concurrent controls',loc='left',pad=14)
    axes[1].set_title('Full parent + search + geometry + readout',loc='left',pad=14)
    fig.subplots_adjust(left=.24,right=.98,bottom=.18,top=.9,wspace=.17)
    fig.text(.24,.07,'32 exposed development tasks. G limits extra geometry calls, not total time or memory.',fontsize=10)
    fig.text(.24,.025,'Every method rebuilds its own trajectory and candidates. No stored pool or query answer is an online input.',fontsize=9)
    fig.savefig(out/'cross_region_risk_cost.png',dpi=170);plt.close(fig)
    text=['# 359｜从跨区域证书到有限求解预算：真实在线结果','',title,'',
        f"本轮{ps['counts']['new_calls']}次新的完整fit，{es['methods']}种方法、{es['scored_predictors']}个预测器；执行失败{ps['checks']['failures']}。主配置在评分前固定，不根据其它信用或预算更优而换主。这是32任务开发结果，不是新任务确认或完整论文结论。",'',
        '## 数学机制：安全信用传播与有限预算搜索','',
        '对固定观测和区域R，D_R(a)=min_(y∈Y_R)aᵀr(y)>0是严格不可行证书。357允许局部自由信用在区域内更新，把新获证书的方向用于其它区域重新计算D。源区域的数值不直接搬给目标区域；所有排除都精确复核。每8次检查传播新方向、每任务最多16个，局部上限128。','',
        '传播不改变尚未排除区域的本地更新，因此不传播算法在任意时刻能够排除的区域，传播算法也已排除或会在原时刻排除。这是覆盖包含保证，不是ALM独有定理。相同有序池取前G个未排除区域时，可以把几何预算让给更后面的候选，但更多有效区域并不保证有限粒子查询风险下降。','',
        '```text\n本次观察 x/v → 本次629起点母轨迹 → 当前已访问F（原有效池保留）\n                     ↓\n必要单观察语言乘积 → 排除F → C5/C20 → 区域内自由信用\n                                         ↕ 新方向传播到其它区域\n                                  取前G=8个未排除模式\n                                         ↓\n                     实际全局几何（公开计费）→ 2048粒子 → 未见查询\n```','',
        '## 组件结果：传播确实产生新证书','',
        '必要语言共277,604个模式，经排除本次已访问模式和区间收缩后剩601个。共同几何核验为548不可行、47正体积、6零体积或未判空；没有把这些分类输入局部候选。','',
        table(['初始化','不传播排除','传播排除','净新增','传播命中','不传播响应对','传播总响应对','不传播组件秒','传播组件秒'],component_rows),'',
        '主dual新增19个严格排除，其中41个区域由跨区域证书命中；41与19不同，因为部分命中只提前替代了本地原本也会获得的证书。总响应量仍增加，不能只按减少的局部响应量宣称加速。zero256不传播排除231个、组件2.127790秒；额外普通迭代也是必须保留的竞争路线。','',
        '## 14种方法的同期真实在线结果','',
        table(['方法','MSE257','完整均秒','失败','额外几何调用总数','主−该方法','主改善/同/变差'],rows),'',
        '![未见查询误差及完整运行费用](cross_region_risk_cost.png)','',
        '所有新方法实际重新计算母轨迹、触发、必要语言、筛选、几何、采样和读出，不免费加载357的候选池或信用。G=8是区域几何调用预算，不是总墙钟或总状态预算；C20的G=16/完整池、zero256和同期Adam是明确增强的强对照。每任务一次随机交错计时，无重复置信或进程峰值匹配结论。','',
        '### 本轮改善的明确归因','',
        f"六种128步传播、dual/zero不传播，与无信用C20-G8对照，均在全部32任务得到完全相同的正区域池、粒子、分配和预测；额外逐位核验{attribution['bitwise_arrays']}个数组、{attribution['identical_positive_pools']}个池通过。因此，相对旧版的预测改善已由新的无信用候选流程复现，不能归因于PC-ALM或传播。",'',
        f"主信用方法比C20-G8少做{attribution['saved_geometry_calls_vs_reference'][PRIMARY]}次额外几何调用，但本次完整均时{main['mean_current_seconds']:.6f}秒，C20-G8为{by['cross_c20_g8']['mean_current_seconds']:.6f}秒；没有建立净提速。传播相对主不传播又省7次几何调用，但单次计时波动不能当作稳定速度优势。",'',
        'C20完整剩余池得到更低的0.0504878776 MSE，但这是预列强控制，不将其事后替换为主方法或归入PC贡献。下一步必须针对决定查询质量的候选覆盖和总成本提出新机制，不能继续把远离有限预算前缀的新增证书数量当作核心收益。','',
        '## 原有强优化器和回归对照仍保留','',
        table(['旧冻结对照','MSE257','主−对照','主改善/同/变差','最差留一均差'],refrows),'',
        f"相对旧unvisited_dual，本轮主均差{old['mean_difference']:+.10f}，改善/相同/变差{old['improved']}/{old['equal']}/{old['worse']}。收益可能来自更完整的必要语言池、区间收缩、固定几何预算下的安全删空或信用传播；必须用同增强控制分开归因，不能统称PC-ALM作用。",'',
        '全部151方法、四指标和8,400条配对描述性比较均保留在评分文件。旧方法没有本轮计时，不能拼接历史时间给出同期速度排名。当前暴露任务的留一检查不是显著性证明；核/回归类比较不是排除任意闭式解。','',
        '## 审计、失败和边界','',
        f"独立重算{audit['counts']['risk_fields']:,}个风险字段和{audit['counts']['comparison_rows']:,}条比较；最大标量差{audit['maximum_scalar_risk_gap']:.3g}。另核验{audit['counts']['credit_certificates']:,}个局部/传播信用证书、{audit['counts']['geometry_certificates']:,}个几何证书，以及原轨迹、原有效池保留、支持粒子和有序前缀包含。",'',
        '357首批随机小预检没有覆盖传播路径，因覆盖断言失败保留原测试；补充早已使用的5900001合成原语后通过，算法和正式任务预算没有修改。358首任务14种预检全部通过后才进入32任务；全部预测封存后才读取查询答案。','',
        '本项目仍未完成官方TTT-MLP、LLM/VLM或其它真实下游验证。阶段性查询结果、覆盖定理与最终独立优势是不同要求；核心目标仅在同增强非乘子/强BP、强回归、资源与新任务证据同时支持时才能完成。','',
        '[357数学协议](../../../outputs/ttt-pc-alm-research/357_cross_region_credit_protocol_v1.md) · [358冻结在线协议](../../../outputs/ttt-pc-alm-research/358_cross_region_online_protocol_v1.md) · [全部151方法](../development_evaluation_v1/methods.json) · [全部配对比较](../development_evaluation_v1/comparisons.json) · [独立审计](../development_audit_v1/summary.json)','']
    write(out/'report.md','\n'.join(text));write(root/ENTRY,'# 359｜跨区域信用的实际在线比较\n\n'+title+'\n\n[完整图文和全部对照](../../results/cross_region_online/report_v1/report.md)\n')
    save(out/'manifest.json',dict(passed=True,source_sha256=sha(Path(__file__)),entry_file=ENTRY,entry_sha256=sha(root/ENTRY),
        input_summary_sha256={BASE+'/development_audit_v1/summary.json':sha(base/'development_audit_v1/summary.json'),
                              BASE+'/attribution_v1/summary.json':sha(base/'attribution_v1/summary.json')},
        outputs_sha256={n:sha(out/n) for n in ['report.md','table_data.json','figure_data.json','cross_region_risk_cost.png']}))
    print(dict(passed=True,report=str(out/'report.md')),flush=True)


if __name__=='__main__':main()
