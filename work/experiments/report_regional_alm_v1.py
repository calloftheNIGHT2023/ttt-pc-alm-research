"""377 active regional multiplier feedback: all controls and complete costs."""
from pathlib import Path
import math
import numpy as np
from run_regional_alm_v1 import BASE,model,read,sha,save,complete
from report_search_radius_development_v1 import table,write


def main():
    root=Path(__file__).resolve().parents[2];base=root/BASE;s=complete(base/'development_v1');audit=complete(base/'audit_v1')
    assert audit['development_summary_sha256']==sha(base/'development_v1/summary.json')
    groups=read(base/'audit_v1/groups.json');pairs=read(base/'audit_v1/paired.json');cp=read(base/'audit_v1/checkpoints.json');idx={(g['method'],g['source_status']):g for g in groups}
    for file,h in read(base/'development_v1/protocol.json')['source_sha256'].items():assert sha(root/file)==h
    rows=[]
    for name in model.METHODS:
        f=idx[name,'budget_exhausted'];c=idx[name,'complete'];rr=[r for r in pairs if r['control']==name and not r['source_completed']]
        rows.append([name,f['certified'],f"{1000*f['mean_total_seconds']:.3f}",c['certified'],f"{1000*c['mean_total_seconds']:.3f}",sum(r['primary_only'] for r in rr),sum(r['control_only'] for r in rr)])
    curves=[]
    for name in ['regional_active1024','regional_passive1024','pdhg_cold1024','pdhg_box1024']:
        for step in [0,32,64,128,256,512,1024]:
            rr=[r for r in cp if r['method']==name and r['step']==step and not r['source_completed']];assert len(rr)==16
            curves.append(dict(method=name,step=step,certified=sum(r['certified'] for r in rr),mean_checkpoint_ms=1000*math.fsum(r['seconds'] for r in rr)/16))
    headline='主动区域乘子128步排除958/1023个已证不可行样本；相同原始更新但仅被动累积为343，瞬时残差343，PDHG冷启动580、同盒初始状态574。完整组件均时分别28.168、27.703、26.724、26.729、26.313毫秒。'
    out=base/'report_v1';out.mkdir(parents=True,exist_ok=False);save(out/'table_data.json',rows);save(out/'figure_data.json',dict(groups=groups,curves=curves))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,6.3));yy=np.arange(9)
    names=model.METHODS
    axes[0].barh(yy,[idx[n,'budget_exhausted']['certified'] for n in names],color=['#16847b' if n==model.PRIMARY else '#aab7c5' for n in names])
    axes[0].set_yticks(yy,[n.replace('regional_','').replace('pdhg_','PDHG ').replace('_',' ') for n in names]);axes[0].invert_yaxis();axes[0].set_xlim(0,1130)
    for i,n in enumerate(names):val=idx[n,'budget_exhausted']['certified'];axes[0].text(val+10,i,str(val),va='center',fontsize=9)
    axes[0].set_title('Active feedback changes proof discovery',loc='left',pad=12);axes[0].set_xlabel('Certified exclusions / 1,023 infeasible samples')
    for name,label,color in [('regional_active1024','Regional active','#16847b'),('regional_passive1024','Passive history','#be8d40'),('pdhg_cold1024','PDHG cold','#5878a2'),('pdhg_box1024','PDHG same box state','#955d86')]:
        rr=[r for r in curves if r['method']==name];axes[1].plot([r['mean_checkpoint_ms'] for r in rr],[r['certified'] for r in rr],'-o',markersize=4,label=label,color=color)
    axes[1].set_xlim(0,122);axes[1].set_ylim(-30,1100);axes[1].set_title('Coverage versus measured component cost',loc='left',pad=12)
    axes[1].set_xlabel('Mean cumulative milliseconds per frontier');axes[1].set_ylabel('Certified exclusions');axes[1].legend(loc='lower right',frameon=False,fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.14)
    fig.subplots_adjust(left=.16,right=.98,top=.86,bottom=.20,wspace=.34)
    fig.text(.16,.08,'All nine controls shown. Main: active128. Same information, explicit state/cost accounting; not equal FLOPs.',fontsize=9)
    fig.text(.16,.03,'Old sampled frontiers; no query-risk result. Curves are checkpoints of long runs, not independent shorter timings.',fontsize=9)
    fig.savefig(out/'regional_alm.png',dpi=165);plt.close(fig)
    curvetable=[[r['method'],r['step'],r['certified'],f"{r['mean_checkpoint_ms']:.3f}"] for r in curves if r['step'] in [64,128,256,512,1024]]
    body=['# 377｜当前区域内的主动乘子反馈：正向组件结果','',headline,'',
        '**可以支持的结论：**在这批固定旧前沿、预列预算内，主动反馈比同原始块更新的被动历史更快地产生有效不可行证书，并出现相对强PDHG有竞争力的覆盖—成本区间。**尚不能支持的结论：**新的未见查询收益、完整在线加速、PC-ALM不可替代或论文已完成。','',
        '## 1. 改变的不是模型容量，而是当前区域的约束反馈','',
        '374的历史来自原非线性求解轨迹，本轮乘子从零开始，只对当前待检分支的rp=z-h_prev-b、ra=h-sz-c累积。原始变量分三块作带近端的盒内二次最小化，随后乘子以.5倍残差更新。相同步数的被动对照只把残差积累在旁路，不让它改变原始更新；瞬时对照只用当前残差。所有方法都能提出当前残差证书，PDHG没有被限制成弱信用通道。','',
        '固定乘子时，三个可分离二次块的解析盒投影使E_new+τ||delta||²/2≤E_old，τ=.01。该结论不等于多块ALM全局收敛；拒绝区域仍只能依靠实际宽域的严格分离下界。局部导数/线性斜率均在所在层计算，无全局BP、LP或母状态生成。','',
        '[完整更新公式和预列对照](../../../outputs/ttt-pc-alm-research/376_regional_alm_protocol_v1.md)','',
        '## 2. 完整九组结果','',
        '失败前沿16个，1024样本中1023不可行；完成前沿32个，1093样本中1067不可行。后两列为失败前沿上的固定主独有与控制独有证明，不能当作独立任务数。所有配置0步证明数为0。','',
        table(['配置','失败前沿排除','完整均毫秒','完成前沿排除','完整均毫秒','主独有','控制独有'],rows),'',
        '![主动反馈与完整成本](regional_alm.png)','',
        '短预算主958远高于被动343，不是只给主多存一些历史：被动也累积相同类型的历史，区别在于是否反馈到下一次原始更新。被动与瞬时128的原始终态逐位相同，盒投影PDHG与主初始b,z,h逐位相同。计时是本机一次交错调用，不是重复计时置信结论；相同步数也不是相同FLOPs。','',
        '## 3. 强预算控制与工作曲线','',table(['长轨迹来源','检查点步数','失败前沿排除','累计均毫秒'],curvetable),'',
        '主动64检查点736个、约21.784毫秒；PDHG冷128检查点580个、约26.620毫秒。主动256与PDHG冷512都达到1010个，累计均时约39.794与58.658毫秒。它们是不同固定长轨迹中的预列检查点，不是独立短运行的最终完整计时，不把该比率直接称端到端加速。','',
        '主动1024排除1022，强PDHG1024排除1019；长主略多但完整费用也更高，不能只比较证书数量。固定主仍是128，不用长主替换。下一关口应使用完整树实际预算和重复资源测量，而非继续用单独组件的平均时间保证效果。','',
        '## 4. 下一实验为何有依据、还缺什么','',
        '这次改变了后续动作：已有同初始状态、同信息、同信用通道下的主动反馈增量，且有竞争力的有限成本区间，因此可以进入完整前缀求交与后验读出开发测试。下一主保持regional_active128，不事后改成最有利检查点；须有passive128、cold/box PDHG128/256/512和无额外证书的C20基线，统计生成候选、筛选、最终几何与读出的全部工作。','',
        '最终主指标仍为未见query误差，不是删除更多分支。只有完整预算下胜过强简单控制，才值得启用新冻结任务，并与原TTT更新、回归/核、浅层头和同参数强优化器比较。本轮全部为旧已暴露任务的采样组件，没有新查询预测，也不满足持续目标完成条件。通用ALM/二次块更新本身不宣称新颖性。','',
        '## 5. 完整审计','',
        f'432正式调用、0执行异常；独立{audit["checks"]["exact_wide_domain_proofs"]}个宽域Fraction正证书、{audit["checks"]["final_box_checks"]}最终盒约束、{audit["checks"]["checkpoints"]}检查点、{audit["checks"]["same_initial_arrays"]}同初始状态数组、{audit["checks"]["passive_instant_primal_arrays"]}被动/瞬时原始终态、{audit["checks"]["long_short_prefix_matches"]}长短工作前缀和243正样本保留通过。预检2484标量计算最大差4.86e-17、81块下降不等式通过。','',
        '所有输出封存后才读取372旧几何标签做审核；标签与LP权重不参与求解。没有查询答案、后缀观察或BP信用初始化。状态统计为命名数组，不是进程峰值；组件时间不含原建树、最终几何和读出。样本跨排序/同任务相关，不做独立新任务显著性声明。','',
        '[全部调用](../development_v1/rows.json) · [独立审计](../audit_v1/summary.json) · [配对结果](../audit_v1/paired.json) · [预检与块证明检查](../preflight_v1/block_tests.json)','']
    write(out/'report.md','\n'.join(body));entry='outputs/ttt-pc-alm-research/377_regional_alm_results_v1.md';write(root/entry,'# 377｜主动区域反馈组件结果\n\n'+headline+'\n\n[完整图文](../../results/regional_alm/report_v1/report.md)\n')
    save(out/'manifest.json',dict(source_sha256=sha(Path(__file__)),entry_file=entry,entry_sha256=sha(root/entry),
        input_summary_sha256={p:sha(base/p) for p in ['development_v1/summary.json','audit_v1/summary.json']},
        outputs_sha256={f:sha(out/f) for f in ['table_data.json','figure_data.json','report.md','regional_alm.png']}))
    content=(out/'report.md').read_text(encoding='utf-8');cells=0
    for r in rows+curvetable:assert '| '+' | '.join(map(str,r))+' |' in content;cells+=len(r)
    assert [idx[n,'budget_exhausted']['certified'] for n in ['regional_active128','regional_passive128','pdhg_cold128','pdhg_box128']]==[958,343,580,574]
    save(out/'qa_numeric.json',dict(passed=True,table_cells=cells,methods=9,curve_points=len(curves),manifest_sha256=sha(out/'manifest.json')))
    print(dict(passed=True,report=str(out/'report.md')),flush=True)


if __name__=='__main__':main()
