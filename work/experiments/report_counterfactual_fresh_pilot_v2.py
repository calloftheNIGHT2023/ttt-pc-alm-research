"""294 report v2: fixed in-panel labels; unchanged scientific tables and figure data."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import counterfactual_fresh_pilot_v1 as pilot
from diagnose_gradient_flat_split_states_v1 import read,sha


KEY_CONTROLS = ['credit_control_probe33','probe_alm40_33','probe_alm48_33','probe_alm64_33',
                'probe_then_adam480_33','probe_then_adam960_33','probe_pc128_33','probe_pc256_33',
                'probe_nodual64_33','probe_nodual128_33']


def run(root,out,stage):
    folder=out.parent/(stage+'_evaluation_v1');s=pilot.suite.complete(folder)
    pred=out.parent/(stage+'_predictions_v1');ps=pilot.suite.complete(pred)
    p=read(folder/'protocol.json');hashes=pilot.gate(root)
    assert p['source_sha256']==hashes
    assert p['prediction_summary_sha256']==sha(pred/'summary.json')
    assert p['design_sha256']==sha(root/pilot.DESIGN)
    assert s['independent_scalar_risk_passed'] and s['all_bootstrap_means_checked']==2880000
    assert s['tasks']==(2 if stage=='preflight' else 128) and ps['predictors']==37*s['tasks']
    assert s['stage']==stage and s['comparisons']==144 and s['main_family']==36
    methods=read(folder/'methods.json');comp=read(folder/'comparisons.json')
    names=p['methods'];assert [r['method'] for r in methods]==names and len(names)==37
    assert len(comp)==len({(r['control'],r['metric']) for r in comp})==144
    byname={r['method']:r for r in methods};lookup={(r['control'],r['metric']):r for r in comp}
    with np.load(folder/'task_metrics.npz',allow_pickle=False) as z:
        assert z['risk'].shape==(s['tasks'],37,4)
        assert z['methods'].tolist()==names and z['metric_names'].tolist()==pilot.METRICS
        assert z['seeds'].tolist()==p['seeds']
        risk=z['risk'];seconds=z['seconds'];failures=z['failures']
    with np.load(folder/'bootstrap_means.npz',allow_pickle=False) as z:boot=z['means']
    assert boot.shape==(20000,36,4);pi=names.index(pilot.suite.PRIMARY);assert pi==0
    checked=0
    for j,name in enumerate(names):
        row=byname[name]
        assert row['mean_seconds']==float(seconds[:,j].mean()) and row['failures']==int(failures[:,j].sum())
        for k,metric in enumerate(pilot.METRICS):
            assert row['metrics'][metric]==float(risk[:,j,k].mean())
            if j==0:continue
            d=risk[:,0,k]-risk[:,j,k];rr=lookup[name,metric]
            assert rr['mean_difference']==float(d.mean())
            assert rr['descriptive95']==np.quantile(boot[:,j-1,k],[.025,.975]).tolist()
            if k==0:assert rr['bonferroni_upper']==float(np.quantile(boot[:,j-1,k],1-.05/36))
            assert rr['improved']+rr['equal']+rr['worse']==s['tasks'];checked+=1
    assert checked==144
    main=[lookup[name,'mse257'] for name in names[1:]]
    assert sum(r['adjusted_negative'] for r in main)==s['adjusted_negative_comparisons']
    colors={name:'#175f98' if byname[name]['calibration_class']=='1.00' else '#b07418' for name in names}
    colors[pilot.suite.PRIMARY]='#d03d58'
    title='OLD2 functional preflight only' if stage=='preflight' else '128 NEW development tasks, 4 observed support pairs'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(15.5,13.5),layout='constrained')
    y=np.arange(37)
    ax[0].barh(y,[byname[n]['metrics']['mse257'] for n in names],color=[colors[n] for n in names],height=.65)
    ax[0].set_yticks(y,names,fontsize=8);ax[0].invert_yaxis();ax[0].set_xlabel('Mean query MSE (257 points; lower is better)')
    ax[0].set_title('All 37 predictors; none removed')
    for i,r in enumerate(main):
        lo,hi=r['descriptive95'];x=r['mean_difference'];col=colors[r['control']]
        ax[1].plot([lo,hi],[i,i],color=col,linewidth=1.5)
        ax[1].plot(x,i,'o',color=col,markersize=4)
        ax[1].plot(r['bonferroni_upper'],i,'x',color='#111111',markersize=4)
    ax[1].axvline(0,color='#555555',linestyle='--',linewidth=1)
    ax[1].set_yticks(np.arange(36),names[1:],fontsize=8);ax[1].invert_yaxis()
    ax[1].set_xlabel('Candidate minus control MSE; left of zero favors candidate')
    ax[1].set_title('All 36 primary contrasts; bars = descriptive 95% CI')
    ax[1].legend(handles=[Line2D([0],[0],marker='x',color='black',linestyle='none',label='Bonferroni one-sided upper')],loc='lower left',fontsize=8)
    fig.suptitle(title+'\nBlue: calibration <= 1.00; ochre: above 1.00; red: new candidate',fontsize=12)
    fig.savefig(out/'all_methods.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(14,6.5),layout='constrained')
    for i,name in enumerate(KEY_CONTROLS):
        r=lookup[name,'mse257'];lo,hi=r['descriptive95'];x=r['mean_difference']
        ax[0].plot([lo,hi],[i,i],color=colors[name],linewidth=2)
        ax[0].plot(x,i,'o',color=colors[name]);ax[0].plot(r['bonferroni_upper'],i,'x',color='black')
    ax[0].axvline(0,color='gray',linestyle='--');ax[0].set_yticks(range(len(KEY_CONTROLS)),KEY_CONTROLS)
    ax[0].invert_yaxis();ax[0].set_title('Prelisted strong controls (not outcome-selected)')
    ax[0].set_xlabel('Candidate minus control MSE; 95% CI and adjusted upper x')
    for name in names:
        row=byname[name]
        ax[1].scatter(row['mean_seconds'],row['metrics']['mse257'],color=colors[name],s=55 if name==pilot.suite.PRIMARY else 25,zorder=3)
    annotated=[pilot.suite.PRIMARY,'credit_control_probe33','probe_alm48_33','probe_then_adam480_33',
               'cold__meta_ridge128','cold__meta_shallow64_20','cold__prior16384_ridge','cold__linear_ls']
    positions=[(.56,.33),(.55,.18),(.56,.26),(.56,.41),(.12,.29),(.14,.50),(.53,.51),(.12,.74)]
    for name,position in zip(annotated,positions):
        row=byname[name];ax[1].annotate(name,(row['mean_seconds'],row['metrics']['mse257']),
            xytext=position,textcoords='axes fraction',fontsize=7,annotation_clip=False,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1),
            arrowprops=dict(arrowstyle='-',color='#bbbbbb'))
    ax[1].set_xscale('log');ax[1].set_xlabel('Mean complete call seconds (log scale; excludes loading/I-O)')
    ax[1].set_ylabel('Mean query MSE');ax[1].set_title('All 37 costs and errors; no free extra proposals')
    fig.suptitle(title,fontsize=12);fig.savefig(out/'strong_controls.png',dpi=160);plt.close(fig)
    lines=['# 294｜新分支提议的查询结果' if stage=='pilot' else '# 294｜旧两任务报告前检（非科学结果）','',
        f'本表包含 {s["tasks"]} 个任务、37 个方法、144 个比较；全部预测先封存、后统一评分。',
        f'主比较校正上界为负的数量：{s["adjusted_negative_comparisons"]}/36。这个数量本身不代表完成研究目标。','',
        '![全部方法与主比较](all_methods.png)','', '![强控制与完整费用](strong_controls.png)','',
        '## 完整方法表','',
        '|方法|主 MSE257|MSE129|点估计 MSE257|点估计 MSE129|均值秒|本轮相对费用|校准相对费用|数值失败|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in methods:
        lines.append('|'+r['method']+'|'+ '|'.join(f'{r["metrics"][metric]:.9g}' for metric in pilot.METRICS)+
            f'|{r["mean_seconds"]:.6f}|{r["actual_ratio_to_primary"]:.4f}|{r["calibration_ratio_to_primary"]:.4f}|{r["failures"]}|')
    lines+=['','## 全部比较（候选减控制）','',
        '负差值有利于新候选。95% 区间是描述性的；校正单侧上界仅用于事先固定的 36 个主指标比较。任务是重抽单位，不把查询点当作独立样本。','',
        '|控制|指标|均值差|描述性 95% 下界|上界|校正单侧上界|改善/相同/更差任务|',
        '|---|---|---:|---:|---:|---:|---|']
    for r in comp:
        upper='—' if r['bonferroni_upper'] is None else f'{r["bonferroni_upper"]:.9g}'
        lines.append(f'|{r["control"]}|{r["metric"]}|{r["mean_difference"]:.9g}|{r["descriptive95"][0]:.9g}|'
            f'{r["descriptive95"][1]:.9g}|{upper}|{r["improved"]}/{r["equal"]}/{r["worse"]}|')
    lines+=['','## 范围与证据','',
        '这是四层标量机制任务上的新开发预实验，不是独立论文确认、一般 MLP 或官方 TTT、LLM/VLM 验证。不能把所有优势归因于局部信用，必须具体比较共同初始化、共同读出与相近预算下的强控制。' if stage=='pilot' else '这是旧两任务的报告功能前检，不产生新任务科学结论。',
        '资源校准和本轮费用都列出；快控制仍可能有可利用的预算余量。两任务 tracemalloc 观测不是每方法精确原生内存上界。统计区间采用近似百分位 bootstrap，不是有限样本保证。',
        f'独立标量风险最大差 {s["maximum_scalar_risk_gap"]:.9g}；全部 2,880,000 个重抽均值的直接索引复核最大差 {s["maximum_bootstrap_gap"]:.9g}。',
        f'评分摘要 SHA256：`{sha(folder/"summary.json")}`；预测摘要 SHA256：`{sha(pred/"summary.json")}`。',
        '报告图文仍需实际视觉检查；脚本成功不自动代表图表无重叠。']
    with (out/'report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    pilot.exclusive(out/'summary.json',dict(passed=True,stage=stage,tasks=s['tasks'],methods=37,comparisons=144,
        primary_comparisons=36,report_table_checks=checked,core_research_goal_complete=False,
        evaluation_summary_sha256=sha(folder/'summary.json'),prediction_summary_sha256=sha(pred/'summary.json'),
        report_source_sha256=sha(Path(__file__)),visual_review_required=True,
        outputs_sha256={name:sha(out/name) for name in ['report.md','all_methods.png','strong_controls.png']}))
    print(dict(passed=True,stage=stage,tasks=s['tasks'],methods=37,comparisons=144,visual_review_required=True),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--stage',choices=['preflight','pilot'],required=True);args=p.parse_args();root=args.project.resolve()
    out=root/'results/counterfactual_fresh_pilot'/f'{args.stage}_report_v2';out.mkdir(parents=True,exist_ok=False)
    try:run(root,out,args.stage)
    except BaseException as exc:
        pilot.exclusive(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()
