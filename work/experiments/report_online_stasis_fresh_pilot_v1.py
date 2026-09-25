"""307 all46 / all180 report. Refuses to run before complete independent audit."""
import argparse
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_gradient_flat_split_states_v1 import read,save,sha

PRIMARY='online_stasis_alm_keep64'
KEY_CONTROLS=['credit_control_probe33','probe_alm40_33','probe_alm48_33','probe_alm64_33',
    'probe_then_adam240_33','probe_then_adam480_33','probe_then_nodual33','probe_then_pc33',
    'online_stasis_alm_reset64','online_stasis_nodual64','online_stasis_pc64','online_stasis_adam64',
    'online_stasis_alm_random_sign64','online_stasis_alm_residual64',
    'online_stasis_adam_perturb_00164','online_stasis_adam_perturb_00464']
METRICS=['mse257','mse129','point_mse257','point_mse129']


def run(root,out):
    base=root/'results/online_stasis_fresh_pilot'
    pred=base/'pilot_predictions_v1';evaluation=base/'pilot_evaluation_v1';audited=base/'audit_v1'
    audit=read(audited/'summary.json');assert audit['passed'] and audit['counts']['predictors']==11776
    assert audit['counts']['comparisons']==180 and audit['all_bootstrap_means_checked']==3600000
    ss=read(pred/'summary.json');es=read(evaluation/'summary.json')
    assert ss['passed'] and es['passed'] and es['tasks']==256 and es['main_family']==45
    assert audit['prediction_summary_sha256']==sha(pred/'summary.json')
    assert audit['evaluation_summary_sha256']==sha(evaluation/'summary.json')
    for folder,s in [(pred,ss),(evaluation,es)]:
        for name,digest in s['outputs_sha256'].items():assert sha(folder/name)==digest
    p=read(evaluation/'protocol.json');methods=read(evaluation/'methods.json');comp=read(evaluation/'comparisons.json')
    costs=read(pred/'costs.json');names=p['methods'];byname={r['method']:r for r in methods}
    assert len(methods)==46 and [r['method'] for r in methods]==names
    assert len(comp)==len({(r['control'],r['metric']) for r in comp})==180
    lookup={(r['control'],r['metric']):r for r in comp}
    main=[lookup[n,'mse257'] for n in names if n!=PRIMARY]
    assert len(main)==45 and all(n in byname for n in KEY_CONTROLS)
    with np.load(evaluation/'task_metrics.npz',allow_pickle=False) as z:
        risk=z['risk'];times=z['seconds'];failures=z['failures']
        assert z['methods'].tolist()==names and z['seeds'].tolist()==list(range(307000000,307000256))
    assert risk.shape==(256,46,4);pi=names.index(PRIMARY);checks=0
    for i,name in enumerate(names):
        r=byname[name]
        for j,metric in enumerate(METRICS):
            assert r['metrics'][metric]==float(risk[:,i,j].mean());checks+=1
            if name!=PRIMARY:
                c=lookup[name,metric]
                assert c['mean_difference']==float((risk[:,pi,j]-risk[:,i,j]).mean());checks+=1
                assert c['improved']+c['equal']+c['worse']==256
        assert math.isclose(r['mean_seconds'],float(times[:,i].mean()),rel_tol=1e-13,abs_tol=1e-14)
        assert r['failures']==int(failures[:,i].sum())
    assert sum(r['adjusted_negative'] for r in main)==es['adjusted_negative_comparisons']
    within=[n for n in costs['within_budget'] if n!=PRIMARY]
    sensitivity=[n for n in costs['sensitivity_110_percent'] if n!=PRIMARY]
    wins=lambda names:sum(lookup[n,'mse257']['adjusted_negative'] for n in names)
    colors={n:('#a16207' if byname[n]['budget_class']=='over_1.10' else '#2563eb' if byname[n]['budget_class']=='1.10' else '#64748b') for n in names}
    colors[PRIMARY]='#0f766e'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(14.5,14.5));y=np.arange(46)
    ax.barh(y,[byname[n]['metrics']['mse257'] for n in names],color=[colors[n] for n in names],height=.68)
    ax.set_yticks(y,names,fontsize=8);ax.invert_yaxis()
    ax.set_xlabel('Mean unseen-query MSE (257 query points per task; lower is better)')
    ax.set_title('256 NEW development tasks / all 46 methods\n4 observed support pairs; all predictions sealed before evaluation',fontsize=13,pad=14)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.35,right=.97,top=.93,bottom=.07)
    fig.text(.35,.022,'Teal: candidate; gray: <=1.00x cost; blue: <=1.10x; ochre: >1.10x. No method removed.',fontsize=9)
    fig.savefig(out/'all_methods.png',dpi=150);plt.close(fig)

    def contrasts(selected,filename,title,height):
        fig,ax=plt.subplots(figsize=(14.5,height))
        for i,name in enumerate(selected):
            r=lookup[name,'mse257'];lo,hi=r['descriptive95'];col=colors[name]
            ax.plot([lo,hi],[i,i],color=col,linewidth=1.7)
            ax.scatter([r['mean_difference']],[i],color=col,s=20,zorder=3)
            ax.scatter([r['bonferroni_upper']],[i],color='#111111',marker='x',s=24,zorder=4)
        ax.axvline(0,color='#666666',linestyle='--',linewidth=1)
        ax.set_yticks(range(len(selected)),selected,fontsize=8 if len(selected)>20 else 9)
        ax.invert_yaxis();ax.set_xlabel('Candidate minus control MSE (left of zero favors candidate)')
        ax.set_title(title,fontsize=13,pad=14);ax.grid(axis='x',alpha=.15)
        ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
        fig.subplots_adjust(left=.37,right=.975,top=.92,bottom=.10 if height<10 else .065)
        fig.text(.37,.025,'Dot: mean; line: descriptive 95% interval; x: Bonferroni one-sided upper (45 primary contrasts).',fontsize=8)
        fig.savefig(out/filename,dpi=150);plt.close(fig)
    contrasts([n for n in names if n!=PRIMARY],'all_primary_contrasts.png','All 45 primary comparisons; no outcome-based subset',14.5)
    contrasts(KEY_CONTROLS,'key_controls.png','Prelisted same-parameter and same-trigger controls',8.5)
    candidate=byname[PRIMARY]
    lines=['# 307：在线停滞续接的新任务预实验','',
        '## 本轮要回答的问题','',
        '在同样已观察上下文、冻结先验与完整读出下，保留局部乘子的停滞续接，能否改善未见查询预测，并超出额外迭代、普通PC、清零乘子或固定扰动Adam的作用？','',
        '固定256个新开发任务（307000000–307000255），46个方法，共11776份预测。全部预测及成本分类封存后才统一打开查询真值；138份旧任务前检不是本表的新样本。查询点不是统计独立样本，重抽单位为任务。','',
        '任务仍是四层标量折叠网络、四维内部偏置、四个带均匀噪声的支持观测及257个未见查询。这是可控机制任务，不是原官方TTT、一般GELU MLP或LLM/VLM验证。候选保留原路径与support-best，只把局部可观察的预测停滞状态复制出来保留乘子续接64步；所有控制使用共同分支几何和读出。','',
        '## 完整结果摘要','',
        f'候选主查询MSE为 **{candidate["metrics"]["mse257"]:.12g}**，完整调用均值 **{candidate["mean_seconds"]:.6f}秒**。全部45个主比较中，近似多重校正单侧上界为负的有 **{es["adjusted_negative_comparisons"]}/45**。',
        f'候选均值成本内为{wins(within)}/{len(within)}；1.10倍敏感性成本内为{wins(sensitivity)}/{len(sensitivity)}。所有超预算方法仍完整报告，成本分类未使用查询质量。',
        f'全部数值失败（按既定零偏置回退并计费）共{es["numerical_failures"]}次。失败数不等于完整的数学稳定性证明。','',
        '这些计数不能单独证明“必须使用PC-ALM”；需要具体查看下面预列强控制和同位置干预。如果仅回归/浅头被超过，而同位置清零、扰动或强BP没有可靠差距，尚不足以归因于保留的局部信用。','',
        f'![全部46方法]({(out/"all_methods.png").as_posix()})','',
        f'![预列强控制]({(out/"key_controls.png").as_posix()})','',
        '## 全部方法表','',
        '|方法|MSE257|MSE129|点估计MSE257|点估计MSE129|完整均值秒|相对候选成本|成本类|数值失败|',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in methods:
        lines.append('|'+r['method']+'|'+'|'.join(f'{r["metrics"][m]:.12g}' for m in METRICS)+
            f'|{r["mean_seconds"]:.9g}|{r["actual_ratio_to_primary"]:.9g}|{r["budget_class"]}|{r["failures"]}|')
    lines+=['','## 全部180个比较','',
        '负均值差有利于候选。区间采用20000次配对任务bootstrap；95%区间是描述性的，主MSE257另给1−.05/45分位的单侧上界。它们是近似百分位区间，不是有限样本严格覆盖保证。辅助指标不替换主指标。','',
        f'![全部45主比较]({(out/"all_primary_contrasts.png").as_posix()})','',
        '|控制|指标|候选−控制|95%下界|95%上界|主比较校正单侧上界|改善/相同/更差任务|',
        '|---|---|---:|---:|---:|---:|---|']
    for r in comp:
        upper='—' if r['bonferroni_upper'] is None else f'{r["bonferroni_upper"]:.12g}'
        lines.append(f'|{r["control"]}|{r["metric"]}|{r["mean_difference"]:.12g}|{r["descriptive95"][0]:.12g}|{r["descriptive95"][1]:.12g}|{upper}|{r["improved"]}/{r["equal"]}/{r["worse"]}|')
    lines+=['','## 数学联系与尚需证明的部分','',
        '平方查询风险可分解为完整后验的不可约条件方差与预测均值偏差。因此新增分支是否有益，取决于它是否修正均值偏差，而不是分支数或参数数目是否增加。294的三区域风险恒等式给出条件性改进判据；本轮测试其真实有限计算后果，不把条件命题升级为对任意闭式回归或浅网络的普遍优势。','',
        '完整计费包括原适应轨迹、检测、影子续接、几何及共同修复、采样、读出和投影；模型加载与I/O另列。costs.json保留所有方法的命名状态字节字段，但本轮没有统一原生峰值内存测量，不能用小计宣称严格内存优势。更便宜控制仍可能有尚未使用的预算空间。','',
        '这是预先固定的新开发预实验。最终任务效果、样本效率、有限步稳定性、严格预算匹配及一般模型桥接仍须按证据逐项推进，不以本报告或某个显著比较宣布完整论文目标达成。','',
        '## 可追溯证据','',
        f'独立审计重新核对{audit["counts"]["predictors"]}份预测、{audit["counts"]["scalar_risks"]}个标量风险及{audit["all_bootstrap_means_checked"]}个bootstrap均值；标量风险最大差{audit["maximum_scalar_risk_gap"]:.12g}，重抽均值最大差{audit["maximum_bootstrap_gap"]:.12g}。','',
        f'- [独立审计]({(audited/"summary.json").as_posix()})',
        f'- [评分摘要]({(evaluation/"summary.json").as_posix()})',
        f'- [完整成本和状态字段]({(pred/"costs.json").as_posix()})',
        f'- [预先固定协议]({(root/"outputs/ttt-pc-alm-research/307_online_stasis_fresh_pilot_protocol.md").as_posix()})','']
    with (out/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    actual=(out/'report.md').read_text(encoding='utf-8').splitlines()
    method_lines=[s for s in actual if s.startswith('|') and len(s.split('|'))==11 and s.split('|')[1] in byname]
    contrast_lines=[s for s in actual if s.startswith('|') and len(s.split('|'))==9 and (s.split('|')[1],s.split('|')[2]) in lookup]
    assert len(method_lines)==46 and len(contrast_lines)==180
    save(out/'figure_data.json',dict(methods=methods,primary_comparisons=main,key_controls=KEY_CONTROLS))
    save(out/'summary.json',dict(passed=True,tasks=256,methods=46,comparisons=180,metric_mean_checks=checks,
        method_table_rows=46,comparison_table_rows=180,visual_review_required=True,core_research_goal_complete=False,
        source_sha256={Path(__file__).name:sha(Path(__file__))},audit_summary_sha256=sha(audited/'summary.json'),
        outputs_sha256={n:sha(out/n) for n in ['report.md','all_methods.png','all_primary_contrasts.png','key_controls.png','figure_data.json']}))
    print(dict(passed=True,report=str(out/'report.md'),visual_review_required=True),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();out=root/'results/online_stasis_fresh_pilot/report_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise
