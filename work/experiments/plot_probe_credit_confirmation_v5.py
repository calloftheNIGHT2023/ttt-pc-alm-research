"""287 complete audited report and plots; no subset promoted after results.

Old-task preflight by default. Plot generation is not visual QA or goal success.
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--stage',choices=['preflight','confirmation'],default='preflight');args=ap.parse_args();root=args.project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';assert not (base/'predictions/RUNNING.lock').exists(),'No rendering alongside formal prediction timing'
    inp=base/('evaluation_preflight_v4' if args.stage=='preflight' else 'evaluation_v4');audit=base/('evaluation_audit_preflight_v4' if args.stage=='preflight' else 'evaluation_audit_v4')
    out=base/('figures_preflight_v5' if args.stage=='preflight' else 'figures_v5');out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    ss=json.loads((inp/'summary.json').read_text());aa=json.loads((audit/'summary.json').read_text());p=json.loads((inp/'protocol.json').read_text());ap0=json.loads((audit/'protocol.json').read_text())
    assert ss['passed'] and aa['passed'] and aa['evaluation_summary_sha256']==sha(inp/'summary.json')
    for folder,summary in [(inp,ss),(audit,aa)]:
        for name,h in summary['outputs_sha256'].items():assert sha(folder/name)==h
    hashes=dict(ap0['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for name,h in hashes.items():assert sha(src/name)==h,name
    if args.stage=='confirmation':
        old=base/'figures_preflight_v5';gate=json.loads((old/'summary.json').read_text());assert gate['passed'] and gate['functional_preflight_only']
        assert json.loads((old/'protocol.json').read_text())['source_sha256']==hashes
        for name,h in gate['outputs_sha256'].items():assert sha(old/name)==h
    names=p['methods'];primary=p['primary'];controls=p['controls'];methods=json.loads((inp/'methods.json').read_text());contrasts=json.loads((inp/'comparisons.json').read_text())
    mm={r['method']:r for r in methods};cc={(r['control'],r['metric']):r for r in contrasts}
    assert [r['method'] for r in methods]==names and len(mm)==27 and len(cc)==104
    note=root/'outputs/ttt-pc-alm-research/287_ols_moment_addendum.md';assert note.exists()
    exclusive_json(out/'protocol.json',dict(stage=args.stage,source_sha256=hashes,evaluation_summary_sha256=sha(inp/'summary.json'),evaluation_audit_sha256=sha(audit/'summary.json'),
        runtime_interference_note_sha256=sha(root/'outputs/ttt-pc-alm-research/287_runtime_resource_observation_20260924.md'),publication_interference_note_sha256=sha(root/'release/SCOPE.md'),head_reference_note_sha256=sha(root/'outputs/ttt-pc-alm-research/287_head_reference_diagnosis.md'),pool_budget_note_sha256=sha(root/'outputs/ttt-pc-alm-research/287_pool_budget_scope_correction.md'),moment_note_sha256=sha(note),method_order=names,contrast_order=controls,all_methods_retained=True,no_quality_subselection=True,
        difference_axis=dict(scale='symlog',linthresh=.001),scope='All27 methods and104 comparisons; frozen criterion, actual resource gaps and OLS moment caveat'))
    n=ss['tasks'];preflight=args.stage=='preflight';banner='OLD-TASK FUNCTIONAL PREFLIGHT - NOT CONFIRMATION' if preflight else f'Frozen independent confirmation: {n} tasks, 27 methods'
    suffix={'main_1.00':'','sensitivity_1.10':' [1.10]','measured_over_budget':' [over]'}
    labels=[name+suffix[mm[name]['old_calibration_resource_class']] for name in names]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    palette={'main_1.00':'#5d8187','sensitivity_1.10':'#9cafb3','measured_over_budget':'#cecece'}
    colors=['#c7742e' if name==primary else palette[mm[name]['old_calibration_resource_class']] for name in names]
    yy=np.arange(27);fig,axes=plt.subplots(1,2,figsize=(16,12),layout='constrained')
    axes[0].barh(yy,[mm[name]['metrics']['mse257']['mean'] for name in names],color=colors,height=.64)
    axes[0].set(yticks=yy,yticklabels=labels,xlabel='Actual query MSE; lower is better',title='A. All27 fixed predictors (257-point primary grid)')
    axes[0].invert_yaxis();axes[0].grid(axis='x',alpha=.2)
    axes[1].scatter([mm[name]['complete_time']['mean_seconds'] for name in names],yy,c=colors,s=35)
    axes[1].set(yticks=yy,yticklabels=['']*27,xlabel='Full original-input call, seconds (log axis)',title='B. Actual complete call times on these tasks',xscale='log')
    axes[1].axvline(mm[primary]['complete_time']['mean_seconds'],color='#c7742e',ls='--',lw=1);axes[1].invert_yaxis();axes[1].grid(axis='x',alpha=.2)
    fig.suptitle(banner,fontweight='bold',fontsize=13)
    fig.supxlabel('Labels/colors use frozen development budget classes. Actual new-task ratios, failures and state subtotals are all reported separately.',fontsize=9)
    fig.savefig(out/'287_all_methods.png',dpi=175);plt.close(fig)
    fig,ax=plt.subplots(figsize=(16,12),layout='constrained')
    for i,name in enumerate(controls):
        row=cc[name,'mse257'];mean=row['mean_difference'];lo,hi=row['descriptive95'];upper=row['bonferroni_one_sided_upper']
        color='#227a64' if row['adjusted_upper_below_zero'] else '#787b83'
        ax.hlines(i,lo,hi,color=color,lw=2);ax.plot(mean,i,'o',color=color,ms=5)
        if upper is not None:ax.plot(upper,i,'>',color=color,ms=7)
    ax.axvline(0,color='#aa4f40',ls='--',lw=1);ax.set_xscale('symlog',linthresh=.001)
    ax.set_yticks(np.arange(26),[name+(' [background]' if name=='probe_all_alm64' else '') for name in controls]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('Primary minus control MSE (negative favors primary); symlog with linear region +/-0.001')
    ax.set_title(banner+'\nAll26 contrasts: descriptive95% intervals and fixed25-comparison adjusted upper bounds',loc='left',fontweight='bold')
    ax.legend(handles=[Line2D([0],[0],color='#787b83',marker='o',label='Mean and descriptive95% interval'),
        Line2D([0],[0],color='#787b83',marker='>',ls='',label='One-sided Bonferroni upper bound'),
        Line2D([0],[0],color='#227a64',marker='o',ls='',label='Frozen numerical criterion: adjusted upper <0')],loc='upper left',bbox_to_anchor=(1.015,1),fontsize=8)
    fig.supxlabel('Bootstrap is approximate; task is the resampling unit. Raw OLS has a population-moment caveat. No finite-sample familywise guarantee is asserted.',fontsize=9)
    fig.savefig(out/'287_all_contrasts.png',dpi=175);plt.close(fig)
    passed=ss['main_adjusted_negative_upper_bounds'];pr=mm[primary];remaining=[r['control'] for r in contrasts if r['predeclared_main_comparison'] and not r['adjusted_upper_below_zero']]
    headline='旧两任务的功能前检，不是新任务确认结果' if preflight else '固定新任务确认：全部预定对照与资源结果'
    lines=[f'# 287｜{headline}','',f'任务数{n}；27方法；{ss["predictors"]}预测；先封存后开答案；独立风险和全部100000次bootstrap重算通过。','',
        '## 1．原判据的结果','',f'固定候选`{primary}`的主查询MSE为{pr["metrics"]["mse257"]["mean"]:.10g}。25项预定比较中，{passed}项校正上界低于零。',
        '这只是冻结的近似bootstrap数值判据；无论通过几项，都不自动证明有限样本家族错误率、所有回归方法的不可替代性或整个研究目标完成。','']
    if preflight:lines.extend(['**本页只有两个旧任务，用于检验报告代码。不得将任何数值作为确认实验、统计功效或科学结论。**',''])
    lines.extend(['未满足主数值阈值的预定控制：'+('、'.join('`'+name+'`' for name in remaining) if remaining else '无；仍需保留下述推断及资源边界。'),'','![全部方法与完整费用](287_all_methods.png)','','![全部预定比较](287_all_contrasts.png)','',
        '图中横线为描述性95%区间，三角为单侧Bonferroni上界；二者不是同一种区间。差值轴使用固定±.001线性区的对称对数比例，不能按视觉长度作线性比例解释。','',
        '## 2．全部方法，按冻结顺序而非表现排名','',
        '|方法|主MSE257|MSE129（次要）|点MSE257（次要）|完整均秒|p90秒|相对主时间|旧预算类|实际预算类|失败数|',
        '|---|---:|---:|---:|---:|---:|---:|---|---|---:|'])
    for r in methods:
        lines.append(f'|{r["method"]}|{r["metrics"]["mse257"]["mean"]:.10g}|{r["metrics"]["mse129"]["mean"]:.10g}|{r["metrics"]["point_mse257"]["mean"]:.10g}|{r["complete_time"]["mean_seconds"]:.7g}|{r["complete_time"]["p90_seconds"]:.7g}|{r["actual_mean_time_ratio_to_primary"]:.5g}|{r["old_calibration_resource_class"]}|{r["actual_mean_resource_class"]}|{r["failures"]}|')
    lines.extend(['','## 3．主比较与三组次要比较全部保留','',
        '主判据只有mse257。mse129、point_mse257和point_mse129不能替代它。全试探ALM64为预定超预算背景，仍报告但不列入主25阈值。',''])
    for metric in p['metrics']:
        lines.extend([f'### {metric}','','|控制|主减控制|配对标准差|描述95%下限|描述95%上限|校正单侧上界|主阈值|好/同/差任务|','|---|---:|---:|---:|---:|---:|---|---|'])
        for name in controls:
            r=cc[name,metric];lo,hi=r['descriptive95'];upper=r['bonferroni_one_sided_upper'];verdict='通过数值阈值' if r['adjusted_upper_below_zero'] else '未通过数值阈值' if upper is not None else '不适用'
            lines.append(f'|{name}|{r["mean_difference"]:.10g}|{r["paired_sample_sd"]:.10g}|{lo:.10g}|{hi:.10g}|{upper if upper is not None else "—"}|{verdict}|{r["improved_tasks"]}/{r["equal_tasks"]}/{r["worse_tasks"]}|')
        lines.append('')
    resources=ss['resource_notes'];cheap=[r for r in methods if r['method']!=primary and r['actual_mean_time_ratio_to_primary']<1]
    lines.extend(['## 4．资源，不把更准说成更快','',f'本批有{len(cheap)}个控制的完整平均调用比候选便宜。是否需要给这些强控制更多起点、不同求解器或更充分适应预算，要另行论证，不能仅凭通过质量阈值宣布整个目标完成。',
        f'完整调用总秒{resources["successful_and_fallback_calls_seconds"]:.7g}；完成任务的含I/O墙钟总秒{resources["completed_task_wall_including_io_seconds"]:.7g}；各启动模型加载共{resources["model_loading_seconds_all_invocations"]:.7g}秒。',
        f'未提交尝试的已知额外调用{resources["extra_interrupted_completed_calls"]}次、{resources["extra_known_interrupted_seconds"]:.7g}秒；缺少完成记录的在途调用{resources["unresolved_started_calls"]}次，费用若未知就保持未知。','',
        '正式预测期间本机曾出现物理内存压力和换页，且有阶段快照的Git整理/打包/上传并发。全部原任务和费用保留；当前计时不声称独占环境或无干扰，不删除慢任务、不选择性复测。见[运行期资源观察](../../../outputs/ttt-pc-alm-research/287_runtime_resource_observation_20260924.md)与[发布范围](../../../release/SCOPE.md)。','',
        '以下是被记录的数值数组小计，不是峰值原生内存，不可盲目相加；临时工作区、Python对象及共享模型的范围各异。旧内存仪器测量另见285资源记录。','',
        '|方法|状态字段|记录任务数|平均bytes|最大bytes|','|---|---|---:|---:|---:|'])
    for r in methods:
        for key,value in r['named_numeric_state_subtotals'].items():lines.append(f'|{r["method"]}|{key}|{value["observed_rows"]}|{value["mean_bytes"]:.7g}|{value["maximum_bytes"]}|')
    lines.extend(['','## 5．数学与研究边界','',
        '两个无正则线性头在理想连续四样本带噪任务下，查询MSE均值有限而任务方差可无限。有限数组当然仍有有限经验方差；这不是已观测到异常点，也不能据此删任务。常规有限方差的bootstrap论证不能直接套用；重抽次数和Bonferroni公式并不自动保证覆盖率。详见[矩边界推导](../../../outputs/ttt-pc-alm-research/287_ols_moment_addendum.md)。',
        '本次不修改冻结方法或判据，不以无正则LS的尾部作为PC-ALM贡献。岭/RLS、核、强先验与元学习特征、浅头及同参数BP/PC/ALM全部在本页保留。',
        '元岭回归的独立核验保留原预测、原1e-9绝对/1e-10相对容差；双精度参考出现超差时，使用预先验证的70/110位精度参考复核，离线核验不改变在线模型或费用。完整数值诊断见[参考算术说明](../../../outputs/ttt-pc-alm-research/287_head_reference_diagnosis.md)。',
        '路径审计v4明确修正了接受逻辑：保留真实几何与误差证书，将原1e-12严格施加于最终混合均值；区域内同阈值仅作诊断，不能声称各区域单独通过。原失败和新查询开放前的[数学说明](../../../outputs/ttt-pc-alm-research/287_pool_budget_scope_correction.md)均保留。此项不改变预测、方法、资源或质量判据，也不把名义均值界当有限粒子的逐样本误差。',
        '普通PC/无乘子/强BP共享试探池的对照用于隔离续接信用；普通ALM控制用于比较整个分支机制。共同几何与多解释读出是全局计算，不能把整个系统称为只有局部运算。此方法不运行全链BP或用BP信用初始化候选，但“无BP”本身不是贡献。',
        '这仍是四层可控任务，尚无官方TTT同任务、一般MLP或LLM/VLM桥接验证。若优势可由简单回归或其它更简单方法解释，应保留该解释，不强留两篇论文组合。','',
        '## 6．核验链接','',f'[评价记录](../{inp.name}/summary.json)；[独立风险与重抽核验](../{audit.name}/summary.json)；[原协议](../../../outputs/ttt-pc-alm-research/287_probe_credit_confirmation_protocol.md)。',
        '图片生成不能代替视检；图像布局应在交付前另行打开检查。以上任何passed字段都不等同核心研究目标完成。',''])
    with (out/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    ans=dict(passed=True,stage=args.stage,functional_preflight_only=preflight,tasks=n,method_tables=27,contrast_rows=104,all_methods_retained=True,
        visual_qa_required=True,core_research_goal_complete=False,outputs_sha256={name:sha(out/name) for name in ['protocol.json','287_all_methods.png','287_all_contrasts.png','report.md']})
    exclusive_json(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
