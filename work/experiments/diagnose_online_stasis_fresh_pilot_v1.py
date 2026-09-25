"""307 sealed-result attribution and all45 task influence; never refits a model.

Descriptive follow-up only. Requires completed prediction, score and independent
audit. The whole-task frozen comparisons remain the primary evidence.
"""
import argparse
from collections import Counter
import math
from pathlib import Path
import time
import numpy as np
from diagnose_gradient_flat_split_states_v1 import read,save,sha

PRIMARY='online_stasis_alm_keep64'
BASELINE='credit_control_probe33'
FIXED_ARRAYS=['initial_b','initial_h','initial_u','initial_best','origins',
    'assigned_actions','atomic_trial_b','atomic_trial_origins','effective_initial_u',
    'selected_b','best_bank','point_prediction']


def identical(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def scalar_truth(seed,q):
    biases=np.random.default_rng(seed).uniform(-.12,.12,4)
    result=[]
    for x in q:
        value=float(x)
        for b in biases:
            z=value+float(b)
            value=max(0.,min(2.*z,2.-2.*z))
        result.append(value)
    return np.array(result)


def risk_terms(new,old,truth):
    delta=new-old
    movement=math.fsum(float(x)**2 for x in delta)/len(delta)
    alignment=2*math.fsum(float(d)*(float(g)-float(y)) for d,g,y in zip(delta,old,truth))/len(delta)
    return movement,alignment


def influence(values):
    a=[float(x) for x in values];n=len(a);assert n>1
    total=math.fsum(a)
    without=[math.fsum(a[:i]+a[i+1:])/(n-1) for i in range(n)]
    absolute_total=math.fsum(abs(x) for x in a)
    return dict(tasks=n,mean_difference=total/n,leave_one_out_min=min(without),
        leave_one_out_max=max(without),leave_one_out_sign_reversals=sum(x*total<0 for x in without),
        maximum_single_absolute_contribution_fraction=(max(map(abs,a))/absolute_total if absolute_total else None),
        improved=sum(x<0 for x in a),equal=sum(x==0 for x in a),worse=sum(x>0 for x in a),
        all_tasks_retained=True,not_an_additional_hypothesis_test=True)


def selftest():
    a=np.array([.25,.75]);b=np.array([.75,.25]);truth=np.array([0.,1.])
    assert risk_terms(a,b,truth)==(.25,-.75)
    assert risk_terms(a,a,truth)==(0.,0.)
    zero=influence([0.,0.,0.])
    assert zero['mean_difference']==0 and zero['maximum_single_absolute_contribution_fraction'] is None
    assert zero['leave_one_out_sign_reversals']==0
    fragile=influence([-3.,1.,1.])
    assert fragile['mean_difference']==-1/3 and fragile['leave_one_out_min']==-1.
    assert fragile['leave_one_out_max']==1. and fragile['leave_one_out_sign_reversals']==1
    assert fragile['maximum_single_absolute_contribution_fraction']==.6
    assert identical(a,a.copy()) and not identical(a,a.reshape(1,2))
    return dict(known_risk_cases=2,known_influence_cases=2,byte_shape_checks=2)


def run(root,out):
    start=time.perf_counter();selftests=selftest();base=root/'results/online_stasis_fresh_pilot'
    pred=base/'pilot_predictions_v1';evaluation=base/'pilot_evaluation_v1';audited=base/'audit_v1'
    ps=read(pred/'summary.json');es=read(evaluation/'summary.json');audit=read(audited/'summary.json')
    assert ps['passed'] and es['passed'] and audit['passed']
    assert ps['tasks']==es['tasks']==256 and ps['predictors']==es['predictors']==11776
    assert audit['counts']['predictors']==11776 and audit['counts']['comparisons']==180
    assert audit['prediction_summary_sha256']==sha(pred/'summary.json')
    assert audit['evaluation_summary_sha256']==sha(evaluation/'summary.json')
    for folder,summary in [(pred,ps),(evaluation,es)]:
        for name,digest in summary['outputs_sha256'].items():assert sha(folder/name)==digest
    p=read(pred/'protocol.json');names=p['methods'];seeds=p['seeds']
    assert seeds==list(range(307000000,307000256)) and len(names)==46 and names[37]==PRIMARY
    online=names[37:];assert len(online)==9
    rows=read(pred/'rows.json');lookup={(r['seed'],r['method']):r for r in rows}
    comparisons=read(evaluation/'comparisons.json');comparison={(r['control'],r['metric']):r for r in comparisons}
    with np.load(evaluation/'task_metrics.npz',allow_pickle=False) as z:
        risk=z['risk'];failures=z['failures']
        assert z['seeds'].tolist()==seeds and z['methods'].tolist()==names
    assert risk.shape==(256,46,4) and failures.shape==(256,46)
    pi=names.index(PRIMARY);bi=names.index(BASELINE);counts=Counter()
    protocol=dict(source_sha256={Path(__file__).name:sha(Path(__file__))},primary=PRIMARY,baseline=BASELINE,
        online_methods=online,fixed_arrays=FIXED_ARRAYS,prediction_summary_sha256=sha(pred/'summary.json'),
        evaluation_summary_sha256=sha(evaluation/'summary.json'),audit_summary_sha256=sha(audited/'summary.json'),
        descriptive_only=True,groups_not_used_for_method_selection=True,all45_frozen_main_comparisons_unchanged=True,
        query_targets_accessed=True,no_fitting_or_optimizer_calls=True,core_research_goal_complete=False)
    save(out/'protocol.json',protocol)
    tasks=[];maximum_identity_gap=0.;support_array_checks=0
    for si,seed in enumerate(seeds):
        arrays={};metadata={}
        for name in [BASELINE]+online:
            r=lookup[seed,name];assert sha(pred/r['file'])==r['sha256']
            with np.load(pred/r['file'],allow_pickle=False) as z:arrays[name]={k:z[k] for k in z.files}
            metadata[name]=r['metadata']
        original=arrays[BASELINE];truth=scalar_truth(seed,original['q_observed'])
        for name in online:
            mi=names.index(name);a=arrays[name];m=metadata[name];om=metadata[BASELINE]
            for field in ['x_observed','v_observed','q_observed']:
                assert identical(a[field],original[field]),(seed,name,field)
                support_array_checks+=1
            successful=not (m['execution_failed'] or om['execution_failed'])
            assert bool(failures[si,mi])==m['execution_failed'] and bool(failures[si,bi])==om['execution_failed']
            counts['all_original_pairs']+=1
            counts['successful_original_pairs' if successful else 'original_pairs_with_fallback']+=1
            if successful:
                for field in FIXED_ARRAYS:
                    assert identical(a[field],original[field]),(seed,name,field)
                    counts['fixed_original_arrays']+=1
                assert np.array_equal(risk[si,mi,2:],risk[si,bi,2:])
                assert m['shadow_horizon']==64 and m['shadow_state_steps']==64*m['selected_states']
                assert not m['query_targets_accessed'] and not m['archive_or_reference_access_in_fit']
                assert not m['selected_using_global_bp'] and not m['diagnostic_trace_enabled']
                if not name.startswith('online_stasis_adam'):
                    assert not m['global_bp_used']
                else:
                    assert m['global_bp_used']==(m['selected_states']>0)
                if not metadata[PRIMARY]['execution_failed']:
                    assert m['selected_states']==metadata[PRIMARY]['selected_states']
                    counts['matching_trigger_counts']+=1
                if m['selected_states']==0:
                    assert identical(a['prediction'],original['prediction']),(seed,name,'no-trigger readout')
                    counts['no_trigger_identical_readouts']+=1
            old_modes=set(om.get('positive_modes',[]));new_modes=set(m.get('positive_modes',[]))
            added=sorted(new_modes-old_modes) if successful else None
            missing=sorted(old_modes-new_modes) if successful else None
            same_pool=(new_modes==old_modes) if successful else None
            same_readout=identical(a['prediction'],original['prediction'])
            fields=[]
            for k,stride in enumerate([1,2]):
                movement,alignment=risk_terms(a['prediction'][::stride],original['prediction'][::stride],truth[::stride])
                change=float(risk[si,mi,k]-risk[si,bi,k]);gap=abs(change-movement-alignment)
                maximum_identity_gap=max(maximum_identity_gap,gap);assert gap<2e-12
                fields.append(dict(metric=['mse257','mse129'][k],risk_change=change,
                    movement_squared=movement,twice_realized_error_alignment=alignment,identity_gap=gap))
                counts['risk_identity_checks']+=1
            tasks.append(dict(seed=seed,method=name,successful_pair=successful,selected_states=m.get('selected_states'),
                shadow_state_steps=m.get('shadow_state_steps'),positive_modes_added=added,positive_modes_missing=missing,
                same_positive_pool=same_pool,identical_readout=same_readout,risk_terms=fields,
                charged_seconds=lookup[seed,name]['seconds'],baseline_charged_seconds=lookup[seed,BASELINE]['seconds']))
        if (si+1)%32==0:print(dict(phase='sealed_result_attribution',tasks=si+1,total=256),flush=True)
    assert counts['all_original_pairs']==2304 and counts['risk_identity_checks']==4608
    # All45 primary effects are retained. Leave-one-out values are sensitivity
    # descriptions, not a rule to delete a task or modify the fixed estimand.
    influences=[]
    for mi,name in enumerate(names):
        if name==PRIMARY:continue
        row=dict(control=name,**influence(risk[:,pi,0]-risk[:,mi,0]))
        reported=comparison[name,'mse257']
        assert abs(row['mean_difference']-reported['mean_difference'])<2e-12
        assert [row[k] for k in ['improved','equal','worse']]==[reported[k] for k in ['improved','equal','worse']]
        influences.append(row)
    groups=[]
    predicates=[('all_tasks',lambda r:True),
        ('successful_no_trigger',lambda r:r['successful_pair'] and r['selected_states']==0),
        ('successful_trigger',lambda r:r['successful_pair'] and r['selected_states']>0),
        ('successful_added_pool',lambda r:r['successful_pair'] and bool(r['positive_modes_added'])),
        ('successful_same_pool',lambda r:r['successful_pair'] and r['same_positive_pool']),
        ('successful_missing_old_pool',lambda r:r['successful_pair'] and bool(r['positive_modes_missing'])),
        ('fallback_pairs',lambda r:not r['successful_pair'])]
    for name in online:
        for label,predicate in predicates:
            selected=[r for r in tasks if r['method']==name and predicate(r)]
            risks=[r['risk_terms'][0] for r in selected]
            groups.append(dict(method=name,group=label,tasks=len(selected),groups_may_overlap=True,
                mean_risk_change=math.fsum(r['risk_change'] for r in risks)/len(risks) if risks else None,
                mean_movement_squared=math.fsum(r['movement_squared'] for r in risks)/len(risks) if risks else None,
                mean_alignment=math.fsum(r['twice_realized_error_alignment'] for r in risks)/len(risks) if risks else None,
                improved=sum(r['risk_change']<0 for r in risks),equal=sum(r['risk_change']==0 for r in risks),
                worse=sum(r['risk_change']>0 for r in risks)))
    partitions=[]
    for name in online:
        three=[next(r for r in groups if r['method']==name and r['group']==label)
               for label in ['successful_no_trigger','successful_trigger','fallback_pairs']]
        assert sum(r['tasks'] for r in three)==256
        total=math.fsum(r['tasks']*(r['mean_risk_change'] or 0.) for r in three)/256
        all_row=next(r for r in groups if r['method']==name and r['group']=='all_tasks')
        assert abs(total-all_row['mean_risk_change'])<2e-12
        assert three[0]['mean_risk_change'] in [None,0.]
        partitions.append(dict(method=name,all_task_mean=all_row['mean_risk_change'],
            no_trigger_tasks=three[0]['tasks'],trigger_tasks=three[1]['tasks'],fallback_tasks=three[2]['tasks'],
            trigger_contribution=three[1]['tasks']*(three[1]['mean_risk_change'] or 0.)/256,
            fallback_contribution=three[2]['tasks']*(three[2]['mean_risk_change'] or 0.)/256,
            identity_gap=abs(total-all_row['mean_risk_change'])))
    missing_rows=[r for r in tasks if r['successful_pair'] and r['positive_modes_missing']]
    same_pool_changed=[r for r in tasks if r['successful_pair'] and r['same_positive_pool'] and not r['identical_readout']]
    lines=['# 307：原路径核对与新任务收益归因','',
        '在全量预测、评分及独立审计完成后执行。仅描述已冻结的256个任务，不改方法、不删除任务、不另设显著性检验。查询真值只用于事后风险解释。','',
        f'九个在线方法与原probe-ALM共{counts["all_original_pairs"]}组配对；核对{counts["fixed_original_arrays"]}个原始状态/原子操作/保留点数组逐位一致。',
        f'其中双方正常执行{counts["successful_original_pairs"]}组，包含数值回退{counts["original_pairs_with_fallback"]}组；回退仍计入原主比较和下表all_tasks。',
        f'成功配对中缺失原正体积分支的有{len(missing_rows)}组；同正体积分支集合但预测非逐位相同的有{len(same_pool_changed)}组，均保存逐任务明细，不能自动归因于发现了新分支。','',
        '实际程序关闭了完整轨迹；这里只核对保存的原始数组与触发数，并依据冻结代码、旧全轨迹前检确认结构，不冒称逐位检查了新任务的未保存轨迹或全部触发位置。','',
        '## 逐任务风险分解','',
        '令δ=在线方法均值预测−原probe-ALM均值预测，精确展开ΔMSE=E_q[δ²]+2E_q[δ(原预测−真值)]。第一项是移动代价，第二项是相对这次真实任务误差的方向项；方向项必须足够负才能产生收益。它不是可供在线方法读取的收益标签，也不是后验质量覆盖证明。',
        f'257/129点共{counts["risk_identity_checks"]}个恒等式已核对，最大数值差{maximum_identity_gap:.12g}。','',
        '|在线方法|描述性组（可能重叠）|任务数|均值风险变化|移动项|方向项|改善/相同/更差|',
        '|---|---|---:|---:|---:|---:|---|']
    def fmt(value):return '—' if value is None else f'{value:.12g}'
    for r in groups:
        lines.append(f'|{r["method"]}|{r["group"]}|{r["tasks"]}|{fmt(r["mean_risk_change"])}|{fmt(r["mean_movement_squared"])}|{fmt(r["mean_alignment"])}|{r["improved"]}/{r["equal"]}/{r["worse"]}|')
    lines+=['','正常无触发、正常有触发和包含回退三组构成不重叠的全任务分解；已核对每方法组任务数之和为256，正常无触发组风险差为零。各组对全任务均值的加权贡献在trigger_partition.json中完整保存。','',
        '## 全部45个主比较的单任务影响','',
        '每次暂去一个任务重新计算均值，仅用于检查符号和集中度的敏感性；没有从正式统计中删除任何任务，也不能据此换种子或追加样本追求显著。绝对贡献份额的分母是所有任务差值绝对值之和，不是净收益。','',
        '|控制|全量候选−控制|逐一删除后最小均值|逐一删除后最大均值|符号翻转次数|最大单任务绝对贡献份额|改善/相同/更差|',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in influences:
        lines.append(f'|{r["control"]}|{fmt(r["mean_difference"])}|{fmt(r["leave_one_out_min"])}|{fmt(r["leave_one_out_max"])}|{r["leave_one_out_sign_reversals"]}|{fmt(r["maximum_single_absolute_contribution_fraction"])}|{r["improved"]}/{r["equal"]}/{r["worse"]}|')
    lines+=['','参数点未变而池读出改变，只能定位收益发生在分支发现和读出这条路径。乘子本身是否提供独立价值，仍必须看原报告中同触发清零、残差方向、随机符号、普通PC和强Adam的全任务比较及完整成本；这些描述性分组不替代它们。','',
        f'[完整主报告]({(base/"report_v1/report.md").as_posix()})；[全部逐任务明细]({(out/"tasks.json").as_posix()})。','']
    with (out/'report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    for name,value in [('tasks.json',tasks),('groups.json',groups),('trigger_partition.json',partitions),('all_primary_influence.json',influences)]:save(out/name,value)
    result=dict(passed=True,selftests=selftests,counts=dict(counts),support_array_checks=support_array_checks,
        maximum_identity_gap=maximum_identity_gap,all_primary_influence_comparisons=len(influences),
        original_positive_pool_missing_pairs=len(missing_rows),same_pool_nonidentical_readout_pairs=len(same_pool_changed),
        saved_arrays_not_full_trace=True,descriptive_only=True,all_tasks_retained=True,
        query_targets_accessed=True,core_research_goal_complete=False,seconds=time.perf_counter()-start,
        trigger_partition_checks=len(partitions),
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','tasks.json','groups.json','trigger_partition.json','all_primary_influence.json','report.md']})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();out=root/'results/online_stasis_fresh_pilot/diagnosis_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise
