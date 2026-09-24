"""294 post-evaluation attribution: changed mode pool, fixed point path, risk terms.

Descriptive diagnosis only. Never changes the fixed whole-task comparisons.
"""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
import counterfactual_fresh_pilot_v1 as pilot
from diagnose_gradient_flat_split_states_v1 import read,sha


def run(root,out,stage):
    start=time.perf_counter();base=out.parent;pred=base/(stage+'_predictions_v1');score=base/(stage+'_evaluation_v1')
    ps=pilot.suite.complete(pred);es=pilot.suite.complete(score);p=read(pred/'protocol.json')
    assert p['source_sha256']==pilot.gate(root)==read(score/'protocol.json')['source_sha256']
    assert read(score/'protocol.json')['prediction_summary_sha256']==sha(pred/'summary.json')
    assert ps['tasks']==es['tasks']==(2 if stage=='preflight' else 128)
    assert es['independent_scalar_risk_passed'] and es['all_bootstrap_means_checked']==2880000
    rows=read(pred/'rows.json');lookup={(r['seed'],r['method']):r for r in rows}
    with np.load(score/'task_metrics.npz',allow_pickle=False) as z:
        assert z['seeds'].tolist()==p['seeds']
        names=z['methods'].tolist();risk=z['risk'];failures=z['failures']
    primary=pilot.suite.PRIMARY;old='credit_control_probe33';pi=names.index(primary);oi=names.index(old)
    counts=Counter();tasks=[];maximum_identity_gap=0.
    protocol=dict(stage=stage,source_sha256=sha(Path(__file__)),prediction_summary_sha256=sha(pred/'summary.json'),
        evaluation_summary_sha256=sha(score/'summary.json'),primary=primary,control=old,
        post_hoc_descriptive_groups=True,fixed_all36_comparisons_unchanged=True,
        claim='Paired fixed-point checks and exact realized squared-error decomposition; not posterior-mass causal proof',
        query_targets_accessed=True,core_research_goal_complete=False)
    pilot.exclusive(out/'protocol.json',protocol)
    for si,seed in enumerate(p['seeds']):
        newrow,oldrow=lookup[seed,primary],lookup[seed,old]
        arrays=[]
        for row in [newrow,oldrow]:
            assert sha(pred/row['file'])==row['sha256']
            with np.load(pred/row['file'],allow_pickle=False) as z:arrays.append({k:z[k] for k in z.files})
        new,previous=arrays
        for k in ['x_observed','v_observed','q_observed']:
            assert new[k].tobytes()==previous[k].tobytes()
        successful=not newrow['metadata']['execution_failed'] and not oldrow['metadata']['execution_failed']
        counts['all_pairs']+=1;counts['successful_pairs' if successful else 'pairs_with_fallback']+=1
        old_modes=set(oldrow['metadata'].get('positive_modes',[]))
        new_modes=set(newrow['metadata'].get('positive_modes',[]))
        missing=sorted(old_modes-new_modes);added=sorted(new_modes-old_modes)
        if successful:
            for key in ['selected_b','best_bank','point_prediction']:
                assert new[key].dtype==previous[key].dtype and new[key].shape==previous[key].shape
                assert new[key].tobytes()==previous[key].tobytes(),(seed,key)
                counts['fixed_point_arrays']+=1
            assert not (risk[si,pi,2:]-risk[si,oi,2:]).any()
            counts['old_positive_subset']+=int(not missing)
            counts['tasks_with_added_modes']+=bool(added)
            counts['added_modes']+=len(added)
            counts['missing_old_modes']+=len(missing)
        delta=new['prediction']-previous['prediction']
        truth=pilot.scalar_truth(seed,new['q_observed'])
        movement=float(np.mean(delta**2))
        alignment=float(2*np.mean(delta*(previous['prediction']-truth)))
        change=float(risk[si,pi,0]-risk[si,oi,0]);gap=abs(change-(movement+alignment))
        maximum_identity_gap=max(maximum_identity_gap,gap);assert gap<2e-12
        counts['identical_mean_readouts']+=int(new['prediction'].tobytes()==previous['prediction'].tobytes())
        tasks.append(dict(seed=seed,successful_pair=successful,new_positive_modes=added,missing_old_positive_modes=missing,
            same_positive_pool=new_modes==old_modes,identical_readout=new['prediction'].tobytes()==previous['prediction'].tobytes(),
            risk_change=change,mean_prediction_movement_squared=movement,twice_alignment_with_realized_old_error=alignment,
            identity_gap=gap,counterfactual_steps=newrow['metadata'].get('counterfactual_state_steps'),
            new_mse=float(risk[si,pi,0]),old_mse=float(risk[si,oi,0]),
            new_seconds=newrow['seconds'],old_seconds=oldrow['seconds']))
    assert counts['all_pairs']==ps['tasks'] and int(failures[:,[pi,oi]].any(1).sum())==counts['pairs_with_fallback']
    groups=[]
    predicates=[('all_tasks',lambda r:True),('successful_added_positive_modes',lambda r:r['successful_pair'] and bool(r['new_positive_modes'])),
        ('successful_same_positive_pool',lambda r:r['successful_pair'] and r['same_positive_pool']),
        ('successful_missing_old_modes',lambda r:r['successful_pair'] and bool(r['missing_old_positive_modes'])),
        ('pairs_with_fallback',lambda r:not r['successful_pair'])]
    for name,predicate in predicates:
        rr=[r for r in tasks if predicate(r)]
        groups.append(dict(group=name,tasks=len(rr),not_disjoint_partition=True,
            mean_risk_change=float(np.mean([r['risk_change'] for r in rr])) if rr else None,
            mean_movement=float(np.mean([r['mean_prediction_movement_squared'] for r in rr])) if rr else None,
            mean_alignment=float(np.mean([r['twice_alignment_with_realized_old_error'] for r in rr])) if rr else None,
            improved=sum(r['risk_change']<0 for r in rr),equal=sum(r['risk_change']==0 for r in rr),worse=sum(r['risk_change']>0 for r in rr)))
    for name,value in [('tasks.json',tasks),('groups.json',groups)]:pilot.exclusive(out/name,value)
    lines=['# 294｜新提议相对原 ALM 的机制归因','',
        '以下分析在完整评分封存后进行，分组只作事后描述，不替代全 128 任务的主比较，也不改变方法、预算或统计判据。' if stage=='pilot' else '旧两个任务功能前检，不是新任务科学结论。','',
        f'完整配对数 {counts["all_pairs"]}，双方无回退的配对数 {counts["successful_pairs"]}；检查 {counts["fixed_point_arrays"]} 个支持最优点/参数存档/点预测数组逐位一致。',
        f'成功配对中，{counts["tasks_with_added_modes"]} 个任务新增 {counts["added_modes"]} 个正体积分支；原分支缺失总数 {counts["missing_old_modes"]}。均值读出逐位相同的任务数 {counts["identical_mean_readouts"]}。','',
        '以 δ=新均值预测−原均值预测，逐任务核对 ΔMSE = E_q[δ²] + 2E_q[δ(原预测−查询真值)]。前项是移动幅度，后项是相对本次真实误差的方向项；后项为负且幅度足够才能得到实际查询收益。这不是知道真值后选择在线动作，真值仅用于完整实验结束后的解释。','',
        '|描述性组（可能重叠）|任务数|平均风险变化|平均移动项|平均方向项|改善/相同/更差|',
        '|---|---:|---:|---:|---:|---|']
    for r in groups:
        fmt=lambda v:'—' if v is None else f'{v:.9g}'
        lines.append(f'|{r["group"]}|{r["tasks"]}|{fmt(r["mean_risk_change"])}|{fmt(r["mean_movement"])}|'
            f'{fmt(r["mean_alignment"])}|{r["improved"]}/{r["equal"]}/{r["worse"]}|')
    lines+=['',f'平方误差恒等式最大数值差 {maximum_identity_gap:.9g}。',
        '这些分组没有新的显著性主张；有限粒子重分配与几何数值误差仍可能参与均值改变。没有完整后验质量积分，因此不能由分支个数推断覆盖质量或后验均值误差。',
        '双方发生回退的任务仍包含在原主比较与上面的 all_tasks 中；只从要求同一正常 ALM 路径的逐位机制检查中单列。']
    with (out/'report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    result=dict(passed=True,stage=stage,counts=dict(counts),maximum_identity_gap=maximum_identity_gap,
        evaluation_summary_sha256=sha(score/'summary.json'),post_hoc_descriptive_only=True,
        query_targets_accessed=True,core_research_goal_complete=False,seconds=time.perf_counter()-start,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','tasks.json','groups.json','report.md']})
    pilot.exclusive(out/'summary.json',result);print(result,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--stage',choices=['preflight','pilot'],required=True);args=p.parse_args();root=args.project.resolve()
    out=root/'results/counterfactual_fresh_pilot'/f'{args.stage}_diagnosis_v1';out.mkdir(parents=True,exist_ok=False)
    try:run(root,out,args.stage)
    except BaseException as exc:
        pilot.exclusive(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()
