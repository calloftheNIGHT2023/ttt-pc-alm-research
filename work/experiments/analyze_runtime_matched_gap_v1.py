"""411 descriptive decomposition of already published 405 predictions.

No new targets, counterfactual completion, or fair-budget superiority claim.
Packet labels alone cannot distinguish a prior fallback from a useful stage.
"""
from pathlib import Path
import math
import time
import traceback
import numpy as np
import deadline_risk_io_v1 as io

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/runtime_matched_prefix'
OUT=BASE/'gap_diagnostic_v1'
ENTRY='outputs/ttt-pc-alm-research/411_runtime_matched_gap_diagnostic_v1.md'
PRIMARY='regional_active512_dfs_farthest_x'
CONTROL='adam_gn64_256_anytime'


def main():
    started=time.perf_counter()
    ss=io.read(BASE/'evaluation_v1/summary.json');assert ss['passed']
    for name,digest in ss['outputs_sha256'].items():assert io.sha(BASE/'evaluation_v1'/name)==digest
    source_hash=io.sha(Path(__file__))
    rows=io.read(BASE/'evaluation_v1/risk_rows.json')
    picked={(r['method'],r['n'],r['seed']):r for r in rows if r['budget']==.5}
    seals={};cache={}
    def prediction(row):
        folder=ROOT/row['directory']
        for name in ['outputs.npz','events.json']:
            path=folder/name;digest=io.sha(path);assert digest==row['files'][name]
            seals[path.relative_to(ROOT).as_posix()]=digest
        if row['directory'] not in cache:
            with np.load(folder/'outputs.npz',allow_pickle=False) as z:cache[row['directory']]=z['prediction'].copy()
        return cache[row['directory']]
    pairs=[];decomposition=[]
    for n in [4,8,16,24]:
        current=[]
        for seed in range(5920000,5920004):
            a,b=picked[PRIMARY,n,seed],picked[CONTROL,n,seed]
            ap,bp=prediction(a),prediction(b)
            ae=io.read(ROOT/a['directory']/'events.json');be=io.read(ROOT/b['directory']/'events.json')
            row=dict(n=n,seed=seed,primary_mse=a['query_mse'],control_mse=b['query_mse'],
                risk_difference=a['query_mse']-b['query_mse'],prediction_bitwise_equal=bool(np.array_equal(ap,bp)),
                primary_selected=a['selected'],control_selected=b['selected'],
                primary_event_count=len(ae),control_event_count=len(be))
            if a['selected']=='fallback':
                prior=picked['prior4096_ridge',n,seed]
                row['primary_equals_prior4096']=bool(np.array_equal(ap,prediction(prior)))
                row['primary_had_only_initial_packet']=len(ae)==1 and ae[0]['kind']=='fallback'
            pairs.append(row);current.append(row)
        delta=math.fsum(r['risk_difference'] for r in current)/4
        decomposition.append(dict(n=n,primary_mean=math.fsum(r['primary_mse'] for r in current)/4,
            control_mean=math.fsum(r['control_mse'] for r in current)/4,difference=delta,
            contribution_to_four_prefix_gap=delta/4,
            exact_prediction_pairs=sum(r['prediction_bitwise_equal'] for r in current),
            primary_final=sum(r['primary_selected']=='final' for r in current),
            primary_initial_prior_only=sum(r.get('primary_equals_prior4096',False) and r.get('primary_had_only_initial_packet',False) for r in current)))
    gap=math.fsum(r['risk_difference'] for r in pairs)/16
    assert abs(gap-math.fsum(r['contribution_to_four_prefix_gap'] for r in decomposition))<1e-16
    means=io.read(BASE/'evaluation_v1/prefix_averages.json')
    mm={r['method']:r['mean_query_mse'] for r in means if r['budget']==.5}
    assert abs(gap-(mm[PRIMARY]-mm[CONTROL]))<1e-16
    io.save(OUT/'pairs.json',pairs);io.save(OUT/'decomposition.json',decomposition)
    io.save(OUT/'inputs_manifest.json',seals)
    lines=['# 411｜主预算风险差的逐前缀诊断','',
        '这是405已经评估／发布的四旧任务上的描述性分析，不是新的预注册预测或独立确认。没有重新生成查询答案、修改预测、代入假想完成结果或改变主预算。','',
        '比较本轮主active512 DFS与强BP组合Adam/GN 64+256起点，预算固定0.5秒。差值为候选减控制；四前缀等权平均。','',
        '| 支持数 | 候选MSE | 组合MSE | 对总差的贡献 | 逐查询预测完全相同的任务数 | 候选仅交付初始先验备用 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in decomposition:
        lines.append(f"| {r['n']} | {r['primary_mean']:.12g} | {r['control_mean']:.12g} | {r['contribution_to_four_prefix_gap']:.12g} | {r['exact_prediction_pairs']}/4 | {r['primary_initial_prior_only']}/4 |")
    lines+=['',f'四前缀总差为 **{gap:.12g}**。表中逐前缀贡献的和已经与原评估均值差独立核对。','',
        '备用状态不能一概解释为无进展：强组合的阶段性预测也使用fallback标签。诊断同时检查真实事件与预测数组，只有候选确实仅有初始包且与同支持prior4096逐元素相等时，才记作“仅初始备用”。','',
        '因此下一条可检验正向假设是：在任务尚未整体完成时，以付费的部分证书／中间读出将已有信息交付给查询，可能减少这部分截止时间损失。当前归因只到交付边界；超时状态未完整归档，不能断言三个任务内部已找到足够信息，更不能把假想补全当作实际改进。','',
        '400提供条件性风险非增的数学接口；下一步先用可观察支持测内质量、总质量上界、区间宽度、排除基线比例与实际成本。并保留无需搜索的Lipschitz支持包络、BP／PC／PDHG共享证书读出；不把通用投影效应归为PC-ALM独占。','',
        '[逐任务证据](../../results/runtime_matched_prefix/gap_diagnostic_v1/pairs.json) · [分解数据](../../results/runtime_matched_prefix/gap_diagnostic_v1/decomposition.json) · [完整405报告](../../results/runtime_matched_prefix/report_v1/report.md)']
    (ROOT/ENTRY).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert io.sha(Path(__file__))==source_hash
    io.save(OUT/'summary.json',dict(passed=True,old_task_pairs=len(pairs),budget=.5,total_gap=gap,
        prediction_exact_pairs=sum(r['prediction_bitwise_equal'] for r in pairs),
        primary_initial_prior_only=sum(r.get('primary_equals_prior4096',False) and r.get('primary_had_only_initial_packet',False) for r in pairs),
        new_query_targets_generated=False,post_hoc_development_diagnostic=True,
        independent_confirmation=False,counterfactual_predictions_used=False,seconds=time.perf_counter()-started,
        source_sha256={Path(__file__).relative_to(ROOT).as_posix():source_hash},
        evaluation_summary_sha256=io.sha(BASE/'evaluation_v1/summary.json'),
        entry_file=ENTRY,entry_sha256=io.sha(ROOT/ENTRY),
        outputs_sha256={f:io.sha(OUT/f) for f in ['pairs.json','decomposition.json','inputs_manifest.json']}))
    print(io.read(OUT/'summary.json'),flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    try:main()
    except Exception:
        io.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
