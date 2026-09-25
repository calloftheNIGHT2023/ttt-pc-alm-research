"""414 archived-search component diagnostic; no query truth or deadline claim."""
from pathlib import Path
from fractions import Fraction as F
import math
import time
import traceback
import numpy as np
import frontier_range_certificate_v1 as model
import partial_support_certificate_v1 as old_model
import run_partial_support_certificate_v1 as previous
import candidate_set_readout_v1 as serial
import deadline_risk_io_v1 as io

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/frontier_range_certificate'
SOURCE=ROOT/'results/partial_support_certificate/development_v1'
ENTRY='outputs/ttt-pc-alm-research/415_support_certificate_results_v1.md'
DESIGN='outputs/ttt-pc-alm-research/414_frontier_range_certificate_protocol_v1.md'


def main(out):
    started=time.perf_counter()
    pre=io.read(BASE/'preflight_v1/summary.json');assert pre['passed'];io.verify_hashes(ROOT,pre['source_sha256'])
    audit=io.read(SOURCE.parent/'audit_v1/summary.json');assert audit['passed']
    assert audit['development_summary_sha256']==io.sha(SOURCE/'summary.json')
    ss=io.read(SOURCE/'summary.json');assert ss['passed']
    for path,digest in ss['outputs_sha256'].items():assert io.sha(SOURCE/path)==digest
    pp=io.read(SOURCE/'protocol.json');io.verify_hashes(ROOT,pp['source_sha256'])
    hashes={p.relative_to(ROOT).as_posix():io.sha(p) for p in sorted((ROOT/'work/experiments').glob('*.py'))+[ROOT/DESIGN]}
    io.save(out/'protocol.json',dict(source_sha256=hashes,source_summary_sha256=io.sha(SOURCE/'summary.json'),
        source_audit_sha256=io.sha(SOURCE.parent/'audit_v1/summary.json'),preflight_sha256=io.sha(BASE/'preflight_v1/summary.json'),
        contexts=16,queries=33,archived_search_used=True,hard_end_to_end_budget=False,
        query_targets_accessed=False,paired_with_mass_diagnostic_only=True))
    rows=[]
    for old in io.read(SOURCE/'rows.json'):
        previous_dir=ROOT/old['directory']
        for name,digest in old['files'].items():assert io.sha(previous_dir/name)==digest
        sa=io.load_arrays(previous_dir/'search_arrays.npz');sm=io.read(previous_dir/'search_metadata.json')
        ra=io.load_arrays(previous_dir/'certificate_arrays.npz')
        directory=out/f"{old['seed']}_n{old['n']}";directory.mkdir()
        begin=time.perf_counter();result=model.fit(sa,sm,ra['q']);seconds=time.perf_counter()-begin
        previous.fraction_save(directory/'certificate.json',result)
        refs=io.read(previous_dir/'baseline_references.json');metrics={};predictions={}
        for name in previous.BASELINES:
            matches=[p for p in refs if f'_{name}_b500/' in p];assert len(matches)==1
            path=ROOT/matches[0];assert io.sha(path)==refs[matches[0]]
            bp=io.load_arrays(path)['prediction'][::8]
            metrics[name]=dict(frontier=previous.interval_metrics(result['frontier_intervals'],result['cheap_intervals'],bp),
                intersection=previous.interval_metrics(result['intersected_intervals'],result['cheap_intervals'],bp),
                cheap=previous.interval_metrics(result['cheap_intervals'],result['cheap_intervals'],bp))
            predictions[name]=np.array([float(old_model.mathcore.project(F(float(a)),bounds)) for a,bounds in zip(bp,result['intersected_intervals'])])
        io.save(directory/'metrics.json',metrics);np.savez_compressed(directory/'projected_predictions.npz',q=ra['q'],**predictions)
        # Component replay covers exact coordinate boxes and every interval;
        # it is reported as a replay, not a second independent implementation.
        check=model.fit(sa,sm,ra['q'])
        for key in ['coordinate_boxes','empty_cells','cell_query_ranges','frontier_intervals','cheap_intervals','intersected_intervals']:
            assert serial.serializable(check[key])==serial.serializable(result[key])
        row=dict(seed=old['seed'],n=old['n'],source_directory=old['directory'],directory=directory.relative_to(ROOT).as_posix(),
            source_search_completed=old['search_completed'],source_full_modes=old['full_modes'],
            cover_cells=len(result['coordinate_boxes']),empty_cells=len(result['empty_cells']),
            mean_mass_interval_width=old['mean_posterior_width'],
            mean_frontier_width=float(sum(b-a for a,b in result['frontier_intervals'])/len(ra['q'])),
            mean_cheap_width=old['mean_cheap_width'],
            mean_intersection_width=float(sum(b-a for a,b in result['intersected_intervals'])/len(ra['q'])),
            source_search_seconds=old['search_seconds'],range_component_seconds=seconds,
            metrics=metrics,files={p.name:io.sha(p) for p in directory.iterdir() if p.is_file()})
        rows.append(row);io.save(out/'rows_partial.json',rows)
        print({k:row[k] for k in ['seed','n','source_full_modes','cover_cells','mean_frontier_width','mean_cheap_width','mean_intersection_width','range_component_seconds']},flush=True)
    io.save(out/'rows.json',rows);io.verify_hashes(ROOT,hashes)
    aggregates={}
    for name in previous.BASELINES:
        aggregates[name]=dict(additional_queries_beyond_cheap=sum(r['metrics'][name]['intersection']['additional_changes_beyond_cheap'] for r in rows),
            total_queries=16*33,contexts_with_additional_changes=sum(r['metrics'][name]['intersection']['additional_changes_beyond_cheap']>0 for r in rows))
    n24=[r for r in rows if r['n']==24]
    summary=dict(passed=True,contexts=16,replayed_contexts=16,exact_replayed_query_intervals=16*33,
        n24_mean_interval_width=math.fsum(r['mean_intersection_width'] for r in n24)/4,
        n24_mean_cheap_width=math.fsum(r['mean_cheap_width'] for r in n24)/4,
        n24_mean_component_seconds=math.fsum(r['range_component_seconds'] for r in n24)/4,
        baseline_aggregates=aggregates,query_targets_accessed=False,query_risk_compared=False,
        archived_search_used=True,matched_budget_comparison=False,pc_alm_independent_advantage_established=False,
        seconds=time.perf_counter()-started,outputs_sha256={f:io.sha(out/f) for f in ['protocol.json','rows.json']})
    io.save(out/'summary.json',summary)
    lines=['# 415｜真实支持上的部分质量界与搜索前沿范围诊断','',
        '2026-09-25。16个旧支持上下文、每例33个查询输入；不读取查询答案、不计算MSE。413完成新搜索与证书生成并通过独立搜索审计；414复用这些已审计搜索状态测试不同读出，每个上下文的精确区间均重放。','',
        '413的部分质量公式在四个n24上下文均因尚无完整模式而得到M=0；尚未搜索完不能假称已知后验质量。其前沿却已约束14–23个样本，因此414直接包络所有未排除函数，而不要求已找到正内质量。','',
        '| seed | n | 完整模式数 | 前沿块数 | 部分质量区间均宽 | 廉价包络均宽 | 前沿相交后均宽 | 范围组件秒 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['n']} | {r['source_full_modes']} | {r['cover_cells']} | {r['mean_mass_interval_width']:.8g} | {r['mean_cheap_width']:.8g} | {r['mean_intersection_width']:.8g} | {r['range_component_seconds']:.6g} |")
    lines+=['','## 相对廉价支持包络的额外基线修正','',
        '这是区间是否排除已有预测的计数，不能代替任务误差或PC-ALM独立收益。基线预测来自405封存的0.5秒数组；它们不属于此组件的在线输入，生成成本未加进本表。','',
        '| 归档基线 | 有额外修正的上下文 | 额外修正查询数 / 528 |','| --- | ---: | ---: |']
    for name,r in aggregates.items():lines.append(f"| {name} | {r['contexts_with_additional_changes']}/16 | {r['additional_queries_beyond_cheap']}/528 |")
    lines+=['','## 数学条件和下一步','',
        '范围覆盖每个与支持相容的参数，因此在生成教师确实相容时，投影平方误差逐查询不增；这不是只对发现模式的条件后验作保证。零体积非空块也保留。实际收益仍需新的封存预测与查询评估检验，当前未读目标。','',
        '本表使用33点和归档前沿，计时是新组件实际运行时间，不是硬截止的端到端结果。下一步必须接入新连续worker，计搜索、界构造、所有257查询读出、备用预测、清理与状态；并让无乘子／PDHG及强BP使用同样范围接口。','',
        '[413冻结方案](413_partial_support_certificate_protocol_v1.md) · [414冻结方案](414_frontier_range_certificate_protocol_v1.md) · [413审计](../../results/partial_support_certificate/audit_v1/summary.json) · [414全部数据](../../results/frontier_range_certificate/development_v1/rows.json) · [414摘要](../../results/frontier_range_certificate/development_v1/summary.json)','',
        '尚无独立任务优势，未完成三域真实模型要求，Goal保持ACTIVE。']
    (ROOT/ENTRY).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    io.save(out/'report_manifest.json',dict(entry_file=ENTRY,entry_sha256=io.sha(ROOT/ENTRY),summary_sha256=io.sha(out/'summary.json')))
    print(summary,flush=True)


if __name__=='__main__':
    out=BASE/'development_v1';out.mkdir(parents=True,exist_ok=False)
    try:main(out)
    except Exception:
        io.save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
