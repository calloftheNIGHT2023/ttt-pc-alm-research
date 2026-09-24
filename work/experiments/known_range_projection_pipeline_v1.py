"""288 supplementary projection and evaluation; originals remain immutable."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from verify_known_range_projection_v1 import project
from analyze_recovered_online_comparison import forward
import probe_confirmation_statistics as statistics


METRICS = ['mse257', 'mse129', 'point_mse257', 'point_mse129']
FIELDS = ['prediction', 'point_prediction']
SOURCES = ['known_range_projection_pipeline_v1.py', 'audit_known_range_projection_v1.py',
           'continue_known_range_projection_v1.py', 'verify_known_range_projection_v1.py',
           'diagnose_gradient_flat_split_states_v1.py', 'probe_confirmation_statistics.py',
           'analyze_recovered_online_comparison.py']


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def save_npz(path, **arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def paths(root, stage):
    original = root/'results/probe_credit_confirmation'
    extra = root/'results/known_range_projection'
    return dict(original=original, pred=original/('runner_preflight_v2' if stage=='preflight' else 'predictions'),
                raw_eval=original/('evaluation_preflight_v6' if stage=='preflight' else 'evaluation_v6'),
                projected=extra/f'predictions_{stage}_v1', evaluation=extra/f'evaluation_{stage}_v1',
                audit=extra/f'audit_{stage}_v1', extra=extra)


def source_hashes(root):
    return {name: sha(root/'work/experiments'/name) for name in SOURCES}


def complete(folder):
    result = read(folder/'summary.json')
    assert result['passed']
    for name, digest in result['outputs_sha256'].items():
        assert sha(folder/name) == digest, (folder, name)
    return result


def gate(root, stage):
    seal = read(root/'results/round_287_audit_v6.json')
    assert seal['passed'] and seal['tasks'] == 8192 and not seal['core_research_goal_complete']
    for name, digest in seal['source_sha256'].items():
        assert sha(root/'work/experiments'/name) == digest, name
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    extra = root/'results/known_range_projection'
    old = read(extra/'preflight_v1/summary.json')
    assert old['passed'] and old['prediction_arrays'] == 108 and old['exact_inequality_checks'] == 55
    assert old['protocol_sha256'] == sha(extra/'preflight_v1/protocol.json')
    assert old['rows_sha256'] == sha(extra/'preflight_v1/rows.json')
    hashes = source_hashes(root)
    if stage == 'confirmation':
        preflight = complete(extra/'audit_preflight_v1')
        assert preflight['tasks'] == 2 and preflight['supplementary_analysis']
        assert read(extra/'audit_preflight_v1/protocol.json')['source_sha256'] == hashes
    return hashes


def changes(original, projected):
    return dict(below_zero=int(np.sum(original < 0)), above_one=int(np.sum(original > 1)),
                changed_values=int(np.sum(original != projected)),
                bitwise_unchanged=original.tobytes() == projected.tobytes(),
                original_min=float(original.min()), original_max=float(original.max()),
                maximum_absolute_change=float(np.max(abs(original-projected))))


def produce(root, stage, out, hashes):
    loc = paths(root, stage)
    inp = loc['pred']
    original_summary = read(inp/'summary.json')
    before = read(inp/'before_query_manifest.json')
    p = read(inp/'protocol.json')
    seeds, names = p['seeds'], p['methods']
    assert seeds == ([5910000, 5910063] if stage=='preflight' else list(range(5500000,5508192)))
    assert original_summary['passed'] and original_summary['predictors'] == len(seeds)*27
    assert before['protocol_sha256'] == sha(inp/'protocol.json') and not before['query_targets_accessed']
    assert [entry['seed'] for entry in before['task_commits']] == seeds
    protocol = dict(stage=stage, created_utc=now(), supplementary_analysis=True,
         original_query_results_already_known=True, query_targets_accessed=False,
         seeds=seeds, methods=names, primary=p['primary'], fields=FIELDS, block_tasks=128,
         source_sha256=hashes, final_original_seal_sha256=sha(root/'results/round_287_audit_v6.json'),
         original_summary_sha256=sha(inp/'summary.json'), original_protocol_sha256=sha(inp/'protocol.json'),
         original_before_query_manifest_sha256=sha(inp/'before_query_manifest.json'),
         supplementary_plan_sha256=sha(root/'outputs/ttt-pc-alm-research/288_full_projection_protocol.md'),
         original_plan_sha256=sha(root/'outputs/ttt-pc-alm-research/288_bounded_output_baseline_audit_plan.md'),
         rule='clip every saved prediction and point_prediction to [0,1], no adaptation',
         resource_scope='Projection microcall seconds only; original total excludes this operation; output buffers not process peak')
    save(out/'protocol.json', protocol)
    begin = time.perf_counter()
    blocks, outputs = [], {'protocol.json': sha(out/'protocol.json')}
    for first in range(0, len(seeds), 128):
        selection = seeds[first:first+128]
        projected = np.empty((len(selection),27,2,257), dtype=np.float64)
        elapsed = np.empty((len(selection),27), dtype=np.float64)
        records = []
        for i, seed in enumerate(selection):
            entry = before['task_commits'][first+i]
            commit_file = inp/entry['file']
            assert sha(commit_file) == entry['sha256']
            commit = read(commit_file)
            assert commit['seed'] == seed and commit['methods'] == names
            assert not commit['query_targets_accessed']
            assert commit['protocol_sha256'] == before['protocol_sha256']
            assert sha(inp/commit['rows_file']) == commit['files'][commit['rows_file']]
            rows = {r['method']: r for r in read(inp/commit['rows_file'])}
            assert set(rows) == set(names) and len(rows) == 27
            local = []
            for j, name in enumerate(names):
                row = rows[name]
                assert row['seed'] == seed
                assert sha(inp/row['file']) == row['sha256'] == commit['files'][row['file']]
                with np.load(inp/row['file'], allow_pickle=False) as z:
                    original = [z[field] for field in FIELDS]
                assert all(a.shape == (257,) and a.dtype == np.float64 for a in original)
                tick = time.perf_counter()
                pair = [project(a) for a in original]
                elapsed[i,j] = time.perf_counter()-tick
                projected[i,j] = pair
                local.append(dict(method=name, original_file=row['file'], original_sha256=row['sha256'],
                                  original_execution_failed=row['metadata']['execution_failed'],
                                  fields={field: changes(a,b) for field,a,b in zip(FIELDS,original,pair)}))
            records.append(dict(seed=seed, original_commit=entry, original_rows_file=commit['rows_file'],
                                original_rows_sha256=commit['files'][commit['rows_file']], methods=local))
        stem = f'block_{first:05d}'
        datafile, recordsfile = stem+'.npz', stem+'.json'
        save_npz(out/datafile, seeds=np.array(selection), methods=np.array(names), fields=np.array(FIELDS),
                 projected=projected, projection_seconds=elapsed)
        save(out/recordsfile, records)
        block = dict(first=first, tasks=len(selection), data_file=datafile, records_file=recordsfile,
                     data_sha256=sha(out/datafile), records_sha256=sha(out/recordsfile))
        blocks.append(block)
        outputs.update({datafile: block['data_sha256'], recordsfile: block['records_sha256']})
        print(json.dumps(dict(stage=stage, phase='projection', tasks=first+len(selection), total=len(seeds),
                              seconds=time.perf_counter()-begin)), flush=True)
    save(out/'blocks.json', blocks)
    outputs['blocks.json'] = sha(out/'blocks.json')
    assert source_hashes(root) == hashes
    result = dict(passed=True, stage=stage, supplementary_analysis=True, tasks=len(seeds), methods=27,
                  predictors=len(seeds)*27, arrays=len(seeds)*54, blocks=len(blocks),
                  query_targets_accessed=False, adaptation_rerun=False, original_predictions_modified=False,
                  projected_output_bytes_per_method=2*257*8, seconds=time.perf_counter()-begin,
                  outputs_sha256=outputs, core_research_goal_complete=False,
                  next='Evaluate all saved projected readouts, then independent all-original-file audit')
    save(out/'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='outputs_sha256'}), flush=True)


def evaluate(root, stage, out, hashes):
    loc = paths(root, stage)
    pred = loc['projected']
    produced = complete(pred)
    p = read(pred/'protocol.json')
    assert p['source_sha256'] == hashes and not produced['query_targets_accessed']
    raw_summary = complete(loc['raw_eval'])
    raw_p = read(loc['raw_eval']/'protocol.json')
    names, seeds, primary = p['methods'], p['seeds'], p['primary']
    assert raw_p['methods'] == names and raw_p['seeds'] == seeds
    controls = [name for name in names if name != primary]
    family = [name for name in controls if name != 'probe_all_alm64']
    protocol = dict(stage=stage, created_utc=now(), supplementary_analysis=True, original_query_results_already_known=True,
         source_sha256=hashes, seeds=seeds, methods=names, primary=primary, controls=controls,
         primary_family=family, metrics=METRICS, primary_metric='mse257', query_targets_accessed=True,
         prediction_summary_sha256=sha(pred/'summary.json'), raw_evaluation_summary_sha256=sha(loc['raw_eval']/'summary.json'),
         supplementary_plan_sha256=p['supplementary_plan_sha256'],
         bootstrap=dict(repetitions=100000,seed=288193,block=64,quantile_method='linear',
                        descriptive_quantiles=[.025,.975],adjusted_upper_quantile=.998),
         inference='Supplementary approximate paired task bootstrap, not new independent confirmation or finite-sample guarantee')
    save(out/'protocol.json', protocol)
    begin = time.perf_counter()
    n = len(seeds)
    with np.load(loc['raw_eval']/'task_metrics.npz', allow_pickle=False) as z:
        assert z['seeds'].tolist() == seeds and z['methods'].tolist() == names and z['metric_names'].tolist() == METRICS
        raw_risk, original_times, failures = z['risk'].copy(), z['seconds'].copy(), z['failures'].copy()
    risk = np.empty_like(raw_risk)
    projection_times = np.empty_like(original_times)
    field_records = defaultdict(list)
    q = np.linspace(0,1,257)
    for block in read(pred/'blocks.json'):
        first, count = block['first'], block['tasks']
        with np.load(pred/block['data_file'], allow_pickle=False) as z:
            assert z['seeds'].tolist() == seeds[first:first+count] and z['methods'].tolist() == names and z['fields'].tolist() == FIELDS
            values = z['projected']
            projection_times[first:first+count] = z['projection_seconds']
        assert values.shape == (count,27,2,257) and np.isfinite(values).all()
        assert np.all((values >= 0) & (values <= 1))
        for i, seed in enumerate(seeds[first:first+count]):
            truth = forward(q,np.random.default_rng(seed).uniform(-.12,.12,4)[None])[0]
            assert np.all((truth>=0)&(truth<=1))
            errors = (values[i]-truth)**2
            risk[first+i] = np.stack([errors[:,0].mean(1),errors[:,0,::2].mean(1),
                                      errors[:,1].mean(1),errors[:,1,::2].mean(1)],axis=1)
        for record in read(pred/block['records_file']):
            for method in record['methods']:
                for field, values in method['fields'].items():
                    field_records[method['method'],field].append(values)
    tolerance = 1e-12 + 1e-11*abs(raw_risk)
    assert np.all(risk-raw_risk <= tolerance), 'Projection increased risk beyond frozen floating tolerance'
    assert np.all(np.isfinite(projection_times)) and np.all(projection_times >= 0)
    save_npz(out/'task_metrics.npz',seeds=np.array(seeds),methods=np.array(names),metric_names=np.array(METRICS),
             risk=risk,raw_risk=raw_risk,original_seconds=original_times,projection_seconds=projection_times,failures=failures)
    pi = names.index(primary)
    indices = [names.index(name) for name in controls]
    differences = risk[:,pi:pi+1]-risk[:,indices]
    boot, digest = statistics.bootstrap_means(differences.reshape(n,-1),100000,288193,64)
    boot = boot.reshape(100000,26,4)
    save_npz(out/'bootstrap_means.npz',means=boot,controls=np.array(controls),metrics=np.array(METRICS))
    comparisons = []
    for j, name in enumerate(controls):
        for k, metric in enumerate(METRICS):
            dd = differences[:,j,k]
            upper = float(np.quantile(boot[:,j,k],.998,method='linear')) if name in family and k==0 else None
            comparisons.append(dict(control=name,metric=metric,tasks=n,mean_difference=float(dd.mean()),
                 paired_sample_sd=float(dd.std(ddof=1)),descriptive95=np.quantile(boot[:,j,k],[.025,.975],method='linear').tolist(),
                 supplementary_main_comparison=upper is not None,adjusted_upper=upper,
                 upper_below_zero=upper<0 if upper is not None else None,
                 improved_tasks=int(np.sum(dd<0)),equal_tasks=int(np.sum(dd==0)),worse_tasks=int(np.sum(dd>0))))
    methods = []
    for j, name in enumerate(names):
        diagnostics = {}
        for field in FIELDS:
            rr = field_records[name,field]
            assert len(rr)==n
            diagnostics[field] = dict(below_zero=sum(r['below_zero'] for r in rr),above_one=sum(r['above_one'] for r in rr),
                changed_values=sum(r['changed_values'] for r in rr),changed_tasks=sum(r['changed_values']>0 for r in rr),
                all_tasks_bitwise_unchanged=all(r['bitwise_unchanged'] for r in rr),
                original_min=min(r['original_min'] for r in rr),original_max=max(r['original_max'] for r in rr),
                maximum_absolute_change=max(r['maximum_absolute_change'] for r in rr))
        methods.append(dict(method=name, metrics={metric:dict(raw_mean=float(raw_risk[:,j,k].mean()),
             projected_mean=float(risk[:,j,k].mean()),mean_change=float((risk[:,j,k]-raw_risk[:,j,k]).mean()),
             projected_task_sd=float(risk[:,j,k].std(ddof=1))) for k,metric in enumerate(METRICS)},
             original_complete_mean_seconds=float(original_times[:,j].mean()),
             isolated_projection_pair_mean_seconds=float(projection_times[:,j].mean()),
             projected_output_numeric_bytes=4112,failures=int(failures[:,j].sum()),projection=diagnostics))
    save(out/'comparisons.json',comparisons)
    save(out/'methods.json',methods)
    passed = sum(r['upper_below_zero'] for r in comparisons if r['supplementary_main_comparison'])
    result = dict(passed=True,stage=stage,supplementary_analysis=True,tasks=n,predictors=n*27,comparisons=104,
         supplementary_main_family_size=25,supplementary_negative_upper_bounds=int(passed),
         original_main_negative_upper_bounds=raw_summary['main_adjusted_negative_upper_bounds'],
         bootstrap_index_sha256=digest,query_targets_accessed=True,core_research_goal_complete=False,
         resource_scope='Original complete time unchanged/excludes projection; isolated projection timings are not matched end-to-end benchmarks',
         seconds=time.perf_counter()-begin,outputs_sha256={name:sha(out/name) for name in
         ['protocol.json','task_metrics.npz','bootstrap_means.npz','comparisons.json','methods.json']},
         next='Independent projection, risk, all bootstrap values and summary replay before interpreting supplementary results')
    assert source_hashes(root)==hashes
    save(out/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='outputs_sha256'}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--stage',choices=['preflight','confirmation'],required=True)
    parser.add_argument('--phase',choices=['predict','evaluate'],required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    hashes = gate(root,args.stage)
    out = paths(root,args.stage)['projected' if args.phase=='predict' else 'evaluation']
    out.mkdir(parents=True,exist_ok=False)
    try:
        (produce if args.phase=='predict' else evaluate)(root,args.stage,out,hashes)
    except BaseException as exc:
        save(out/'failure.json',dict(utc=now(),error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True))
        raise


if __name__=='__main__': main()
