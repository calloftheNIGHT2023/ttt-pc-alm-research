"""Independent census and frozen original predictions for287 extra controls."""
import argparse,json,statistics
from pathlib import Path
from collections import Counter
import numpy as np
import probe_credit_confirmation_suite as suite
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';inp=base/'head_preflight';out=base/'head_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    s=json.loads((inp/'summary.json').read_text());assert s['passed']
    for n,h in s['outputs_sha256'].items():assert sha(inp/n)==h,n
    p=json.loads((inp/'protocol.json').read_text());hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    assert p['all_configurations']==suite.catalogue(root) and len(p['additional_configurations'])==8
    dump(out/'protocol.json',dict(source_sha256=hashes,head_preflight_sha256=sha(inp/'summary.json'),phase_accesses_new_contexts_or_targets=False))
    counts=Counter();index=suite.resources.frozen_inputs(root);configs={c['name']:c for c in p['additional_configurations']};files=json.loads((inp/'files.json').read_text())
    rows=json.loads((inp/'primitive_rows.json').read_text());assert len(rows)==32
    assert {(r['seed'],r['method']) for r in rows}=={(seed,name) for seed in p['primitive_seeds'] for name in configs}
    for r in rows:
        assert sha(inp/r['file'])==files[r['file']]==r['sha256']
        with np.load(inp/r['file']) as z:a={k:z[k].copy() for k in z.files}
        counts['independent_gold_arrays']+=suite.resources.check_frozen(root,r['seed'],configs[r['method']],a,r['metadata'],index);counts['predictors']+=1
    timings=json.loads((inp/'timings.json').read_text());memory=json.loads((inp/'memory.json').read_text());table={r['method']:r for r in json.loads((inp/'methods.json').read_text())};assert len(timings)==192 and len(memory)==16
    jobs=[(seed,c['name'],rep) for seed in p['timing_seeds'] for c in p['additional_configurations'] for rep in range(3)];order=np.random.default_rng(287929).permutation(192)
    assert [(r['seed'],r['method'],r['repeat']) for r in timings]==[jobs[int(i)] for i in order]
    for i,r in enumerate(timings):assert r['order']==i and r['seconds']==r['metadata']['charged_complete_seconds'] and r['seconds']>0;counts['complete_timed_calls']+=1
    assert {(r['seed'],r['method']) for r in memory}=={(seed,name) for seed in p['memory_seeds'] for name in configs}
    for r in memory:assert r['traced_peak_bytes']>=r['traced_current_bytes']>=0 and r['process_after']['rss']>0;counts['separate_memory_calls']+=1
    old=root/'results/probe_credit_resources/calibration/summary.json';assert sha(old)==p['original_resource_summary_sha256'];cap=json.loads(old.read_text())['primary_seconds']
    for name,row in table.items():
        rr=[r for r in timings if r['method']==name];assert len(rr)==24;values=[r['seconds'] for r in rr];mean=statistics.mean(values)
        for key,value in [('mean_seconds',mean),('median_seconds',statistics.median(values)),('p90_seconds',float(np.quantile(values,.9))),('maximum_seconds',max(values))]:assert abs(row[key]-value)<1e-12
        label='main_1.00' if mean<=cap else 'sensitivity_1.10' if mean<=cap*1.1 else 'measured_over_budget';assert row['resource_class']==label
        assert row['failures']==sum(r['metadata']['execution_failed'] for r in rr);assert row['max_traced_peak_bytes']==max(r['traced_peak_bytes'] for r in memory if r['method']==name);counts['resource_tables']+=1
    assert s['failures']==sum(r['metadata']['execution_failed'] for r in timings)
    ans=dict(passed=True,counts=counts,phase_accesses_new_contexts_or_targets=False,may_preflight_confirmation_runner=True,protocol_sha256=sha(out/'protocol.json'),
        scope='Old-context extra-control validation only; not a new independent query result')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
