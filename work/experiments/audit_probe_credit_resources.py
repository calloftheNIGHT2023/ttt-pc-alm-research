"""Independently reconstruct285 timing/memory census and time-only inclusion."""
import argparse,json,math,statistics
from collections import Counter
from pathlib import Path
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump
import probe_credit_resource_suite as suite


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_resources';inp=base/'calibration';out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    ss=json.loads((inp/'summary.json').read_text());assert ss['passed']
    for n,h in ss['outputs_sha256'].items():assert sha(inp/n)==h,n
    p=json.loads((inp/'protocol.json').read_text());hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    assert p['phase_accesses_query_targets'] is False and p['budget_factor']==1 and p['configs']==suite.catalogue()
    dump(out/'protocol.json',dict(source_sha256=hashes,calibration_summary_sha256=sha(inp/'summary.json'),phase_accesses_query_targets=False,scope='Independent census, saved predictor identities, runtime aggregate and inclusion reconstruction; not a second wall-clock benchmark'))
    cfgs=p['configs'];names=[c['name'] for c in cfgs];configs={c['name']:c for c in cfgs};seeds=p['seeds'];timings=json.loads((inp/'timings.json').read_text());memory=json.loads((inp/'memory.json').read_text());warm=json.loads((inp/'warmup.json').read_text());files=json.loads((inp/'files.json').read_text());counts=Counter();index=suite.frozen_inputs(root)
    assert len(timings)==600 and len(memory)==50 and len(warm)==25 and len(files)==200
    expected=[(seed,cfg['name'],rep) for seed in seeds for cfg in cfgs for rep in range(3)];order=np.random.default_rng(285929).permutation(600)
    assert [(r['seed'],r['method'],r['repeat']) for r in timings]==[expected[int(i)] for i in order]
    for i,r in enumerate(timings):
        assert r['order']==i and r['seconds']==r['metadata']['charged_complete_seconds'] and math.isfinite(r['seconds']) and r['seconds']>0
        assert files[r['file']]==r['sha256'];counts['timed_calls']+=1
        if configs[r['method']]['family'] in ['local','probe'] and not r['metadata']['execution_failed']:
            assert r['metadata']['trace_enabled'] is False and r['metadata']['scans']['scanned_states']==0
    for r in warm:assert r['seed']==5910001 and r['method'] in names and r['seconds']>0;counts['separate_warmups']+=1
    assert len({r['method'] for r in warm})==25
    assert {(r['seed'],r['method']) for r in memory}=={(s,n) for s in p['memory_seeds'] for n in names}
    for r in memory:
        assert r['traced_peak_bytes']>=r['traced_current_bytes']>=0
        assert r['process_before']['rss']>0 and r['process_after']['rss']>0
        assert r['instrumented_seconds_not_benchmark']>0 and r['sha256']==files[r['file']];counts['separate_memory_calls']+=1
    table={r['method']:r for r in json.loads((inp/'methods.json').read_text())};means={}
    for cfg in cfgs:
        name=cfg['name'];rr=[r for r in timings if r['method']==name];assert len(rr)==24;values=[r['seconds'] for r in rr];means[name]=statistics.mean(values);row=table[name]
        for field,value in [('mean_seconds',means[name]),('median_seconds',statistics.median(values)),('p90_seconds',float(np.quantile(values,.9))),('maximum_seconds',max(values))]:assert abs(row[field]-value)<1e-12
        assert row['failures']==sum(r['metadata']['execution_failed'] for r in rr)
        assert row['max_traced_peak_bytes']==max(r['traced_peak_bytes'] for r in memory if r['method']==name)
        for seed in seeds:
            zz=[r for r in rr if r['seed']==seed];assert {r['repeat'] for r in zz}=={0,1,2};first=zz[0];fn=first['file'];assert sha(inp/fn)==files[fn]
            with np.load(inp/fn) as z:
                a={k:z[k].copy() for k in z.files if k not in ['x_observed','v_observed','q_observed']}
                assert a['prediction'].shape==(257,) and np.isfinite(a['prediction']).all()
                if not first['metadata']['execution_failed']:counts['frozen_gold_arrays']+=suite.check_frozen(root,seed,cfg,a,first['metadata'],index)
                for mr in memory:
                    if mr['seed']==seed and mr['method']==name:assert mr['returned_array_element_bytes_subtotal']==sum(v.nbytes for v in a.values() if isinstance(v,np.ndarray))
            counts['saved_predictors']+=1
        counts['method_aggregates']+=1
    selection=json.loads((inp/'selected_configs.json').read_text());cap=means[suite.PRIMARY];within=[c for c in cfgs if means[c['name']]<=cap];above=[]
    for group in sorted({c['group'] for c in cfgs}):
        cc=[c for c in cfgs if c['group']==group]
        if not any(c in within for c in cc):above.append(min(cc,key=lambda c:(means[c['name']],c['name'])))
    assert selection['within_budget']==[c['name'] for c in within] and selection['over_budget_background']==[c['name'] for c in above]
    assert selection['configs']==within+above and selection['sensitivity_110_percent']==[c['name'] for c in cfgs if means[c['name']]<=1.1*cap]
    assert abs(selection['budget_seconds']-cap)<1e-12 and abs(ss['primary_seconds']-cap)<1e-12
    assert ss['failures']==sum(r['metadata']['execution_failed'] for r in timings)
    for n,h in means.items():assert abs(selection['calibration_means'][n]-h)<1e-12
    counts['time_only_inclusion_decisions']=25
    ans=dict(passed=True,counts=counts,primary_seconds=cap,within_budget=selection['within_budget'],over_budget_background=selection['over_budget_background'],
        protocol_sha256=sha(out/'protocol.json'),phase_accesses_query_targets=False,may_run_selected_budget_development=True,
        scope='Resource selection only; no new query quality or deployment/native-peak generalization')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
