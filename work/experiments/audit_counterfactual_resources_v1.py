"""293 independent timing/resource arithmetic and all output/journal checks."""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
import counterfactual_resource_suite_v1 as suite
from diagnose_gradient_flat_split_states_v1 import read,save,sha


def run(root,out):
    start=time.perf_counter();hashes=suite.gate(root)
    folder=root/'results/counterfactual_resources/calibration_v1'
    s=suite.complete(folder);p=read(folder/'protocol.json')
    assert p['source_sha256']==hashes and p['configs']==suite.catalogue(root)
    assert p['seeds']==suite.SEEDS and p['repeats']==3 and p['order_seed']==293929
    assert p['memory_seeds']==[5910000,5910063] and not p['query_targets_accessed']
    assert p['preflight_summary_sha256']==sha(root/'results/counterfactual_resources/preflight_v1/summary.json')
    save(out/'protocol.json',dict(source_sha256=hashes,calibration_summary_sha256=sha(folder/'summary.json'),
        scope='All37 methods,888 timings,74 memory calls,all saved arrays and call journals; no teacher',
        query_targets_accessed=False))
    cfgs=p['configs'];names=[c['name'] for c in cfgs]
    timings=read(folder/'timings.json');memory=read(folder/'memory.json');warmup=read(folder/'warmup.json')
    files=read(folder/'files.json');counts=Counter()
    assert len(timings)==888 and len(memory)==74 and len(warmup)==37 and len(files)==296
    expected=[(seed,c['name'],rep) for seed in suite.SEEDS for c in cfgs for rep in range(3)]
    order=np.random.default_rng(293929).permutation(888)
    assert [(r['seed'],r['method'],r['repeat']) for r in timings]==[expected[int(i)] for i in order]
    assert [(r['seed'],r['method']) for r in memory]==[(seed,name) for seed in p['memory_seeds'] for name in names]
    assert [(r['seed'],r['method']) for r in warmup]==[(5910001,name) for name in names]
    for kind,rows in [('timing',timings),('memory',memory),('warmup',warmup)]:
        for i,r in enumerate(rows):
            assert r==read(folder/'calls'/f'{kind}_{i:03d}.json')
            assert r['seconds']>0 and np.isfinite(r['seconds'])
            assert r['metadata']['charged_complete_seconds']==r['seconds']
            assert r['sha256']==files[r['file']]
            if kind=='timing':assert r['order']==i
            if kind=='memory':
                assert r['instrumented_seconds_not_benchmark']==r['seconds']
                assert 0<=r['traced_current_bytes']<=r['traced_peak_bytes']
            counts[kind+'_records']+=1
    for name,digest in files.items():
        assert sha(folder/name)==digest
        with np.load(folder/name,allow_pickle=False) as z:
            for field in suite.FIELDS:
                assert z[field].shape==(257,) and z[field].dtype==np.float64
                assert np.isfinite(z[field]).all() and np.all((z[field]>=0)&(z[field]<=1))
                counts['projected_arrays']+=1
            for key in z.files:
                if np.issubdtype(z[key].dtype,np.number):assert np.isfinite(z[key]).all()
            for m in [r for r in memory if r['file']==name]:
                assert m['returned_array_element_bytes_subtotal']==sum(z[k].nbytes for k in z.files)
    env=read(folder/'environments.json')
    assert env[0]==p['environment_start'] and all(not r['other_research_or_git_pack_processes'] for r in env)
    for i,row in enumerate(env[1:],2):assert row==read(folder/'calls'/f'environment_{i:03d}.json')
    table=read(folder/'methods.json');assert [r['method'] for r in table]==names
    means={}
    for c,row in zip(cfgs,table):
        rr=[r for r in timings if r['method']==c['name']]
        seconds=np.array([r['seconds'] for r in rr]);means[c['name']]=float(seconds.mean())
        expectedrow=dict(method=c['name'],group=c['group'],mean_seconds=means[c['name']],
            median_seconds=float(np.median(seconds)),p90_seconds=float(np.quantile(seconds,.9)),maximum_seconds=float(seconds.max()),
            failures=sum(r['metadata']['execution_failed'] for r in rr),
            max_traced_peak_bytes=max(r['traced_peak_bytes'] for r in memory if r['method']==c['name']),
            task_repeats=[dict(seed=seed,seconds=[r['seconds'] for r in rr if r['seed']==seed]) for seed in suite.SEEDS])
        assert row==expectedrow,c['name'];counts['method_tables']+=1
    cap=means[suite.PRIMARY]
    eligible=[n for n in names if means[n]<=cap]
    background=[]
    for group in sorted({c['group'] for c in cfgs}):
        nn=[c['name'] for c in cfgs if c['group']==group]
        if not any(n in eligible for n in nn):background.append(sorted(nn,key=lambda n:(means[n],n))[0])
    expectedselection=dict(primary=suite.PRIMARY,budget_seconds=cap,budget_factor=1.0,sensitivity_factor=1.10,
        within_budget=eligible,over_budget_background=background,
        sensitivity_110_percent=[n for n in names if means[n]<=cap*1.1],all_methods_retained=True,query_quality_used=False)
    assert read(folder/'selection.json')==expectedselection
    assert s['primary_seconds']==cap and s['failures']==sum(r['metadata']['execution_failed'] for r in timings)
    assert s['timing_runs']==888 and s['memory_runs']==74 and s['warmup_runs']==37
    # Check the boundary convention independently on a synthetic non-quality example.
    tiny=[dict(name=suite.PRIMARY,group='a'),dict(name='eq',group='b'),dict(name='hi',group='c')]
    tinyresult=suite.select_budget(tiny,{suite.PRIMARY:1.,'eq':1.,'hi':1.1})
    assert tinyresult['within_budget']==[suite.PRIMARY,'eq'] and tinyresult['over_budget_background']==['hi']
    assert tinyresult['sensitivity_110_percent']==[suite.PRIMARY,'eq','hi']
    assert suite.gate(root)==hashes
    result=dict(passed=True,methods=37,counts=dict(counts),files=296,environments=len(env),
        primary_seconds=cap,within_budget=eligible,sensitivity_110_percent=expectedselection['sensitivity_110_percent'],
        over_budget_background=background,query_targets_accessed=False,core_research_goal_complete=False,
        calibration_summary_sha256=sha(folder/'summary.json'),seconds=time.perf_counter()-start,
        outputs_sha256={'protocol.json':sha(out/'protocol.json')},
        next='Read all37 resource results, then freeze a new independent query experiment with competitive controls')
    save(out/'summary.json',result);print(result,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    root=p.parse_args().project.resolve();out=root/'results/counterfactual_resources/audit_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise


if __name__=='__main__':main()
