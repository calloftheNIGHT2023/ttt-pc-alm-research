"""Independent expanded-polynomial replay of 266C, including every interval."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from run_multiplier_fixed_point_screen import sha, dump


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve(); src=Path(__file__).parent
    base=root/'results/confirmation_conditional_risk'; inp=base/'decomposition'
    out=base/'decomposition_audit'; out.mkdir(parents=True,exist_ok=True)
    assert not (out/'protocol.json').exists()
    s=json.loads((inp/'summary.json').read_text()); assert s['passed']
    for n,h in s['outputs_sha256'].items(): assert sha(inp/n)==h
    p=json.loads((inp/'protocol.json').read_text()); assert not p['phase_accesses_query_targets']
    hashes=dict(p['source_sha256']); hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items(): assert sha(src/n)==h,n
    for n,h in p['design_sha256'].items(): assert sha(root/'outputs/ttt-pc-alm-research'/n)==h
    assert sha(base/'moments_audit/summary.json')==p['moments_audit_sha256']
    old=root/'results/matched_budget_confirmation/conditioned_confirmation'
    assert sha(old/'before_query_manifest.json')==p['frozen_predictor_manifest_sha256']
    before=json.loads((old/'before_query_manifest.json').read_text())
    assert sha(old/'rows.json')==before['rows_sha256']
    originals={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text())}
    dump(out/'protocol.json',dict(source_sha256=hashes,decomposition_summary_sha256=sha(inp/'summary.json'),
        phase_accesses_query_targets=False,scope='Rebuild mixtures from regional matrices, expanded-polynomial risks, summaries, both grids and all270 bootstrap intervals'))
    saved={(r['seed'],r['method'],r['grid']):r for r in json.loads((inp/'rows.json').read_text())}
    sensitivity={(r['seed'],r['method']):r for r in json.loads((inp/'sensitivity.json').read_text())}
    reference_gaps={(r['seed'],r['grid']):r for r in json.loads((inp/'reference_gaps.json').read_text())}
    files=json.loads((inp/'files.json').read_text())
    tasks=json.loads((base/'moments/tasks.json').read_text())
    metrics=p['metrics']; names=p['methods']; seeds=p['seeds']; counts=Counter()
    replay={}; maximum=0.; begin=time.perf_counter()

    def check(x,y):
        nonlocal maximum
        gap=float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
        maximum=max(maximum,gap); assert gap<1e-12,(gap,x,y)
        counts['scalar_or_array_numeric_checks']+=1

    for task in tasks:
        seed=task['seed']; path=base/'moments'/task['file']; assert sha(path)==task['sha256']
        filename=f'{seed}_curves.npz'; assert sha(inp/filename)==files[filename]
        with np.load(path) as z:
            w=z['volumes']/z['volumes'].sum(); mu=z['means']; sec=z['seconds']
            # Direct summation, not the analyzer's einsum path.
            m=np.sum(w[:,None,None]*mu,axis=0); t=np.sum(w[:,None,None]*sec,axis=0)
            with np.load(inp/filename) as c:
                assert c['methods'].tolist()==names and c['keys'].tolist()==task['keys']
                check(c['weights'],w); check(c['full_means'],m); check(c['full_second_moments'],t)
                assert c['q'].tobytes()==z['q'].tobytes()
                for j,name in enumerate(names):
                    original=originals[seed,name]
                    assert sha(old/original['file'])==original['sha256']==before['prediction_files'][original['file']]
                    with np.load(old/original['file']) as zz:
                        actual=zz['prediction']; assert actual.tobytes()==c['predictions'][j].tobytes()
                    keys=original['metadata']['positive_modes'] if original['readout']=='mode' else []
                    mask=np.array([key in keys for key in task['keys']])
                    assert mask.tobytes()==c['masks'][j].tobytes()
                    if original['readout']=='mode' and keys:
                        kind='nonempty_pool'; mass=float(w[mask].sum()); wk=w[mask]/mass
                        k=np.sum(wk[:,None,None]*mu[mask],axis=0)
                        tk=np.sum(wk[:,None,None]*sec[mask],axis=0)
                    else:
                        kind='empty_pool_fallback' if original['readout']=='mode' else 'fixed_prediction_head'
                        mass=0. if original['readout']=='mode' else None
                        k=np.tile(actual,(4,1)); tk=k*k
                    assert c['kinds'][j]==kind
                    check(k,c['strategy_means'][j]); check(tk,c['strategy_second_moments'][j])
                    for grid in p['grids']:
                        sl=slice(None) if grid==257 else slice(None,None,2)
                        pp=actual[sl]; ss=saved[seed,name,grid]; assert ss['kind']==kind
                        assert ss['posterior_mass'] is None if mass is None else abs(ss['posterior_mass']-mass)<1e-12
                        assert ss['complete_call_seconds']==original['seconds']
                        pairvalues=[]
                        for index,(a,b) in enumerate(p['batch_pairs']):
                            ma,mb=m[a,sl],m[b,sl]; ka,kb=k[a,sl],k[b,sl]
                            aex=float(np.average(pp*pp-pp*(ma+mb)+ma*mb))
                            pex=float(np.average(ka*kb-ka*mb-kb*ma+ma*mb))
                            read=float(np.average(pp*pp-pp*(ka+kb)+ka*kb))
                            cross=float(np.average(pp*(ka+kb-ma-mb)-2*ka*kb+ma*kb+mb*ka))
                            mc=float(np.average((tk[a,sl]+tk[b,sl])*.5-ka*kb)/2048) if keys else 0.
                            bayes=float(np.average((t[a,sl]+t[b,sl])*.5-ma*mb))
                            total=float(np.average(pp*pp-pp*(ma+mb)+(t[a,sl]+t[b,sl])*.5))
                            vals=dict(actual_excess=aex,policy_excess=pex,read_error=read,cross_term=cross,
                                expected_mc=mc,resampled_expected_excess=pex+mc,bayes_variance=bayes,conditional_total=total)
                            for field,value in vals.items(): check(value,ss['pairs'][index][field]); counts['pair_metric_replays']+=1
                            check(aex-pex,read+cross); check(total,bayes+aex)
                            pairvalues.append(vals)
                        avg={f:float(np.average([v[f] for v in pairvalues])) for f in metrics}
                        for f,value in avg.items(): check(value,ss[f]); counts['averaged_metric_replays']+=1
                        replay[seed,name,grid]=dict(**avg,pairs=pairvalues,kind=kind,seconds=original['seconds'])
                    sen=sensitivity[seed,name]
                    for f in metrics:
                        check(replay[seed,name,257][f]-replay[seed,name,129][f],sen['grid_257_minus_129'][f])
                        check(replay[seed,name,257]['pairs'][0][f]-replay[seed,name,257]['pairs'][1][f],sen['pair01_minus_pair23'][f])
                    counts['frozen_predictors']+=1
            for grid in p['grids']:
                sl=slice(None) if grid==257 else slice(None,None,2)
                expected=[float(np.mean((m[a,sl]-m[b,sl])**2)) for a,b in p['batch_pairs']]
                check(expected,reference_gaps[seed,grid]['pair_l2'])
        counts['task_curves']+=1
    for ss in json.loads((inp/'methods.json').read_text()):
        rr=[replay[seed,ss['method'],ss['grid']] for seed in seeds]
        for f in metrics:
            check(np.mean([r[f] for r in rr]),ss[f])
            check(np.mean([r['pairs'][0][f]-r['pairs'][1][f] for r in rr]),ss['mean_pair01_minus_pair23'][f])
            check(max(abs(r['pairs'][0][f]-r['pairs'][1][f]) for r in rr),ss['maximum_abs_pair_difference'][f])
            # Counts use the published arithmetic estimator; roundoff-sign flips near zero are not errors.
            assert ss['negative_task_estimates'][f]==sum(saved[seed,ss['method'],ss['grid']][f]<0 for seed in seeds)
        for field,kind in [('empty_pool_tasks','empty_pool_fallback'),('nonempty_pool_tasks','nonempty_pool'),('head_tasks','fixed_prediction_head')]:
            assert ss[field]==sum(r['kind']==kind for r in rr)
        check(ss['mean_complete_call_seconds'],np.mean([r['seconds'] for r in rr]))
        counts['method_grid_summaries']+=1
    ids=np.random.default_rng(262193).integers(0,64,(20000,64))
    for ss in json.loads((inp/'paired.json').read_text()):
        g=ss['grid']; f=ss['metric']; delta=np.array([replay[s,p['primary'],g][f]-replay[s,ss['control'],g][f] for s in seeds])
        check(delta.mean(),ss['mean_difference']); check(np.quantile(np.mean(delta[ids],axis=1),[.025,.975]),ss['descriptive_ci95'])
        assert int((delta < -1e-12).sum())==ss['lower_tasks']
        assert int((abs(delta)<=1e-12).sum())==ss['equal_tasks']
        assert int((delta > 1e-12).sum())==ss['higher_tasks']
        counts['paired_intervals']+=1
    assert len(replay)==len(saved)==5888 and counts['paired_intervals']==270
    ans=dict(passed=True,counts=counts,maximum_numeric_replay_gap=maximum,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),phase_accesses_query_targets=False)
    dump(out/'summary.json',ans); print(json.dumps(ans),flush=True)


if __name__=='__main__': main()
