"""Independent scalar teacher and expanded posterior risk for all69 methods, including exact previous65 inheritance."""
import argparse,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump


def scalar_teacher(q,seed):
    biases=np.random.default_rng(seed).uniform(-.12,.12,4);ans=[]
    for x in q:
        h=float(x)
        for b in biases:
            z=h+float(b)
            h=0. if z<0 or z>1 else (2*z if z<=.5 else 2-2*z)
        ans.append(h)
    return np.array(ans)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_continuation_credit';inp=base/'evaluation';out=base/'evaluation_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    s=json.loads((inp/'summary.json').read_text());assert s['passed']
    for n,h in s['outputs_sha256'].items():assert sha(inp/n)==h
    p=json.loads((inp/'protocol.json').read_text());hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'protocol.json',dict(source_sha256=hashes,evaluation_summary_sha256=sha(inp/'summary.json'),scope='All69x64x2 risks and544 paired intervals; scalar branch teacher and direct sums from regional moments'))
    rows={(r['seed'],r['method'],r['grid']):r for r in json.loads((inp/'rows.json').read_text())};files=json.loads((inp/'files.json').read_text())
    posterior=root/'results/confirmation_conditional_risk';tasks={r['seed']:r for r in json.loads((posterior/'moments/tasks.json').read_text())}
    previous=posterior/'decomposition';oldfiles=json.loads((previous/'files.json').read_text());oldp=json.loads((previous/'protocol.json').read_text())
    directories={'new':base/'development','credit':root/'results/certificate_activity_attribution/development','distinct':root/'results/distinct_prior_credit/development','anchor':root/'results/anchor_preserving_fork/development','mixed':root/'results/fixed_restart_credit/development','old':root/'results/matched_budget_confirmation/conditioned_confirmation'}
    inherited=root/'results/certificate_activity_attribution/evaluation';inherit_summary=json.loads((inherited/'summary.json').read_text())
    for n,h in inherit_summary['outputs_sha256'].items():assert sha(inherited/n)==h
    inherited_rows={(r['seed'],r['method'],r['grid']):r for r in json.loads((inherited/'rows.json').read_text())}
    inputs={key:{(r['seed'],r['method']):r for r in json.loads((directory/'rows.json').read_text())} for key,directory in directories.items()}
    replay={};counts=Counter();maxgap=0.;begin=time.perf_counter()

    def check(a,b):
        nonlocal maxgap
        gap=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));maxgap=max(maxgap,gap);assert gap<1e-12,gap;counts['numeric_checks']+=1

    for seed in p['seeds']:
        task=tasks[seed];assert sha(posterior/'moments'/task['file'])==task['sha256'];fn=f'{seed}_evaluation_curves.npz';assert sha(inp/fn)==files[fn]
        ofn=f'{seed}_curves.npz';assert sha(previous/ofn)==oldfiles[ofn]
        with np.load(posterior/'moments'/task['file']) as z,np.load(inp/fn) as curves,np.load(previous/ofn) as old:
            q=z['q'];w=z['volumes']/z['volumes'].sum();mu=z['means'];second=z['seconds'];m=np.sum(w[:,None,None]*mu,axis=0);t=np.sum(w[:,None,None]*second,axis=0)
            check(m,curves['full_means']);check(t,curves['full_second_moments']);truth=scalar_teacher(q,seed)
            assert curves['methods'].tolist()==p['methods'] and curves['new_methods'].tolist()==p['new_methods']
            for j,name in enumerate(p['methods']):
                new=name in p['new_methods'];source=next(key for key,records in inputs.items() if (seed,name) in records);r=inputs[source][seed,name];assert sha(directories[source]/r['file'])==r['sha256']
                with np.load(directories[source]/r['file']) as z0:
                    pred=z0['prediction'];point=z0['point_prediction'];assert pred.tobytes()==curves['predictions'][j].tobytes() and point.tobytes()==curves['point_predictions'][j].tobytes()
                if source in ['new','credit','distinct','anchor','mixed']:
                    keys=r['metadata']['positive_modes'];mask=np.array([key in keys for key in task['keys']]);nonempty=bool(keys)
                    if new:assert mask.tobytes()==curves['new_masks'][j].tobytes()
                else:
                    oi=oldp['methods'].index(name);mask=old['masks'][oi];nonempty=old['kinds'][oi]=='nonempty_pool'
                if nonempty:
                    ww=w[mask]/w[mask].sum();k=np.sum(ww[:,None,None]*mu[mask],axis=0);tk=np.sum(ww[:,None,None]*second[mask],axis=0)
                else:k=np.tile(pred,(4,1));tk=k*k
                if new:check(k,curves['new_strategies'][j]);check(tk,curves['new_strategy_seconds'][j])
                for grid in p['grids']:
                    sl=slice(None) if grid==257 else slice(None,None,2);pp=pred[sl];rr=rows[seed,name,grid]
                    if not new:
                        inherited_row=dict(inherited_rows[seed,name,grid]);inherited_row['is_new']=False;assert inherited_row==rr;counts['exact_inherited_rows']+=1
                    estimates=dict(mse=float(np.average((pred[sl]-truth[sl])**2)),point_mse=float(np.average((point[sl]-truth[sl])**2)))
                    pair=[]
                    for index,(a,b) in enumerate(p['batch_pairs']):
                        ma,mb=m[a,sl],m[b,sl];ka,kb=k[a,sl],k[b,sl]
                        actual=float(np.average(pp*pp-pp*(ma+mb)+ma*mb));policy=float(np.average(ka*kb-ka*mb-kb*ma+ma*mb))
                        read=float(np.average(pp*pp-pp*(ka+kb)+ka*kb));cross=float(np.average(pp*(ka+kb-ma-mb)-2*ka*kb+ma*kb+mb*ka))
                        mc=float(np.average((tk[a,sl]+tk[b,sl])/2-ka*kb)/2048) if nonempty else 0.
                        bayes=float(np.average((t[a,sl]+t[b,sl])/2-ma*mb));total=float(np.average(pp*pp-pp*(ma+mb)+(t[a,sl]+t[b,sl])/2))
                        values=dict(actual_excess=actual,policy_excess=policy,read_error=read,cross_term=cross,expected_mc=mc,resampled_expected_excess=policy+mc,bayes_variance=bayes,conditional_total=total)
                        for f,v in values.items():check(v,rr['pairs'][index][f]);counts['cross_pair_metrics']+=1
                        pair.append(values)
                    estimates.update({f:float(np.mean([v[f] for v in pair])) for f in pair[0]})
                    for f,v in estimates.items():check(v,rr[f]);counts['task_risks']+=1
                    replay[seed,name,grid]=estimates
                counts['frozen_predictors']+=1
        if (seed-p['seeds'][0]+1)%16==0:print(json.dumps(dict(tasks=seed-p['seeds'][0]+1,seconds=time.perf_counter()-begin)),flush=True)
    for row in json.loads((inp/'methods.json').read_text()):
        rr=[replay[s,row['method'],row['grid']] for s in p['seeds']]
        for f in p['metrics']:check(np.mean([r[f] for r in rr]),row[f])
        check(np.median([r['mse'] for r in rr]),row['median_mse']);check(max(r['mse'] for r in rr),row['worst_mse'])
        originals=[rows[s,row['method'],row['grid']] for s in p['seeds']]
        assert row['empty_pool_tasks']==sum(r['kind']=='empty_pool_fallback' for r in originals)
        assert row['failures']==sum(r['execution_failed'] for r in originals)
        check(np.mean([r['charged_seconds'] for r in originals]),row['mean_charged_seconds']);counts['method_grid_tables']+=1
    ids=np.random.default_rng(262193).integers(0,64,(20000,64))
    for row in json.loads((inp/'paired.json').read_text()):
        f=row['metric'];g=row['grid'];d=np.array([replay[s,p['primary'],g][f]-replay[s,row['control'],g][f] for s in p['seeds']])
        check(d.mean(),row['mean_difference']);check(np.quantile(d[ids].mean(1),[.025,.975]),row['descriptive_ci95'])
        assert row['lower_tasks']==int((d<-1e-12).sum()) and row['equal_tasks']==int((abs(d)<=1e-12).sum()) and row['higher_tasks']==int((d>1e-12).sum())
        counts['paired_intervals']+=1
    assert counts['frozen_predictors']==4416 and counts['paired_intervals']==544 and counts['exact_inherited_rows']==8320
    ans=dict(passed=True,counts=counts,maximum_independent_replay_gap=maxgap,seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),
        scope='Development evidence only; numerical-volume posterior, fixed grids and finite MC; no matched deployment resource claim')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
