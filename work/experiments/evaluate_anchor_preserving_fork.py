"""273 frozen development evaluation, all54 methods with exact inheritance."""
import argparse,json,time
from pathlib import Path
import numpy as np
from analyze_recovered_online_comparison import forward
from run_multiplier_fixed_point_screen import sha,dump
from evaluate_mixed_restart_credit import CONDITIONAL,METRICS,PAIRED,cross_estimates


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/anchor_preserving_fork';inp=base/'development';gate=base/'audit';out=base/'evaluation';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    gg=json.loads((gate/'summary.json').read_text());assert gg['passed'] and gg['may_evaluate_frozen_development_predictions']
    gp=json.loads((gate/'protocol.json').read_text());hashes=dict(gp['source_sha256'])
    for n in [Path(__file__).name,'evaluate_mixed_restart_credit.py']:hashes[n]=sha(src/n)
    for n,h in hashes.items():assert sha(src/n)==h,n
    p0=json.loads((inp/'protocol.json').read_text());before=json.loads((inp/'before_evaluation_manifest.json').read_text());ss=json.loads((inp/'summary.json').read_text());assert sha(inp/'before_evaluation_manifest.json')==ss['before_evaluation_manifest_sha256']
    for n in ['protocol','rows']:assert sha(inp/f'{n}.json')==before[f'{n}_sha256']
    new={(r['seed'],r['method']):r for r in json.loads((inp/'rows.json').read_text())}
    previous=root/'results/fixed_restart_credit/evaluation';prevs=json.loads((previous/'summary.json').read_text());assert prevs['passed']
    assert json.loads((previous.parent/'evaluation_audit/summary.json').read_text())['passed']
    for n,h in prevs['outputs_sha256'].items():assert sha(previous/n)==h
    prevp=json.loads((previous/'protocol.json').read_text());oldnames=prevp['methods'];newnames=[r['name'] for r in p0['configs']];names=newnames+oldnames;assert len(names)==len(set(names))==54
    oldrisk={(r['seed'],r['method'],r['grid']):r for r in json.loads((previous/'rows.json').read_text())};prevfiles=json.loads((previous/'files.json').read_text())
    posterior=root/'results/confirmation_conditional_risk';taskmoments={r['seed']:r for r in json.loads((posterior/'moments/tasks.json').read_text())}
    p=dict(source_sha256=hashes,geometry_gate_sha256=sha(gate/'summary.json'),new_manifest_sha256=sha(inp/'before_evaluation_manifest.json'),
        previous_evaluation_sha256=sha(previous/'summary.json'),previous_audit_sha256=sha(previous.parent/'evaluation_audit/summary.json'),
        design_sha256=p0['design_sha256'],seeds=p0['seeds'],methods=names,new_methods=newnames,primary=p0['primary'],grids=[257,129],metrics=METRICS,
        paired_metrics=PAIRED,bootstrap_seed=262193,bootstrap_replicates=20000,batch_pairs=[[0,1],[2,3]],
        scope='Development on already evaluated64 tasks; targeted anchor hypothesis after271; all50 previous methods retained, trace elapsed is not a matched resource comparison')
    dump(out/'protocol.json',p);rows=[];files={};begin=time.perf_counter();old_gap=0.
    for seed in p['seeds']:
        task=taskmoments[seed];assert sha(posterior/'moments'/task['file'])==task['sha256']
        with np.load(posterior/'moments'/task['file']) as z:q=z['q'].copy();w=z['volumes']/z['volumes'].sum();mu=z['means'].copy();second=z['seconds'].copy()
        fn=f'{seed}_evaluation_curves.npz';assert sha(previous/fn)==prevfiles[fn]
        with np.load(previous/fn) as z:
            full=z['full_means'].copy();fullsecond=z['full_second_moments'].copy();oldpred=z['predictions'].copy();oldpoint=z['point_predictions'].copy()
            assert z['methods'].tolist()==oldnames and q.tobytes()==z['q'].tobytes()
        truth=forward(q,np.random.default_rng(seed).uniform(-.12,.12,(1,4)))[0]
        predictions=[];point_predictions=[];strategies=[];second_strategies=[];masks=[]
        for name in names:
            is_new=name in newnames
            if is_new:
                r=new[seed,name];assert sha(inp/r['file'])==r['sha256']==before['prediction_files'][r['file']]
                with np.load(inp/r['file']) as z:pred=z['prediction'].copy();point=z['point_prediction'].copy();assert q.tobytes()==z['q_observed'].tobytes()
                keys=r['metadata']['positive_modes'];assert set(keys)<=set(task['keys']);mask=np.isin(task['keys'],keys);mass=float(w[mask].sum())
                if keys:
                    ww=w*mask/mass;k=np.einsum('k,kbq->bq',ww,mu);tk=np.einsum('k,kbq->bq',ww,second);kind='nonempty_pool'
                else:k=np.tile(pred,(4,1));tk=k*k;kind='empty_pool_fallback'
                strategies.append(k);second_strategies.append(tk);masks.append(mask)
            else:
                oi=oldnames.index(name);pred=oldpred[oi];point=oldpoint[oi]
            predictions.append(pred);point_predictions.append(point)
            for grid in p['grids']:
                sl=slice(None) if grid==257 else slice(None,None,2);mse=float(np.mean((pred[sl]-truth[sl])**2));point_mse=float(np.mean((point[sl]-truth[sl])**2))
                if is_new:
                    values,pairs=cross_estimates(pred,full,fullsecond,k,tk,bool(keys),grid)
                    row=dict(seed=seed,method=name,grid=grid,is_new=True,kind=kind,posterior_mass=mass,mse=mse,point_mse=point_mse,**values,pairs=pairs,
                        charged_seconds=r['metadata']['charged_complete_seconds'],timing_scope='trace_enabled_development',execution_failed=r['metadata']['execution_failed'])
                else:
                    ref=oldrisk[seed,name,grid];gap=max(abs(mse-ref['mse']),abs(point_mse-ref['point_mse']));assert gap<1e-12;old_gap=max(old_gap,gap)
                    row=dict(ref);row['is_new']=False
                rows.append(row)
        filename=f'{seed}_evaluation_curves.npz';np.savez_compressed(out/filename,q=q,methods=np.array(names),new_methods=np.array(newnames),predictions=np.array(predictions),point_predictions=np.array(point_predictions),
            new_strategies=np.array(strategies),new_strategy_seconds=np.array(second_strategies),new_masks=np.array(masks),full_means=full,full_second_moments=fullsecond)
        files[filename]=sha(out/filename)
    bykey={(r['seed'],r['method'],r['grid']):r for r in rows};methods=[];paired=[];ids=np.random.default_rng(262193).integers(0,64,(20000,64))
    for grid in p['grids']:
        for name in names:
            rr=[bykey[s,name,grid] for s in p['seeds']]
            methods.append(dict(method=name,grid=grid,is_new=name in newnames,**{f:float(np.mean([r[f] for r in rr])) for f in METRICS},
                median_mse=float(np.median([r['mse'] for r in rr])),worst_mse=max(r['mse'] for r in rr),empty_pool_tasks=sum(r['kind']=='empty_pool_fallback' for r in rr),
                failures=sum(r['execution_failed'] for r in rr),mean_charged_seconds=float(np.mean([r['charged_seconds'] for r in rr])),timing_scope=rr[0]['timing_scope']))
            if name==p['primary']:continue
            for f in PAIRED:
                diff=np.array([bykey[s,p['primary'],grid][f]-bykey[s,name,grid][f] for s in p['seeds']]);paired.append(dict(control=name,grid=grid,metric=f,mean_difference=float(diff.mean()),
                    descriptive_ci95=np.quantile(diff[ids].mean(1),[.025,.975]).tolist(),lower_tasks=int((diff<-1e-12).sum()),equal_tasks=int((abs(diff)<=1e-12).sum()),higher_tasks=int((diff>1e-12).sum())))
    for n,data in [('rows.json',rows),('methods.json',methods),('paired.json',paired),('files.json',files)]:dump(out/n,data)
    ans=dict(passed=True,tasks=64,methods=54,new_predictions=256,rows=len(rows),paired_intervals=len(paired),old_query_replay_maximum_gap=old_gap,
        seconds=time.perf_counter()-begin,outputs_sha256={n:sha(out/n) for n in ['protocol.json','rows.json','methods.json','paired.json','files.json']},
        scope=p['scope'],next='Independent scalar teacher, expanded risk, inherited rows and paired intervals audit before interpretation')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
