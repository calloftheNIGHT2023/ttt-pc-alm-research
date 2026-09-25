"""322 sealed-pool conditional risk and mass; no teacher/query target loading."""
import argparse
from collections import defaultdict
from pathlib import Path
import math
import time
import traceback
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


def scalar_product(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))/len(a)


def run(root,out):
    begin=time.perf_counter();source=root/'results/branch_image_chain/geometry_v1';ss=read(source/'summary.json');assert ss['passed']
    for audit in ['audit_v1','geometry_audit_v1']:assert read(root/'results/branch_image_chain'/audit/'summary.json')['passed']
    for n,digest in ss['outputs_sha256'].items():assert sha(source/n)==digest
    ref=root/'results/confirmation_conditional_risk';mom=ref/'moments';ms=read(mom/'summary.json')
    mas=read(ref/'moments_audit/summary.json');ras=read(ref/'reference_audit/summary.json')
    assert ms['passed'] and mas['passed'] and ras['passed']
    assert ms['tasks']==mas['counts']['task_aggregate_files']==ras['complete_up_to_certified_zero_volume']==64
    assert not any(s['phase_accesses_query_targets'] for s in [ms,mas,ras])
    mp=read(mom/'protocol.json');mapr=read(ref/'moments_audit/protocol.json');rap=read(ref/'reference_audit/protocol.json')
    for folder,s in [(mom,ms),(ref/'moments_audit',mas),(ref/'reference_audit',ras)]:assert sha(folder/'protocol.json')==s['protocol_sha256']
    assert sha(mom/'summary.json')==mapr['moments_summary_sha256']
    assert sha(ref/'reference_audit/summary.json')==mp['reference_audit_sha256']
    assert sha(ref/'reference/summary.json')==rap['reference_summary_sha256']
    assert sha(mom/'tasks.json')==ms['tasks_sha256'];tasks={t['seed']:t for t in read(mom/'tasks.json')}
    design=root/'outputs/ttt-pc-alm-research/322_chain_pool_risk_protocol_v1.md'
    save(out/'protocol.json',dict(source_sha256=sha(Path(__file__)),design_sha256=sha(design),
         geometry_summary_sha256=sha(source/'summary.json'),moments_summary_sha256=sha(mom/'summary.json'),
         moments_audit_sha256=sha(ref/'moments_audit/summary.json'),reference_audit_sha256=sha(ref/'reference_audit/summary.json'),
         channels=CHANNELS,grids=[257,129],batch_pairs=[[0,1],[2,3]],query_targets_accessed=False,
         numerical_volume_and_mc_reference=True,resources_matched=False,posthoc_old_task_diagnosis=True))
    scores=[];novel=[];checks=0;maxgap=0.;inputs={}
    for row in read(source/'tasks.json'):
        task=read(source/row['file']);seed=task['seed'];mt=tasks[seed]
        assert sha(mom/mt['file'])==mt['sha256'];inputs[mt['file']]=mt['sha256']
        with np.load(mom/mt['file'],allow_pickle=False) as z:vol=z['volumes'];means=z['means']
        assert means.shape==(len(vol),4,257) and len(mt['keys'])==len(vol) and np.all(vol>0)
        keys=mt['keys'];indices={k:i for i,k in enumerate(keys)};prior=set(task['prior_positive_modes']);assert prior and prior<=set(keys)
        weights=vol/math.fsum(map(float,vol))
        def mixture(pool):
            selected=np.array([k in pool for k in keys]);mass=math.fsum(map(float,weights[selected]));assert mass>0
            mean=np.einsum('k,kbq->bq',weights*selected/mass,means);return mass,mean
        total=np.einsum('k,kbq->bq',weights,means);oldmass,oldmu=mixture(prior)
        for name in CHANNELS:
            positive=set(task['methods'][name]['positive_modes']);assert positive<=set(keys);added=positive-prior
            if added:
                addedmass,addedmu=mixture(added);mass,newmu=mixture(prior|positive);alpha=addedmass/(oldmass+addedmass)
                assert np.max(abs(newmu-((1-alpha)*oldmu+alpha*addedmu)))<1e-13
                for mode in sorted(added):
                    index=indices[mode];gv=task['geometry'][mode]['volume'];rv=float(vol[index])
                    rel=abs(gv-rv)/rv if gv is not None else None
                    novel.append(dict(seed=seed,channel=name,mode=mode,reference_volume=rv,selector_geometry_volume=gv,
                                      relative_volume_difference=rel,posterior_mass=float(weights[index]),
                                      total_support_volume=math.fsum(map(float,vol)),reference_file=mt['file']))
            else:addedmass=alpha=0.;mass=oldmass;newmu=oldmu
            max_readout=float(np.max(abs(newmu-oldmu)));assert max_readout<=alpha+1e-13
            for grid in [257,129]:
                idx=np.arange(257) if grid==257 else np.arange(0,257,2);pairs=[]
                for a,b in [(0,1),(2,3)]:
                    oa=oldmu[a,idx]-total[a,idx];ob=oldmu[b,idx]-total[b,idx]
                    na=newmu[a,idx]-total[a,idx];nb=newmu[b,idx]-total[b,idx]
                    da=newmu[a,idx]-oldmu[a,idx];db=newmu[b,idx]-oldmu[b,idx]
                    before=float(np.mean(oa*ob));after=float(np.mean(na*nb));delta=after-before
                    independent=scalar_product(na,nb)-scalar_product(oa,ob)
                    identity=scalar_product(da,ob)+scalar_product(oa,db)+scalar_product(da,db)
                    gap=max(abs(delta-independent),abs(delta-identity));maxgap=max(maxgap,gap);assert gap<1e-13
                    pairs.append(dict(pair=[a,b],old_excess=before,new_excess=after,delta=delta,independent_delta=independent,identity_delta=identity));checks+=2
                scores.append(dict(seed=seed,channel=name,grid=grid,new_modes=sorted(added),old_mass=oldmass,new_mass=mass,
                                   added_mass=addedmass,mixture_alpha=alpha,readout_supnorm=max_readout,
                                   numerical_squared_loss_change_bound=2*alpha,pairs=pairs,
                                   delta=math.fsum(p['delta'] for p in pairs)/2))
    groups=defaultdict(list)
    for row in scores:groups[row['channel'],row['grid']].append(row)
    aggregate=[]
    for (channel,grid),group in groups.items():
        assert len(group)==64
        aggregate.append(dict(channel=channel,grid=grid,tasks=64,tasks_with_new_modes=sum(bool(r['new_modes']) for r in group),
                              added_mode_pairs=sum(len(r['new_modes']) for r in group),
                              mean_added_mass=math.fsum(r['added_mass'] for r in group)/64,
                              mean_delta=math.fsum(r['delta'] for r in group)/64,
                              pair_mean_deltas=[math.fsum(r['pairs'][i]['delta'] for r in group)/64 for i in range(2)],
                              mean_loss_change_bound=math.fsum(r['numerical_squared_loss_change_bound'] for r in group)/64))
    files={}
    for name,data in [('scores.json',scores),('aggregate.json',aggregate),('novel_regions.json',novel),('moment_input_hashes.json',inputs)]:
        save(out/name,data);files[name]=sha(out/name)
    files['protocol.json']=sha(out/'protocol.json')
    result=dict(passed=True,tasks=64,channels=6,grid_rows=len(scores),scalar_identity_checks=checks,max_identity_gap=maxgap,
                aggregate=aggregate,novel_regions=novel,seconds=time.perf_counter()-begin,outputs_sha256=files,
                query_targets_accessed=False,posthoc_old_task_diagnosis=True,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print({k:v for k,v in result.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
