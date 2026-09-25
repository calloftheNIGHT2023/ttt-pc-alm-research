"""325 posthoc value beyond all nondual controls, with exact ranking witnesses."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path
import math
import time
import traceback
import numpy as np
from independent_branch_image_search_v1 import IndependentSearch
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

CANDIDATES=[p+'/'+n for p in ['first_fit','uniform_state'] for n in ['dual','dual_plus_residual']]
CONTROLS=[p+'/'+n for p in ['first_fit','uniform_state'] for n in ['residual','bp','random_sign','zero']]
CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


def run(root,out):
    start=time.perf_counter();base=root/'results/support_consistency_trigger'
    assert read(base/'risk_audit_v1/summary.json')['passed']
    geo=base/'geometry_v1';gs=read(geo/'summary.json');assert gs['passed']
    for n,d in gs['outputs_sha256'].items():assert sha(geo/n)==d
    dev=base/'development_v1';ss=read(dev/'summary.json');manifest=read(dev/'before_geometry_manifest.json')
    assert sha(dev/'before_geometry_manifest.json')==ss['manifest_sha256']
    for n,d in manifest['files_sha256'].items():assert sha(dev/n)==d
    source={(r['seed'],r['policy']):read(dev/r['file']) for r in read(dev/'rows.json')}
    mom=root/'results/confirmation_conditional_risk/moments';ms=read(mom/'summary.json');assert sha(mom/'tasks.json')==ms['tasks_sha256']
    tasks={r['seed']:r for r in read(mom/'tasks.json')};rows=[];witnesses=[];pools=[];identity_checks=0;maxgap=0.
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'independent_branch_image_search_v1.py']},
         design_sha256=sha(root/'outputs/ttt-pc-alm-research/325_credit_exclusive_value_protocol_v1.md'),
         risk_audit_sha256=sha(base/'risk_audit_v1/summary.json'),geometry_summary_sha256=sha(geo/'summary.json'),
         proposals_summary_sha256=sha(dev/'summary.json'),query_targets_accessed=False,posthoc_old_task_diagnosis=True,resources_matched=False))
    for tr in read(geo/'tasks.json'):
        t=read(geo/tr['file']);seed=t['seed'];mt=tasks[seed];p=mom/mt['file'];assert sha(p)==mt['sha256']
        with np.load(p,allow_pickle=False) as z:vol,means=z['volumes'],z['means']
        weights=vol/math.fsum(map(float,vol));keys=mt['keys'];index={k:i for i,k in enumerate(keys)}
        full=np.einsum('k,kbq->bq',weights,means)
        control=set(t['prior_positive_modes'])|set().union(*(set(t['methods'][n]['positive_modes']) for n in CONTROLS))
        def mixture(pool):
            mask=np.array([k in pool for k in keys]);mass=math.fsum(map(float,weights[mask]));return mass,np.einsum('k,kbq->bq',weights*mask/mass,means)
        oldmass,old=mixture(control);taskpool=dict(seed=seed,control=sorted(control),candidates={})
        for name in CANDIDATES:
            positive=set(t['methods'][name]['positive_modes']);added=positive-control;newmass,new=mixture(control|positive)
            taskpool['candidates'][name]=sorted(control|positive)
            for grid,idx in [(257,np.arange(257)),(129,np.arange(0,257,2))]:
                pairs=[]
                for a,b in [(0,1),(2,3)]:
                    oa,ob=(old-full)[a,idx],(old-full)[b,idx];na,nb=(new-full)[a,idx],(new-full)[b,idx]
                    da,db=(new-old)[a,idx],(new-old)[b,idx]
                    delta=float(np.mean(na*nb)-np.mean(oa*ob))
                    identity=math.fsum(float(x) for x in da*ob+oa*db+da*db)/grid
                    maxgap=max(maxgap,abs(delta-identity));assert abs(delta-identity)<1e-13;identity_checks+=1;pairs.append(delta)
                rows.append(dict(seed=seed,method=name,grid=grid,added_modes=sorted(added),added_mass=newmass-oldmass,
                                 delta_pairs=pairs,delta=math.fsum(pairs)/2))
            if not added:continue
            rec=source[seed,name.split('/')[0]];original=np.array(list(bytes.fromhex(rec['original_mode']))).reshape(4,4)
            for mode in sorted(added):
                target=np.array(list(bytes.fromhex(mode))).reshape(4,4);distance=int(np.sum(target!=original));details={}
                for channel in CHANNELS:
                    checker=IndependentSearch(np.array(rec['x_observed']),np.array(rec['v_observed']),np.array(rec['credits'][channel]),original)
                    value=checker.fixed(target);assert value is not None and value<=0
                    result=rec['results'][channel];matches=[r for r in result['proposals'] if r['mode']==mode]
                    shell=[r for r in result['proposals'] if r['hamming']==distance]
                    minimum=result['minimum_nonexcluded_hamming']
                    if matches:reason='selected';assert F(matches[0]['lower'])==value
                    elif minimum is None or not minimum<=distance<=minimum+2:reason='outside_window'
                    else:
                        assert len(shell)==8 and (value,mode)>(F(shell[-1]['lower']),shell[-1]['mode']);reason='ranked_out'
                    details[channel]=dict(lower=str(value),status=reason,selected_rank=matches[0]['rank'] if matches else None,
                                          cutoff_lower=shell[-1]['lower'] if shell else None,cutoff_mode=shell[-1]['mode'] if shell else None)
                    checker.clear();checker.best=None
                witnesses.append(dict(seed=seed,method=name,mode=mode,support_fit=rec['support_fit'],distance=distance,
                                      posterior_mass=float(weights[index[mode]]),credits=details))
        pools.append(taskpool)
    grouped=defaultdict(list)
    for r in rows:grouped[r['method'],r['grid']].append(r)
    aggregate=[]
    for (name,grid),group in grouped.items():
        assert len(group)==64;worst=max(group,key=lambda r:abs(r['delta']));totalabs=math.fsum(abs(r['delta']) for r in group)
        aggregate.append(dict(method=name,grid=grid,tasks=64,mean_delta=math.fsum(r['delta'] for r in group)/64,
            delta_pairs=[math.fsum(r['delta_pairs'][j] for r in group)/64 for j in range(2)],
            added_mode_pairs=sum(len(r['added_modes']) for r in group),tasks_with_added_modes=sum(bool(r['added_modes']) for r in group),
            mean_added_mass=math.fsum(r['added_mass'] for r in group)/64,
            largest_absolute_contributor=worst['seed'] if totalabs else None,largest_absolute_contribution_fraction=abs(worst['delta'])/totalabs if totalabs else 0.,
            leave_largest_out_mean=math.fsum(r['delta'] for r in group if r is not worst)/63))
    files={}
    for name,value in [('scores.json',rows),('pools.json',pools),('witnesses.json',witnesses),('aggregate.json',aggregate)]:save(out/name,value);files[name]=sha(out/name)
    files['protocol.json']=sha(out/'protocol.json')
    result=dict(passed=True,rows=len(rows),identity_checks=identity_checks,max_identity_gap=maxgap,witnesses=len(witnesses),
                aggregate=aggregate,seconds=time.perf_counter()-start,outputs_sha256=files,query_targets_accessed=False,
                resources_matched=False,independent_task_gain_established=False,posthoc_old_task_diagnosis=True)
    save(out/'summary.json',result);print({k:v for k,v in result.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],a.out)
    except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
