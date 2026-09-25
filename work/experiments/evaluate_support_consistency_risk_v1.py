"""324 full old-task conditional-risk panel, including all earlier 297 controls."""
import argparse
from collections import defaultdict
from pathlib import Path
import math
import time
import traceback
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    started=time.perf_counter();base=root/'results/support_consistency_trigger';geo=base/'geometry_v1';gs=read(geo/'summary.json');assert gs['passed']
    for name in ['search_audit_v1','geometry_audit_v1','states_audit_v1']:assert read(base/name/'summary.json')['passed']
    for n,digest in gs['outputs_sha256'].items():assert sha(geo/n)==digest
    ref=root/'results/confirmation_conditional_risk';mom=ref/'moments';ms=read(mom/'summary.json')
    mas=read(ref/'moments_audit/summary.json');ras=read(ref/'reference_audit/summary.json')
    assert ms['passed'] and mas['passed'] and ras['passed']
    assert ms['tasks']==mas['counts']['task_aggregate_files']==ras['complete_up_to_certified_zero_volume']==64
    for folder,s in [(mom,ms),(ref/'moments_audit',mas),(ref/'reference_audit',ras)]:assert sha(folder/'protocol.json')==s['protocol_sha256']
    assert read(ref/'moments_audit/protocol.json')['moments_summary_sha256']==sha(mom/'summary.json')
    assert read(mom/'protocol.json')['reference_audit_sha256']==sha(ref/'reference_audit/summary.json')
    assert sha(mom/'tasks.json')==ms['tasks_sha256'];mt={r['seed']:r for r in read(mom/'tasks.json')}
    fitted={r['seed']:r['support_fit'] for r in read(base/'states_audit_v1/facts.json') if r['policy']=='first_fit'}
    save(out/'protocol.json',dict(source_sha256=sha(Path(__file__)),design_sha256=sha(root/'outputs/ttt-pc-alm-research/324_risk_evaluation_details_v1.md'),
         geometry_summary_sha256=sha(geo/'summary.json'),moment_summary_sha256=sha(mom/'summary.json'),
         search_audit_sha256=sha(base/'search_audit_v1/summary.json'),geometry_audit_sha256=sha(base/'geometry_audit_v1/summary.json'),
         grids=[257,129],batch_pairs=[[0,1],[2,3]],tasks=64,query_targets_accessed=False,resources_matched=False,
         numerical_volume_and_mc_reference=True,posthoc_old_task_diagnosis=True))
    scores=[];pools=[];inputs={};checks=0;maxgap=0.
    for row in read(geo/'tasks.json'):
        t=read(geo/row['file']);seed=t['seed'];r=mt[seed];p=mom/r['file'];assert sha(p)==r['sha256'];inputs[r['file']]=r['sha256']
        with np.load(p,allow_pickle=False) as z:vol,means=z['volumes'],z['means']
        keys=r['keys'];weights=vol/math.fsum(map(float,vol));full=np.einsum('r,rbq->bq',weights,means)
        original=set(t['original_positive_modes']);strong=set(t['prior_positive_modes'])
        sets=dict(original=original,strong_pool=strong)
        sets.update({'old297/'+name:set(ks) for name,ks in t['baseline_methods_297'].items()})
        sets.update({'original+'+name:original|set(m['positive_modes']) for name,m in t['methods'].items()})
        sets.update({'strong+'+name:strong|set(m['positive_modes']) for name,m in t['methods'].items()})
        assert len(sets)==46
        taskpools={};cache={}
        for name,pool in sets.items():
            assert pool and pool<=set(keys);chosen=np.array([k in pool for k in keys]);mass=math.fsum(map(float,weights[chosen]))
            mu=np.einsum('r,rbq->bq',weights*chosen/mass,means);residual=mu-full;taskpools[name]=sorted(pool)
            for grid,idx in [(257,np.arange(257)),(129,np.arange(0,257,2))]:
                pairs=[]
                for a,b in [(0,1),(2,3)]:
                    aa=residual[a,idx];bb=residual[b,idx];value=float(np.mean(aa*bb))
                    scalar=math.fsum(float(x)*float(y) for x,y in zip(aa,bb))/grid
                    gap=abs(value-scalar);maxgap=max(maxgap,gap);assert gap<1e-13;checks+=1;pairs.append(value)
                cache[name,grid]=dict(pairs=pairs,value=math.fsum(pairs)/2,mass=mass)
        for name in sets:
            reference='strong_pool' if name.startswith('strong+') or name=='strong_pool' else 'original'
            for grid in [257,129]:
                value=cache[name,grid];before=cache[reference,grid]
                scores.append(dict(seed=seed,method=name,grid=grid,fitted_trigger=fitted[seed],reference=reference,
                                   mass=value['mass'],added_mass=value['mass']-before['mass'],
                                   added_modes=sorted(sets[name]-sets[reference]),excess_pairs=value['pairs'],conditional_excess=value['value'],
                                   delta_pairs=[a-b for a,b in zip(value['pairs'],before['pairs'])],delta=value['value']-before['value']))
        pools.append(dict(seed=seed,methods=taskpools))
    groups=defaultdict(list)
    for s in scores:
        groups[s['method'],s['grid'],'all'].append(s)
        groups[s['method'],s['grid'],'fitted' if s['fitted_trigger'] else 'fallback'].append(s)
    aggregate=[]
    for (method,grid,stratum),rows in groups.items():
        n=len(rows);aggregate.append(dict(method=method,grid=grid,stratum=stratum,tasks=n,
            conditional_excess=math.fsum(r['conditional_excess'] for r in rows)/n,
            delta=math.fsum(r['delta'] for r in rows)/n,delta_pairs=[math.fsum(r['delta_pairs'][j] for r in rows)/n for j in range(2)],
            mass=math.fsum(r['mass'] for r in rows)/n,added_mass=math.fsum(r['added_mass'] for r in rows)/n,
            added_mode_pairs=sum(len(r['added_modes']) for r in rows),tasks_with_new_modes=sum(bool(r['added_modes']) for r in rows)))
    files={}
    for name,value in [('scores.json',scores),('pools.json',pools),('aggregate.json',aggregate),('moment_input_hashes.json',inputs)]:save(out/name,value);files[name]=sha(out/name)
    files['protocol.json']=sha(out/'protocol.json')
    result=dict(passed=True,tasks=64,methods=46,grid_rows=len(scores),aggregate_rows=len(aggregate),scalar_checks=checks,max_scalar_gap=maxgap,
                seconds=time.perf_counter()-started,outputs_sha256=files,query_targets_accessed=False,resources_matched=False,
                independent_task_gain_established=False,posthoc_old_task_diagnosis=True)
    save(out/'summary.json',result);print(result,flush=True)
    for a in aggregate:
        if a['grid']==257 and a['stratum']=='all' and (a['method'].startswith('original+') or a['method'].startswith('strong+') or a['method'] in ['original','strong_pool','old297/adam_8','old297/nodual_8']):print(a,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],a.out)
    except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
