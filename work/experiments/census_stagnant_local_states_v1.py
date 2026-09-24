"""299: all saved support states; no new adaptation, targets, or posterior."""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
import batched_bp_discovery as bp
from diagnose_gradient_flat_split_states_v1 import mode_list,residuals
from diagnose_counterfactual_conditional_risk_v1 import read,save,sha


def run(root,out):
    started=time.perf_counter()
    raw=root/'results/certificate_activity_attribution/development'
    fg=root/'results/counterfactual_credit_branching/first_global_support_v1'
    fs=read(fg/'summary.json')
    assert fs['passed'] and fs['tasks']==64
    for name,digest in fs['outputs_sha256'].items(): assert sha(fg/name)==digest,name
    firsts={r['seed']:r for r in read(fg/'rows.json')}
    old={r['seed']:r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    source_names=['census_stagnant_local_states_v1.py','batched_bp_discovery.py',
                  'diagnose_gradient_flat_split_states_v1.py','diagnose_counterfactual_conditional_risk_v1.py']
    sources={n:sha(root/'work/experiments'/n) for n in source_names}
    save(out/'protocol.json',dict(seeds=list(range(5910000,5910064)),source_sha256=sources,
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/299_stagnant_state_census_protocol.md'),
        first_global_gate_sha256=sha(fg/'summary.json'),mode_dwell=4,parameter_change=1e-8,
        split_residual=1e-6,support_error=.001001,diagnostic_gradient_tolerance=1e-8,
        global_gradient_is_offline_diagnostic_only=True,query_targets_accessed=False,reference_accessed=False))
    total=Counter()
    records=[]
    metrics=[]
    for seed in range(5910000,5910064):
        path=raw/old[seed]['file']
        assert sha(path)==old[seed]['sha256']==firsts[seed]['source_file_sha256']
        with np.load(fg/firsts[seed]['file'],allow_pickle=False) as z:
            first=set(tuple(v) for v in z['counterfactual_locations'].tolist())
        with np.load(path,allow_pickle=False) as z:
            x,v=z['x_observed'],z['v_observed']
            traces={phase:{name:z[f'{phase}_{name}'] for name in ['b','h','u']} for phase in ['prefix','anchor']}
        counts=Counter()
        selected={'dwell':[],'parameter_stall':[],'float_zero_gradient_nonfeasible':[]}
        prefix_age_end=None
        task_metrics=[]
        for phase_id,phase in enumerate(['prefix','anchor']):
            trace=traces[phase]
            b,h,u=trace['b'],trace['h'],trace['u']
            rcount=b.shape[1]
            modes=[mode_list(bs,x) for bs in b]
            ages=np.ones(rcount,dtype=int) if phase_id==0 else prefix_age_end[:1].copy()
            triggered={'dwell':set(),'parameter_stall':set()}
            for t in range(32):
                if t>0:
                    ages=np.array([age+1 if a==c else 1 for age,a,c in zip(ages,modes[t-1],modes[t])])
                if t>0: change=np.max(abs(b[t]-b[t-1]),axis=1)
                elif phase_id==1: change=np.max(abs(traces['prefix']['b'][-1,:1]-traces['prefix']['b'][-2,:1]),axis=1)
                else: change=np.full(rcount,np.inf)
                split=np.max(abs(residuals(b[t],h[t],x)),axis=(0,2))
                unorm=np.max(abs(u[t]),axis=(0,2))
                _,res,jac,raw=bp.evaluate(b[t],x,v)
                gradient=np.einsum('rni,rn->ri',jac,res,optimize=False)/len(x)
                gradnorm=np.max(abs(gradient),axis=1)
                error=np.max(abs(raw),axis=1)
                nonfeasible=error>.001001
                hasres=split>1e-6
                masks={'dwell':(ages>=4)&nonfeasible&hasres,
                       'parameter_stall':(change<=1e-8)&nonfeasible&hasres,
                       'float_zero_gradient_nonfeasible':(gradnorm==0)&nonfeasible,
                       'small_gradient_nonfeasible':(gradnorm<=1e-8)&nonfeasible}
                counts['states']+=rcount
                counts['nonfeasible']+=int(nonfeasible.sum())
                counts['split_nonzero']+=int(hasres.sum())
                counts['zero_multiplier']+=int(np.count_nonzero(unorm==0))
                for name,mask in masks.items(): counts[name]+=int(mask.sum())
                for origin in range(rcount):
                    location=(phase_id,t+1,origin)
                    isfirst=location in first
                    counts['firstglobal']+=int(isfirst)
                    if isfirst:
                        counts['firstglobal_nonfeasible']+=int(nonfeasible[origin])
                    for name,mask in masks.items():
                        counts['firstglobal_'+name]+=int(isfirst and mask[origin])
                    for name in ['dwell','parameter_stall']:
                        if masks[name][origin] and origin not in triggered[name]:
                            triggered[name].add(origin)
                            selected[name].append(list(location))
                    if masks['float_zero_gradient_nonfeasible'][origin]:
                        selected['float_zero_gradient_nonfeasible'].append(list(location))
                    task_metrics.append([phase_id,t+1,origin,float(error[origin]),float(split[origin]),
                        float(unorm[origin]),float(change[origin]),int(ages[origin]),float(gradnorm[origin]),int(isfirst)])
            if phase_id==0:
                prefix_age_end=np.array([age+1 if a==c else 1 for age,a,c in zip(ages,modes[31],modes[32])])
        assert counts['states']==1088 and counts['firstglobal']==len(first)
        for name in ['dwell','parameter_stall']:
            counts['first_'+name]=len(selected[name])
        total.update(counts)
        records.append(dict(seed=seed,counts=dict(counts),locations=selected,source_sha256=old[seed]['sha256']))
        metrics.append(np.array(task_metrics))
    assert total['states']==69632 and total['firstglobal']==2427
    save(out/'tasks.json',records)
    with (out/'metrics.npz').open('xb') as stream:
        np.savez_compressed(stream,values=np.array(metrics),seeds=np.arange(5910000,5910064))
    save(out/'columns.json',['phase','step','origin','support_max_error','split_max','multiplier_max',
                             'parameter_last_change','mode_dwell','gradient_max','firstglobal'])
    for name,digest in sources.items(): assert sha(root/'work/experiments'/name)==digest,name
    summary=dict(passed=True,tasks=64,counts=dict(total),query_targets_accessed=False,reference_accessed=False,
        gradient_zero_is_float_not_rational_proof=True,new_adaptation=False,core_research_goal_complete=False,
        seconds=time.perf_counter()-started,outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='outputs_sha256'},flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve()
    out=root/'results/stagnant_local_states/development_v1'
    out.mkdir(parents=True,exist_ok=False)
    try: run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True))
        raise


if __name__=='__main__':
    main()
