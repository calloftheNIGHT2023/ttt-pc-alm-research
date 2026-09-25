"""302: seal twelve BP-free support event selections, then inspect certificate overlap."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from diagnose_gradient_flat_split_states_v1 import tent,residuals
from diagnose_counterfactual_conditional_risk_v1 import read,save,sha

RULES=['boundary_entry','forward_stasis','boundary_and_forward_stasis','primal_stasis']
POLICIES=['all','first','onset']


def forward(b,x):
    y=np.broadcast_to(x,(len(b),len(x)))
    for j in range(4):y=tent(y+b[:,j,None])
    return y


def run(root,out):
    raw=root/'results/certificate_activity_attribution/development'
    originals={r['seed']:r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    manifest=read(raw/'before_evaluation_manifest.json')
    assert sha(raw/'rows.json')==manifest['rows_sha256']
    names=['census_support_boundary_events_v1.py','diagnose_gradient_flat_split_states_v1.py',
           'diagnose_counterfactual_conditional_risk_v1.py']
    sources={n:sha(root/'work/experiments'/n) for n in names}
    save(out/'protocol.json',dict(source_sha256=sources,rules=RULES,policies=POLICIES,
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/302_support_event_coverage_protocol.md'),
        threshold=1e-8,support_error=.001001,split_residual=1e-6,bound=.12,
        selected_using_gradients=False,selected_using_certificates=False,query_targets_accessed=False,
        reference_accessed=False,new_adaptation=False))
    rows=[]; totals=Counter(); handoffs=0
    for seed in range(5910000,5910064):
        path=raw/originals[seed]['file'];assert sha(path)==originals[seed]['sha256']
        with np.load(path,allow_pickle=False) as z:
            x,v=z['x_observed'],z['v_observed']
            traces={p:{n:z[f'{p}_{n}'] for n in ['b','h']} for p in ['prefix','anchor']}
        assert np.array_equal(traces['prefix']['b'][-1,:1],traces['anchor']['b'][0])
        assert np.array_equal(traces['prefix']['h'][-1,:,:1],traces['anchor']['h'][0])
        handoffs+=1
        locations={r+'__'+p:[] for r in RULES for p in POLICIES}
        for phase_id,phase in enumerate(['prefix','anchor']):
            bb,hh=traces[phase]['b'],traces[phase]['h'];n=bb.shape[1]
            seen={r:np.zeros(n,dtype=bool) for r in RULES}
            previous={r:np.zeros(n,dtype=bool) for r in RULES}
            for t in range(32):
                b,h=bb[t],hh[t]
                y=forward(b,x)
                eligibility=(np.max(abs(y-v),axis=1)>.001001)&(np.max(abs(residuals(b,h,x)),axis=(0,2))>1e-6)
                if t==0 and phase_id==0:
                    flags={r:np.zeros(n,dtype=bool) for r in RULES}
                else:
                    prevb=bb[t-1] if t>0 else traces['prefix']['b'][-2,:1]
                    prevh=hh[t-1] if t>0 else traces['prefix']['h'][-2,:,:1]
                    face=(b==.12).astype(int)-(b==-.12).astype(int)
                    prevface=(prevb==.12).astype(int)-(prevb==-.12).astype(int)
                    flat=np.max(abs(y-forward(prevb,x)),axis=1)<=1e-8
                    joint=np.maximum(np.max(abs(b-prevb),axis=1),np.max(abs(h-prevh),axis=(0,2)))<=1e-8
                    flags={'boundary_entry':np.any((face!=0)&(face!=prevface),axis=1),
                        'forward_stasis':flat,'boundary_and_forward_stasis':np.any(face!=0,axis=1)&flat,
                        'primal_stasis':joint}
                    flags={r:f&eligibility for r,f in flags.items()}
                for rule,flag in flags.items():
                    choices={'all':flag,'first':flag&~seen[rule],'onset':flag&~previous[rule]}
                    for policy,mask in choices.items():
                        for origin in np.flatnonzero(mask):locations[rule+'__'+policy].append([phase_id,t+1,int(origin)])
                    seen[rule]|=flag;previous[rule]=flag.copy()
        for name,locs in locations.items():totals[name]+=len(locs)
        rows.append(dict(seed=seed,locations=locations,source_sha256=sha(path)))
    save(out/'selections.json',rows)
    seal={n:sha(out/n) for n in ['protocol.json','selections.json']}
    save(out/'before_certificate_manifest.json',dict(files_sha256=seal,certificate_accessed=False,
        gradient_accessed=False,query_targets_accessed=False,reference_accessed=False))
    print(dict(phase='support_events_sealed',counts=dict(totals)),flush=True)
    # Selection is now immutable; read only the old independent mechanism labels.
    audit=root/'results/projected_stationarity/audit_v1'
    cs=read(audit/'summary.json');assert cs['passed'] and cs['certified_positions']==24
    for n,d in cs['outputs_sha256'].items():assert sha(audit/n)==d,n
    certs=read(audit/'certificates.json')
    labels={(r['seed'],*r['location']) for r in certs if r['smooth_box_local_minimum']}
    near={(r['seed'],*r['location']) for r in certs}
    old=root/'results/stagnant_local_states/development_v2'
    oldselect={r['seed']:r for r in read(old/'tasks.json')}
    aggregate=[]
    for rule in RULES:
        for policy in POLICIES:
            name=rule+'__'+policy
            all_selected={(r['seed'],*loc) for r in rows for loc in r['locations'][name]}
            covered=all_selected&labels
            aggregate.append(dict(rule=rule,policy=policy,selected_positions=len(all_selected),
                selected_tasks=len({v[0] for v in all_selected}),mean_positions=len(all_selected)/64,
                known_certified_overlap=len(covered),certified_tasks=len({v[0] for v in covered}),
                near_stationary_overlap=len(all_selected&near),covered_certificates=sorted(covered)))
    chronology=[]
    for seed,phase,step,origin in sorted(labels):
        entry=dict(seed=seed,location=[phase,step,origin])
        for group in ['dwell','parameter_stall']:
            matched=[t for p,t,o in oldselect[seed]['locations'][group] if p==phase and o==origin]
            assert len(matched)<=1
            entry[group+'_first_step']=matched[0] if matched else None
        chronology.append(entry)
    save(out/'aggregates.json',aggregate);save(out/'chronology.json',chronology)
    for n,d in seal.items():assert sha(out/n)==d,n
    for n,d in sources.items():assert sha(root/'work/experiments'/n)==d,n
    summary=dict(passed=True,tasks=64,rules=12,anchor_handoffs_verified=handoffs,
        gradient_accessed=False,selection_before_certificate_access=True,query_targets_accessed=False,reference_accessed=False,
        new_adaptation=False,new_confirmation=False,core_research_goal_complete=False,
        exact_certificate_summary_sha256=sha(audit/'summary.json'),
        outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'summary.json',summary)
    print(aggregate,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();out=root/'results/support_boundary_events/development_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise
