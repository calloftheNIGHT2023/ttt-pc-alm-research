"""318 post-seal mode and exact point geometry, with state-correction controls."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
import complete_credit_mode_geometry_v1 as geometry
from affine_root_validation_v1 import state,forward_metrics
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha,certify_label,describe,hexmode

FAMILIES=['alm_keep','alm_reset','nodual']
NAMES=[f+'_'+variant for f in FAMILIES for variant in ['q_exact','q_float','exit_exact','exit_float']]


def run(root,out):
    begin=time.perf_counter();source=root/'results/primal_stasis_escape/development_v1'
    parent=root/'results/invariant_dual_drift/development_v1';old_dir=root/'results/paired_primal_dual_predictor/geometry_v1'
    audit=root/'results/primal_stasis_escape/audit_v1/summary.json';assert read(audit)['passed']
    ss=read(source/'summary.json');assert ss['passed']
    manifest=read(source/'before_geometry_manifest.json')
    assert sha(source/'before_geometry_manifest.json')==ss['manifest_sha256']
    for n,digest in manifest['files_sha256'].items():assert sha(source/n)==digest
    oldsummary=read(old_dir/'summary.json');assert oldsummary['passed']
    for n,digest in oldsummary['outputs_sha256'].items():assert sha(old_dir/n)==digest
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in
                                                [Path(__file__).name,'complete_credit_mode_geometry_v1.py','evaluate_complete_credit_mode_geometry_v1.py']},
                                 support_summary_sha256=sha(source/'summary.json'),support_audit_sha256=sha(audit),
                                 prior_geometry_summary_sha256=sha(old_dir/'summary.json'),methods=NAMES,tasks=64,
                                 query_targets_accessed=False,posterior_moments_accessed=False,resources_matched=False,
                                 q_control='Same corrected entry state, without dual-drift progression.',
                                 global_lp_scope='Post-seal evaluator only, never candidate credit.'))
    grouped=defaultdict(list)
    for entry in read(source/'rows.json'):grouped[entry['seed']].append(entry)
    counts=Counter();tasks=[];files={};points=[]
    for previous in read(old_dir/'tasks.json'):
        seed=previous['seed'];old=read(old_dir/previous['file'])
        x,v=np.array(old['x_observed']),np.array(old['v_observed']);original=set(old['original_positive_modes'])
        prior=original|set(old['all_308_309_positive_modes'])|set().union(*(set(m['positive_modes']) for m in old['methods'].values()))
        candidates={name:[] for name in NAMES}
        for entry in grouped[seed]:
            payload=read(source/entry['file']);r=payload['result'];line=read(parent/payload['source_file'])
            assert list(map(F,line['x']))==list(map(F,x)) and list(map(F,line['v']))==list(map(F,v))
            if r['applicable']:
                q=list(map(F,line['proposal']['q']));n=len(x);d=len(q)//(1+2*n)
                for backend,qq in [('exact',q),('float',[F(float(z)) for z in q])]:
                    metrics=forward_metrics(state(qq,d,x,v));name=entry['family']+'_q_'+backend
                    candidates[name].append(dict(b=qq[:d],mode=hexmode(metrics['mode']),location=entry['location'],source_file=entry['file']))
            if r['status']=='first_primal_exit_found':
                for backend,key,metrics in [('exact','exit_output',r['exit_metrics']),('float','floating_exit_output',payload['floating_exit_metrics'])]:
                    b=list(map(F,(r if backend=='exact' else payload)[key][:4]));name=entry['family']+'_exit_'+backend
                    candidates[name].append(dict(b=b,mode=hexmode(metrics['mode']),location=entry['location'],source_file=entry['file']))
        modes={n:{c['mode'] for c in p} for n,p in candidates.items()};classified={};matrices={}
        for mode in sorted(set().union(*modes.values())):
            if mode in old['geometry']:
                answer=old['geometry'][mode];counts['reused_classifications']+=1
            else:
                answer=geometry.classify_mode(x,v,mode);counts['new_classifications']+=1;counts['new_lp_calls']+=answer['lp_calls']
            _,a,rhs,_,_=geometry.matrices(x,v,mode);matrices[mode]=(a,rhs)
            counts['certificate_rechecks']+=certify_label(a,rhs,answer)
            classified[mode]=answer;counts[answer['classification']]+=1
        methods={name:describe(modes[name],classified,original) for name in NAMES};comparisons={}
        for family in FAMILIES:
            for backend in ['exact','float']:
                q=set(methods[family+'_q_'+backend]['positive_modes']);ex=set(methods[family+'_exit_'+backend]['positive_modes'])
                comparisons[family+'_'+backend]=dict(new_exit_vs_original=sorted(ex-original),
                    new_exit_vs_prior=sorted(ex-prior),new_exit_vs_prior_and_own_q=sorted(ex-prior-q),
                    exit_not_in_own_q=sorted(ex-q),q_new_vs_prior=sorted(q-prior))
        for name,props in candidates.items():
            for candidate in props:
                a,rhs=matrices[candidate['mode']]
                slacks=[r-sum((z*b for z,b in zip(row,candidate['b'])),F(0)) for row,r in zip(a,rhs)]
                checks=dict(closed_point_feasible=all(s>=0 for s in slacks),minimum_slack=min(slacks),
                            strict_point=all(s>0 for s,row in zip(slacks,a) if any(row)) and all(s>=0 for s in slacks))
                points.append(dict(seed=seed,method=name,**candidate,**checks))
                counts['actual_point_checks']+=1;counts['actual_points_feasible']+=checks['closed_point_feasible']
        record=dict(seed=seed,methods=methods,comparisons=comparisons,geometry=classified,
                    original_positive_modes=sorted(original),prior_308_309_312_positive_modes=sorted(prior),
                    x_observed=x.tolist(),v_observed=v.tolist())
        filename=f'{seed}_geometry.json';save(out/filename,record);files[filename]=sha(out/filename)
        tasks.append(dict(seed=seed,file=filename,sha256=files[filename],methods=methods,comparisons=comparisons))
    assert len(tasks)==64
    aggregate={}
    for name in NAMES:
        aggregate[name]=dict(proposal_points=sum(p['method']==name for p in points),
                            positive_task_mode_pairs=sum(len(t['methods'][name]['positive_modes']) for t in tasks),
                            new_positive_task_mode_pairs=sum(len(t['methods'][name]['new_positive_modes']) for t in tasks),
                            actual_points_feasible=sum(p['method']==name and p['closed_point_feasible'] for p in points))
    comparison_aggregate={name:{field:sum(len(t['comparisons'][name][field]) for t in tasks)
                                  for field in tasks[0]['comparisons'][name]} for name in tasks[0]['comparisons']}
    for name,value in [('tasks.json',tasks),('points.json',points),('aggregate.json',aggregate),('comparisons.json',comparison_aggregate)]:
        save(out/name,value);files[name]=sha(out/name)
    files['protocol.json']=sha(out/'protocol.json')
    result=dict(passed=True,counts=dict(counts),aggregate=aggregate,comparisons=comparison_aggregate,seconds=time.perf_counter()-begin,
                outputs_sha256=files,query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print({k:v for k,v in result.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
