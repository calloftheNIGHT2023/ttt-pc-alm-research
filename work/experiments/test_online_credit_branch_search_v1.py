"""326 compatibility and transparent geometry reuse before the full gate."""
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
import online_credit_branch_search_v1 as new
import certificate_activity_attribution as old
import conditioned_mode_geometry as conditioned
import complete_credit_mode_geometry_v1 as classifier
from posterior_confirmation_pipeline import discovery_box
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    start=time.perf_counter();raw=root/'results/certificate_activity_attribution/development'
    rows={r['seed']:r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    config=next(c for c in old.CONFIGS if c['name']=='credit_control_probe33');counts={};notes=[]
    with discovery_box(.12):
        for seed in [5910000,5910001]:
            src=raw/rows[seed]['file'];assert sha(src)==rows[seed]['sha256']
            with np.load(src,allow_pickle=False) as z:x,v=z['x_observed'],z['v_observed']
            q=np.linspace(0,1,257)
            with conditioned.geometry_scope([]):expected,em=old.fit(config,x,v,q,seed,trace=True)
            actual,am=new.fit(x,v,q,seed,policy='none',trace=True)
            assert set(actual)==set(expected)
            for name in expected:assert actual[name].tobytes()==expected[name].tobytes(),(seed,name)
            assert am['original_positive_modes']==em['positive_modes']==am['positive_modes']
            assert am['original_visited_modes']==em['visited_modes'];counts['none_array_checks']=counts.get('none_array_checks',0)+len(expected)
            for policy,channel in [('first_fit','dual'),('uniform_state','dual_plus_residual')]:
                actual,meta=new.fit(x,v,q,seed,policy=policy,channel=channel,trace=True)
                state=read(root/'results/support_consistency_trigger/states_v1'/f'{seed}_{policy}.json')
                saved=read(root/'results/support_consistency_trigger/development_v1'/f'{seed}_{policy}.json')
                geo=read(root/'results/support_consistency_trigger/geometry_v1'/f'{seed}_geometry.json')
                for name in expected:
                    if name.startswith(('prefix_','anchor_','initial_','effective_','atomic_')) or name in ['selected_b','best_bank','point_prediction']:
                        assert actual[name].tobytes()==expected[name].tobytes(),(seed,policy,channel,name)
                for field in ['b','h','u']:
                    assert np.array_equal(actual['trigger_'+field],np.array(state[field]))
                assert np.array_equal(actual['trigger_credit'],np.array(state['credits'][channel]))
                assert meta['selected_state']['location']==state['location']
                assert meta['proposal']['proposals']==saved['results'][channel]['proposals']
                assert set(meta['positive_modes'])==set(em['positive_modes'])|set(geo['methods'][policy+'/'+channel]['positive_modes'])
                counts['complete_candidate_calls']=counts.get('complete_candidate_calls',0)+1
                notes.append(dict(seed=seed,policy=policy,channel=channel,positive=len(meta['positive_modes']),complete_seconds=meta['total_seconds']))
        # Exercise positive, infeasible and measure-zero explicit geometry.
        geometries=root/'results/support_consistency_trigger/geometry_v1';cases={}
        for seed in range(5910000,5910064):
            task=read(geometries/f'{seed}_geometry.json')
            for mode,result in task['geometry'].items():
                cases.setdefault(result['classification'],(seed,mode,task['x_observed'],task['v_observed']))
            if len(cases)==3:break
        cases['credit_specific_positive']=(5910054,'02010202020101010302010100020101',None,None)
        for label,(seed,mode,x,v) in cases.items():
            if x is None:
                t=read(geometries/f'{seed}_geometry.json');x,v=t['x_observed'],t['v_observed']
            x,v=np.array(x),np.array(v);expected=classifier.classify_mode(x,v,mode);poly,actual=new.explicit_geometry(x,v,mode)
            for name in ['classification','positive_volume_certified','volume','lp_calls','numerical_volume_available']:
                assert actual[name]==expected[name],(label,name)
            assert (poly is not None)==expected['numerical_volume_available']
            assert conditioned.geometry_scope.__module__=='conditioned_mode_geometry' and new.shared.geometry.polytope is conditioned.ORIGINAL
            counts['explicit_geometry_cases']=counts.get('explicit_geometry_cases',0)+1
    result=dict(passed=True,counts=counts,notes=notes,seconds=time.perf_counter()-start,
                source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'online_credit_branch_search_v1.py']},
                query_targets_accessed=False,trace_functional_only=True,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],a.out)
    except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
