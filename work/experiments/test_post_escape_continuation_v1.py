"""319 synthetic entry, mixed horizon, actual Local and tracer agreement."""
import argparse
from collections import Counter
from pathlib import Path
import traceback
import numpy as np
from post_escape_continuation_v1 import entries,rollout,forward,VARIANTS
from solver_policy_trace_v1 import step
from posterior_confirmation_pipeline import discovery_box
from evaluate_complete_credit_mode_geometry_v1 import save,sha


def run(root,out):
    counts=Counter();rng=np.random.default_rng(319001)
    for applicable,found in [(False,False),(True,False),(True,True)]:
        for phase in ([1,26] if found else [0]):
            original=list(map(str,range(36)));q=list(map(str,range(1,37)));e=list(map(float,range(2,38)))
            line=dict(point=original,proposal=dict(q=q));result=dict(applicable=applicable,
                status='first_primal_exit_found' if found else 'other',first_exit_input_phase=phase)
            starts,hs,k=entries(line,dict(result=result,floating_exit_output=e))
            assert np.array_equal(starts['original64'],np.arange(36))
            assert np.array_equal(starts['corrected64'],np.arange(36)+(1 if applicable else 0))
            assert np.array_equal(starts['escaped64'],np.arange(36)+(2 if found else 1 if applicable else 0))
            assert hs['corrected_virtual']==64+k and hs['escaped_matched']==64-k and sum(hs.values())==320
            counts['entry_rules']+=1
    with discovery_box(.12):
        for method in ['alm','nodual']:
            for case in range(5):
                r,d,n=4,4,4;x=rng.uniform(0,1,n);v=rng.uniform(.01,.99,n)
                b=rng.uniform(-.12,.12,(r,d));h=rng.uniform(0,1,(d,r,n));h[-1]=v
                u=rng.uniform(-.2,.2,(d,r,n)) if method=='alm' else np.zeros_like(h)
                points=np.c_[b,h.transpose(1,0,2).reshape(r,-1),u.transpose(1,0,2).reshape(r,-1)]
                horizons=np.array([0,1,3,7]);arrays,meta=rollout(points,horizons,x,v,method)
                assert meta['valid_local_updates']==11
                for i,steps in enumerate(horizons):
                    bb,hh,uu=b[i:i+1].copy(),h[:,i:i+1].copy(),u[:,i:i+1].copy()
                    for t in range(steps+1):
                        assert np.array_equal(arrays['b'][t,i],bb[0])
                        assert np.array_equal(arrays['h'][t,:,i],hh[:,0])
                        assert np.array_equal(arrays['u'][t,:,i],uu[:,0]);counts['individual_state_arrays']+=3
                        cc,vv=forward(bb,x);assert np.array_equal(arrays['forward_codes'][t,i],cc[0])
                        assert np.array_equal(arrays['forward_values'][t,i],vv[0]);counts['readout_arrays']+=2
                        if t<steps:
                            result=step(bb,hh,uu,x,v,method);bb,hh,uu=result['b'],result['h'],result['u']
                    for t in range(steps+1,8):
                        assert np.array_equal(arrays['b'][t,i],bb[0]);counts['padding_checks']+=1
                for variant in VARIANTS:assert variant in ['original64','corrected64','escaped64','corrected_virtual','escaped_matched']
    names=[Path(__file__).name,'post_escape_continuation_v1.py','cold_stagnation_switch.py',
           'solver_policy_trace_v1.py','posterior_confirmation_pipeline.py','streaming_branch_projection.py']
    final=dict(passed=True,counts=dict(counts),real_task_data_accessed=False,
               source_sha256={n:sha(root/'work/experiments'/n) for n in names})
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
