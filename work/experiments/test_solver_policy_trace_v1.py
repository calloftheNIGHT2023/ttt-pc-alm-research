"""313 synthetic-only trace arithmetic and batch/scalar policy checks."""
import argparse
from collections import Counter
from pathlib import Path
import traceback
import numpy as np
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from solver_policy_trace_v1 import step,segments
from test_complete_credit_amplitude_events_v2 import fixtures
from complete_credit_rational_reference_v1 import direct_step
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    checks=Counter();maxgap=0.
    with discovery_box(.12):
        for name,f in fixtures():
            b=np.array(f['b'],dtype=float)[None];h=np.array(f['h'],dtype=float)[:,None]
            u=np.array(f['direction'],dtype=float)[:,None];x=np.array(f['x'],dtype=float);v=np.array(f['v'],dtype=float)
            trace=step(b,h,u,x,v)
            baseline=cold.Local(b,x,v,'alm');baseline.h=h.copy();baseline.u=u.copy();baseline.step()
            for k in ['b','h','u']:assert np.array_equal(trace[k],getattr(baseline,k));checks['fixture_arrays']+=1
            exact=direct_step(f,1)
            maxgap=max(maxgap,float(np.max(abs(trace['b'][0]-np.array(exact['b'],dtype=float)))),
                       float(np.max(abs(trace['h'][:,0]-np.array(exact['h'],dtype=float)))))
            checks['rational_fixture_steps']+=1
        rng=np.random.default_rng(313017)
        for d,n in [(1,1),(2,4),(4,4),(4,7)]:
            for method in ['alm','nodual']:
                b=rng.uniform(-.12,.12,(5,d));x=rng.uniform(0,1,n);v=rng.uniform(-.0005,1.0005,n)
                h=rng.uniform(0,1,(d,5,n));u=rng.uniform(-.4,.4,(d,5,n)) if method=='alm' else np.zeros_like(h)
                baseline=cold.Local(b,x,v,method);baseline.h=h.copy();baseline.u=u.copy()
                for t in range(20):
                    traced=step(b,h,u,x,v,method);baseline.step()
                    for k in ['b','h','u']:assert np.array_equal(traced[k],getattr(baseline,k));checks['random_batch_arrays']+=1
                    for i in range(5):
                        scalar=step(b[i:i+1],h[:,i:i+1],u[:,i:i+1],x,v,method)
                        assert scalar['schema']==traced['schema']
                        for g,a in scalar['policies'].items():assert np.array_equal(a[0],traced['policies'][g][i]);checks['scalar_batch_policy_arrays']+=1
                        for k,a in [('b',scalar['b'][0]),('h',scalar['h'][:,0]),('u',scalar['u'][:,0])]:
                            assert np.array_equal(a,traced[k][i] if k=='b' else traced[k][:,i]);checks['scalar_batch_state_arrays']+=1
                    b,h,u=traced['b'],traced['h'],traced['u']
    assert maxgap<1e-10
    assert segments(np.array([[0],[0],[1],[2],[2]]))==[
        dict(start=1,end=2,length=2),dict(start=3,end=3,length=1),dict(start=4,end=5,length=2)]
    names=[Path(__file__).name,'solver_policy_trace_v1.py','cold_stagnation_switch.py','streaming_branch_projection.py',
           'posterior_confirmation_pipeline.py','complete_credit_rational_reference_v1.py','test_complete_credit_amplitude_events_v2.py']
    final=dict(passed=True,checks=dict(checks),max_rational_gap=maxgap,real_task_data_accessed=False,
        source_sha256={n:sha(root/'work/experiments'/n) for n in names})
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
