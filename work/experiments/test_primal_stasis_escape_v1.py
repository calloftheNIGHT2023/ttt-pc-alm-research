"""318 synthetic gap identities, infinite stasis and finite exact escape."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import traceback
import numpy as np
from primal_stasis_escape_v1 import escape,blocks,gap,beta
from affine_root_validation_v1 import state,exact_step,validate
from affine_root_certificate_v1 import propose
from effective_affine_map_v1 import build,pack
from solver_policy_trace_v1 import step
from posterior_confirmation_pipeline import discovery_box
from audit_affine_root_proposal_v1 import g
from evaluate_complete_credit_mode_geometry_v1 import save,sha


def run(root,out):
    counts=Counter();rng=np.random.default_rng(318001)
    for case in range(20):
        d,n=2,3;x=[F(int(t),64) for t in rng.integers(1,64,n)];v=[F(int(t),64) for t in rng.integers(1,64,n)]
        b=[F(int(t),128) for t in rng.integers(-10,11,d)]
        h=[[F(int(t),64) for t in rng.integers(1,64,n)],v[:]]
        u=[[F(int(t),32) for t in rng.integers(-6,7,n)] for j in range(d)]
        q=b+[t for row in h for t in row]+[t for row in u for t in row]
        delta=[F(0)]*(d+d*n)+[F(int(t),31) for t in rng.integers(-5,6,d*n)]
        s=state(q,d,x,v);bb,du=blocks(s,delta)
        for block in bb:
            for a,z in zip(block['breaks'][:-1],block['breaks'][1:]):
                for value in [a,(a+z)/2,z]:
                    initial=gap(s,du,block,value,F(0));slope=beta(s,du,block,value)
                    for t in [F(-2),F(1),F(7,3),F(19)]:
                        assert gap(s,du,block,value,t)==initial+t*slope
                        counts['exact_gap_identities']+=1
    # Same input, incompatible observed bands: dual drift need never exit.
    x=[F(1,4),F(1,4)];v=[F(3,8),F(5,8)];eps=F(.001)
    hh=[v[0]+eps,v[1]-eps];q=[F(0)]+hh+[F(0),F(0)]
    delta=[F(0)]*3+[(h-F(1,2))/2 for h in hh]
    infinite=escape(q,delta,1,x,v);assert infinite['status']=='primal_stasis_for_all_nonnegative_phases'
    for t in [0,1,2,7,64,1000]:
        current=[z+t*d for z,d in zip(q,delta)];actual,_=exact_step(state(current,1,x,v),'alm')
        assert actual==[z+(t+1)*d for z,d in zip(q,delta)]
        counts['independent_infinite_stasis_steps']+=1
    save(out/'infinite_fixture.json',dict(q=q,delta=delta,x=x,v=v,result=infinite))
    # Reproducible fixture generation is synthetic, not task development data.
    # Stop at the first exact finite-escape example; retain its generation index.
    found=None
    with discovery_box(.12):
        for case in range(64):
            d,n=4,4;x=rng.uniform(.01,.99,n);teacher=rng.uniform(-.12,.12,d)
            v=x.copy()
            for t in teacher:v=np.maximum(0,np.minimum(2*(v+t),2-2*(v+t)))
            b=rng.uniform(-.12,.12,d);h=[];p=x.copy()
            for t in b:p=np.maximum(0,np.minimum(2*(p+t),2-2*(p+t)));h.append(p.copy())
            h=np.array(h);h[-1]=v;u=np.zeros_like(h)
            for iteration in range(64):
                rr=step(b[None],h[:,None],u[:,None],x,v,'nodual')
                b,h,u=rr['b'][0],rr['h'][:,0],rr['u'][:,0]
            rr=step(b[None],h[:,None],u[:,None],x,v,'nodual')
            rows=build({g:a[0] for g,a in rr['policies'].items()},rr['schema'],x,v,'nodual')
            dim=d+d*n;answer=propose(rows,pack(b,h,u),list(range(dim)),{j:F(0) for j in range(dim,len(rows))})
            counts['synthetic_fixture_attempts']+=1
            if not answer['consistent']:continue
            q=answer['full_root'];checked=validate(q,d,x,v,'nodual')
            if not checked['actual_fixed_point']:continue
            s=state(q,d,x,v);du=[]
            for j,bj in enumerate(s['b']):
                prev=s['x'] if j==0 else s['h'][j-1]
                du.extend((hh-g(p+bj))/2 for hh,p in zip(s['h'][j],prev))
            delta=[F(0)]*dim+du;result=escape(q,delta,d,x,v)
            if result['status']!='first_primal_exit_found':continue
            tstar=result['first_exit_input_phase']
            for t in range(tstar+1):
                current=[z+t*e for z,e in zip(q,delta)];actual,_=exact_step(state(current,d,x,v),'alm')
                stays=actual==[z+(t+1)*e for z,e in zip(q,delta)]
                assert stays==(t<tstar);counts['independent_finite_escape_steps']+=1
            found=dict(case=case,teacher=list(map(F,teacher)),x=list(map(F,x)),v=list(map(F,v)),q=q,delta=delta,result=result)
            break
    assert found is not None,'No synthetic finite escape fixture found within fixed cap'
    save(out/'finite_fixture.json',found)
    names=[Path(__file__).name,'primal_stasis_escape_v1.py','audit_affine_root_proposal_v1.py',
           'affine_root_validation_v1.py','affine_root_certificate_v1.py','effective_affine_map_v1.py',
           'solver_policy_trace_v1.py','complete_credit_rational_reference_v1.py']
    final=dict(passed=True,counts=dict(counts),finite_exit_phase=found['result']['first_exit_input_phase'],
               real_task_data_accessed=False,source_sha256={n:sha(root/'work/experiments'/n) for n in names})
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
