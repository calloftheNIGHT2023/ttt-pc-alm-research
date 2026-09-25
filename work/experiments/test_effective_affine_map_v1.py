"""315 synthetic exact map algebra, witness and actual solver replay tests."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import traceback
import numpy as np
from posterior_confirmation_pipeline import discovery_box
from solver_policy_trace_v1 import step
from test_complete_credit_amplitude_events_v2 import fixtures
from complete_credit_rational_reference_v1 import direct_step
from effective_affine_map_v1 import Affine,build,canonical,evaluate,pack,dump,restore
from evaluate_complete_credit_mode_geometry_v1 import save,sha


def one(b,h,u,x,v,method,checks):
    result=step(b[None],h[:,None],u[:,None],x,v,method)
    policies={g:a[0] for g,a in result['policies'].items()}
    rows=build(policies,result['schema'],x,v,method)
    point=pack(b,h,u);computed=evaluate(rows,point)
    actual=np.r_[result['b'].ravel(),result['h'].ravel(),result['u'].ravel()]
    gap=float(np.max(abs(np.array(computed,dtype=float)-actual)))
    assert gap<=1e-10,(gap,method,b)
    assert canonical(restore(dump(rows)))==canonical(rows)
    # Exact affine identity at synthetic off-policy points is algebra only.
    delta=[F((i%5)-2,37) for i in range(len(point))]
    left=evaluate(rows,[p+q for p,q in zip(point,delta)])
    right=[z+sum((c*delta[k] for k,c in row.terms.items() if k>=0),F(0)) for z,row in zip(computed,rows)]
    assert left==right
    checks['actual_solver_steps']+=1;checks['exact_affine_rows']+=len(rows)
    return gap,rows,result


def run(root,out):
    checks=Counter();maxgap=0.;rng=np.random.default_rng(315019)
    with discovery_box(.12):
        for name,s in fixtures():
            b=np.array(s['b'],float);h=np.array(s['h'],float);u=np.array(s['direction'],float)
            x=np.array(s['x'],float);v=np.array(s['v'],float)
            gap,rows,_=one(b,h,u,x,v,'alm',checks);maxgap=max(maxgap,gap)
            exact=direct_step(s,1);pred=evaluate(rows,pack(b,h,u));d=len(b);n=len(x)
            egap=max(float(np.max(abs(np.array(pred[:d],float)-np.array(exact['b'],float)))),
                     float(np.max(abs(np.array(pred[d:d+d*n],float).reshape(d,n)-np.array(exact['h'],float)))))
            assert egap<1e-10;checks['independent_rational_fixture_steps']+=1
        for d,n in [(1,1),(2,4),(4,4),(4,7)]:
            for method in ['alm','nodual']:
                for case in range(8):
                    b=rng.uniform(-.12,.12,d);h=rng.uniform(0,1,(d,n));u=rng.uniform(-.6,.6,(d,n)) if method=='alm' else np.zeros((d,n))
                    x=rng.uniform(0,1,n);v=rng.uniform(-.0005,1.0005,n)
                    for t in range(3):
                        gap,_,result=one(b,h,u,x,v,method,checks);maxgap=max(maxgap,gap)
                        b=result['b'][0];h=result['h'][:,0];u=result['u'][:,0]
        witness=[]
        for bvalue in [-.11,-.09]:
            gap,rows,result=one(np.array([bvalue]),np.array([[.5]]),np.array([[0.]]),np.array([.1]),np.array([.4]),'alm',checks)
            witness.append((rows,result))
        assert canonical(witness[0][0])==canonical(witness[1][0])
        assert not np.array_equal(witness[0][1]['policies']['activity_0'],witness[1][1]['policies']['activity_0'])
        Bvar,Uvar=Affine({0:F(1)}),Affine({2:F(1)})
        H=F(.4)-F(.001);T=F(.01);X=F(.1)
        expected_b=(2*(H-2*X)+2*Uvar+T*Bvar)/(4+T)
        expected=[expected_b,Affine(H),Uvar+F(1,2)*(H-2*X-2*expected_b)]
        assert canonical(expected)==canonical(witness[0][0]);checks['exact_cross_policy_witnesses']+=1
    names=[Path(__file__).name,'effective_affine_map_v1.py','solver_policy_trace_v1.py',
        'complete_credit_rational_reference_v1.py','test_complete_credit_amplitude_events_v2.py','posterior_confirmation_pipeline.py']
    final=dict(passed=True,checks=dict(checks),max_float_gap=maxgap,real_task_data_accessed=False,
        future_guard_validity_established=False,source_sha256={n:sha(root/'work/experiments'/n) for n in names})
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
