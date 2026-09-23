"""Evaluator-only exact integral over U[0,1] for the frozen support witness."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import numpy as np
import multiplier_fixed_point_exact as core
from audit_multiplier_fixed_point import exact_forward
from run_multiplier_fixed_point_screen import sha,dump

def segments(b):
    pieces=[(F(0),F(1),F(1),F(0))]
    for bias in b:
        result=[]
        for lo,hi,s,c in pieces:
            split=sorted({lo,hi}|{(k-c-bias)/s for k in core.K if s and lo<(k-c-bias)/s<hi})
            for a,z in zip(split[:-1],split[1:]):
                reg=core.branch(s*(a+z)/2+c+bias);ss=core.S[reg]
                result.append((a,z,ss*s,ss*(c+bias)+core.C[reg]))
        pieces=result
    return pieces

def risk(b,teacher):
    pp=segments(b);tt=segments(teacher);total=F(0);count=0
    for lo,hi,s,c in pp:
        for tl,th,ts,tc in tt:
            a=max(lo,tl);z=min(hi,th)
            if a>=z:continue
            ds=s-ts;dc=c-tc
            # Squared affine difference integrated exactly, no query grid.
            total+=ds*ds*(z**3-a**3)/3+ds*dc*(z*z-a*a)+dc*dc*(z-a);count+=1
    assert total>=0
    return total,count

def main():
    sys.set_int_max_str_digits(0);ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/multiplier_fixed_point';out=base/'witness/query_risk.json';assert not out.exists()
    summary=json.loads((base/'witness/summary.json').read_text());assert summary['passed']
    assert sha(base/'witness/trajectory.json')==summary['trajectory_sha256'];seed=summary['seed'];restart=summary['restart']
    records=json.loads((base/'witness/trajectory.json').read_text());a=np.load(base/'screen'/f'{seed}.npz')
    x=[F(float(t)) for t in a['x']];v=[F(float(t)) for t in a['v']];best=[F(float(t)) for t in a['best'][restart]]
    def score(b):
        return max(abs(p-y) for p,y in zip(exact_forward(x,b)[0],v)),sum(t*t for t in b)/2
    err,mov=score(best);chosen=-1;threshold=F(float(.001+.000001))
    for r in records:
        b,_=core.decode(r['state']);e,m=score(b);f=e<=threshold;oldf=err<=threshold
        if (f and not oldf) or (f and oldf and m<mov) or (not f and not oldf and e<err):best=b;err=e;mov=m;chosen=r['step']
    # All adaptation/retention above is support-only. The teacher is used here.
    teacher=[F(float(t)) for t in np.random.default_rng(seed).uniform(-.12,.12,4)]
    exactrisk,count=risk(best,teacher);initial=[F(float(t)) for t in a['best'][restart]];initialrisk,ic=risk(initial,teacher)
    assert exactrisk<initialrisk
    methods={}
    for method in ['alm64','nodual64','pc80','adam60','adam240','gn20']:
        saved=np.load(base/'continuation'/f'{seed}_{method}.npz');i=list(saved['restarts']).index(restart);b=[F(float(t)) for t in saved['best'][-1,i]]
        rr,cc=risk(b,teacher)
        methods[method]=dict(uniform_risk=float(rr),exact=core.pack(rr),integration_cells=cc)
    result=dict(passed=True,seed=seed,restart=restart,source_sha256=sha(Path(__file__)),witness_summary_sha256=sha(base/'witness/summary.json'),
        exact_support_selected_step=chosen,exact_support_error=core.pack(err),rational_trajectory_uniform_risk=core.pack(exactrisk),uniform_risk=float(exactrisk),
        initial_uniform_risk=float(initialrisk),initial_exact=core.pack(initialrisk),strict_risk_improvement=core.pack(initialrisk-exactrisk),
        relative_risk_reduction=float((initialrisk-exactrisk)/initialrisk),integration_cells=count,initial_integration_cells=ic,
        saved_float_methods=methods,scope='One support-selected existing development witness; exact U[0,1] risk, not expected performance over tasks or regression superiority')
    dump(out,result);print(json.dumps({k:v for k,v in result.items() if k in ['passed','seed','restart','exact_support_selected_step','uniform_risk','initial_uniform_risk','relative_risk_reduction','integration_cells']}),flush=True)

if __name__=='__main__':main()
