"""Independent polynomial interpolation check and exact branch-escape witness."""
import argparse
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import time
import numpy as np
import multiplier_fixed_point_exact as core
from run_multiplier_fixed_point_screen import sha,dump

def residuals(x,b,h):
    return [[h[j][i]-max(F(0),1-abs(2*((x[i] if j==0 else h[j-1][i])+b[j])-1)) for i in range(len(x))] for j in range(len(b))]

def independent_block_minima(x,v,b,h):
    """No KKT or analytic candidate formula from recovery is reused.

    On every interval recover its quadratic from three exact objective values,
    check its vertex AND endpoints, and require no different equal minimizer.
    """
    d=len(b);n=len(x);checks=[]
    for j,i in [(j,i) for j in range(d) for i in range(n)]+[(j,None) for j in range(d)]:
        isbias=i is None;current=b[j] if isbias else h[j][i]
        lower,upper=(-core.B,core.B) if isbias else ((core.bounds(v)[0][i],core.bounds(v)[1][i]) if j==d-1 else (F(0),F(1)))
        events=[k-p for p in (x if j==0 else h[j-1]) for k in core.K] if isbias else ([k-b[j+1] for k in core.K] if j<d-1 else [])
        breaks=sorted({lower,upper}|{z for z in events if lower<z<upper})
        def energy(z):
            bb=b.copy();hh=[row.copy() for row in h]
            if isbias:bb[j]=z
            else:hh[j][i]=z
            rr=residuals(x,bb,hh)
            # Other residual terms are constant and do not affect minimizers.
            return sum((t*t for row in rr for t in row),F(0))/(n if isbias else 1)+core.TRUST*(z-current)**2
        old=energy(current);minimum=None;points=[];intervals=0
        for lo,hi in zip(breaks[:-1],breaks[1:]):
            e0=energy(lo);em=energy((lo+hi)/2);e1=energy(hi)
            # E(lo+t*(hi-lo))=a*t^2+c*t+e0.
            aa=2*(e1+e0-2*em);cc=e1-e0-aa
            assert aa>0
            t=min(F(1),max(F(0),-cc/(2*aa)));z=lo+t*(hi-lo)
            for z in [lo,z,hi]:
                e=energy(z)
                if minimum is None or e<minimum:minimum=e;points=[z]
                elif e==minimum:points.append(z)
            intervals+=1
        ok=minimum==old and all(t==current for t in points)
        checks.append(dict(block=[j,i],accepted=ok,intervals=intervals,gap=core.pack(minimum-old)))
    return checks

def exact_forward(x,b):
    vals=list(x);patterns=[];jac=[[F(0)]*len(b) for _ in x]
    for j,bias in enumerate(b):
        z=[t+bias for t in vals];patterns.append([core.branch(t) for t in z])
        slopes=[0 if t in core.K else core.S[core.branch(t)] for t in z]
        for i,s in enumerate(slopes):
            jac[i]=[s*t for t in jac[i]];jac[i][j]+=s
        vals=[core.g(t) for t in z]
    return vals,patterns,jac

def multiplier_witness(x,v,b,h):
    rr=residuals(x,b,h);u=[[t/2 for t in row] for row in rr];records=[];d=len(b);n=len(x)
    for j,i in [(j,i) for j in range(d) for i in range(n)]+[(j,None) for j in range(d)]:
        isbias=i is None;candidates=core.bias_candidates(x,b,h,j,u) if isbias else core.activity_candidates(x,v,b,h,j,i,u)
        current=b[j] if isbias else h[j][i]
        for value,_ in candidates:
            bb=b.copy();hh=[row.copy() for row in h]
            if isbias:bb[j]=value
            else:hh[j][i]=value
            rp=residuals(x,bb,hh);weight=F(1,n) if isbias else F(1)
            # Half-energy normalization gives precisely Delta-t*eta*kappa.
            delta=(weight*sum((a*a-c*c for r,s in zip(rp,rr) for a,c in zip(r,s)),F(0))+core.TRUST*(value-current)**2)/2
            kappa=-weight*sum((c*(a-c) for r,s in zip(rp,rr) for a,c in zip(r,s)),F(0))
            assert delta>=0
            if kappa>0:
                bound=delta/(F(1,2)*kappa);t=bound.numerator//bound.denominator+1
                assert delta-t*F(1,2)*kappa<0
                records.append(dict(block=[j,i],competitor=core.pack(value),delta=core.pack(delta),kappa=core.pack(kappa),
                    threshold=core.pack(bound),strict_accumulations=t,block_difference_at_t=core.pack(delta-t*F(1,2)*kappa)))
    return sorted(records,key=lambda r:r['strict_accumulations'])

def rational_trace(x,v,state,steps):
    b,h=copy.deepcopy(state);u=[[F(0)]*len(x) for _ in b];records=[];initial=exact_forward(x,b)[1]
    for step in range(steps+1):
        pred,patterns,_=exact_forward(x,b);rr=residuals(x,b,h)
        records.append(dict(step=step,state=core.encode((b,h)),u=[[core.pack(t) for t in row] for row in u],
            changed_parameter_branch=patterns!=initial,support_error=core.pack(max(abs(a-c) for a,c in zip(pred,v))),residual_max=core.pack(max(abs(t) for row in rr for t in row))))
        if step==steps:break
        for j in reversed(range(len(b))):
            for i in range(len(x)):h[j][i]=min(core.activity_candidates(x,v,b,h,j,i,u),key=lambda z:z[1])[0]
        for j in range(len(b)):b[j]=min(core.bias_candidates(x,b,h,j,u),key=lambda z:z[1])[0]
        rr=residuals(x,b,h)
        u=[[a+c/2 for a,c in zip(row,r)] for row,r in zip(u,rr)]
    return records

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/multiplier_fixed_point';out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists();start=time.perf_counter()
    p=json.loads((base/'continuation/protocol.json').read_text())
    for name,h in p['source_sha256'].items():assert sha(src/name)==h,name
    certs=json.loads((base/'exact/certificates.json').read_text());checks=[];accepted=[]
    for c in certs:
        if c['state'] is None:continue
        a=np.load(base/'screen'/f'{c["seed"]}.npz');x=[F(float(t)) for t in a['x']];v=[F(float(t)) for t in a['v']];b,h=core.decode(c['state'])
        proof=independent_block_minima(x,v,b,h);ok=all(r['accepted'] for r in proof)
        assert ok==c['proof']['accepted']
        pred,_,jac=exact_forward(x,b);res=[(1 if a>=z else -1)*max(F(0),abs(a-z)-core.EPS) for a,z in zip(pred,v)]
        grad=[sum((row[j]*r for row,r in zip(jac,res)),F(0))/len(x) for j in range(4)]
        witnesses=multiplier_witness(x,v,b,h) if ok else []
        entry=dict(seed=c['seed'],restart=c['restart'],accepted=ok,blocks=proof,zero_global_jacobian=not any(t for row in jac for t in row),
            zero_exact_BP_gradient=not any(grad),gradient=[core.pack(t) for t in grad],multiplier_witnesses=witnesses)
        checks.append(entry)
        if ok:accepted.append(entry)
    dump(out/'independent_certificates.json',checks)
    # Witness selected by SUPPORT-only condition, not by query error.
    support=json.loads((base/'continuation/support_rows.json').read_text());chosen=[]
    for r in support:
        if r['method']=='alm64' and r['strict_band_feasible'] and r['first_parameter_branch_change'] is not None:
            controls=[t for t in support if (t['seed'],t['restart'])==(r['seed'],r['restart']) and t['method']!='alm64']
            if all(not t['strict_band_feasible'] for t in controls):chosen.append(r)
    dump(out/'witness_selection.json',[dict(seed=r['seed'],restart=r['restart'],selection='ALM support feasible and branch change, all five controls support infeasible') for r in chosen])
    escapes=[]
    for r in chosen:
        c=next(t for t in certs if (t['seed'],t['restart'])==(r['seed'],r['restart']));a=np.load(base/'screen'/f'{r["seed"]}.npz')
        x=[F(float(t)) for t in a['x']];v=[F(float(t)) for t in a['v']]
        steps=r['first_parameter_branch_change'];records=rational_trace(x,v,core.decode(c['state']),steps)
        assert records[-1]['changed_parameter_branch'] and not any(t['changed_parameter_branch'] for t in records[:-1])
        floatpath=np.load(base/'continuation'/r['file']);index=list(floatpath['restarts']).index(r['restart'])
        difference=0.
        for item in records:
            bb,hh=core.decode(item['state']);s=item['step']
            difference=max(difference,float(np.max(abs(np.array([float(t) for t in bb])-floatpath['b'][s,index]))),
                float(np.max(abs(np.array([[float(t) for t in row] for row in hh])-floatpath['h'][s,:,index]))))
        file=out/f'exact_escape_{r["seed"]}_{r["restart"]}.json';dump(file,records)
        escapes.append(dict(seed=r['seed'],restart=r['restart'],exact_first_forward_branch_change=steps,max_float_vs_exact_primal_error=difference,file=file.name,sha256=sha(file)))
    result=dict(passed=True,independent_points=len(checks),accepted=len(accepted),all_block_checks=sum(len(t['blocks']) for t in checks),
        interval_checks=sum(r['intervals'] for t in checks for r in t['blocks']),positive_multiplier_witness_points=sum(bool(t['multiplier_witnesses']) for t in accepted),
        exact_zero_J_points=sum(t['zero_global_jacobian'] for t in accepted),exact_zero_gradient_points=sum(t['zero_exact_BP_gradient'] for t in accepted),
        escapes=escapes,seconds=time.perf_counter()-start,source_sha256=sha(Path(__file__)),continuation_protocol_sha256=sha(base/'continuation/protocol.json'),
        output_sha256={f.name:sha(f) for f in out.glob('*.json')})
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
