"""Preflight exact reachability versus an independent single-observation LP."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import forward_credit_envelope as model
import exact_credit_hull as hull
from verify_exact_credit_hull import validate
import streaming_branch_projection as base

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def independent_lp(x,pattern,eta):
    """b,z,h variables; bounded equation errors, not interval propagation."""
    d=len(pattern);rows=[];rhs=[]
    for j,k in enumerate(pattern):
        a=np.zeros(3*d);a[d+j]=1;a[j]=-1
        offset=float(x) if j==0 else 0.
        if j:a[2*d+j-1]=-1
        rows.extend([a,-a]);rhs.extend([float(eta)+offset,float(eta)-offset])
        a=np.zeros(3*d);a[2*d+j]=1;a[d+j]=-[0,2,-2,0][k];c=[0,0,2,0][k]
        rows.extend([a,-a]);rhs.extend([c+float(eta),-c+float(eta)])
    bounds=[(-.12,.12)]*d+[[(None,0),(0,.5),(.5,1),(1,None)][k] for k in pattern]+[(0,1)]*d
    c=np.zeros(3*d);c[-1]=1;options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)
    a=linprog(c,A_ub=rows,b_ub=rhs,bounds=bounds,options=options);b=linprog(-c,A_ub=rows,b_ub=rhs,bounds=bounds,options=options)
    assert a.success==b.success and a.status in [0,2] and b.status in [0,2]
    return None if not a.success else (float(a.fun),float(-b.fun))

def crosscheck_intervals(x,depth=4):
    counts=Counter();max_gap=0.
    for eta in [F(0),model.ETA]:
        for xx in x:
            for pattern in product(range(4),repeat=depth):
                exact,_=model.reachable(xx,pattern,eta);numeric=independent_lp(xx,pattern,eta);counts['single_observation_patterns']+=1;counts['lp_calls']+=2
                if exact is None:
                    # Floating LP may admit an almost-empty interval. It is a
                    # crosscheck, never the reason to discard an exact branch.
                    counts['empty_exact_numeric_feasible' if numeric is not None else 'both_empty']+=1
                else:
                    assert numeric is not None,(xx,pattern,eta,exact)
                    gap=max(abs(float(a)-b) for a,b in zip(exact,numeric));assert gap<1e-7,(xx,pattern,eta,gap)
                    max_gap=max(max_gap,gap);counts['nonempty_agree']+=1
    return dict(checks=dict(counts),max_endpoint_gap=max_gap)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    parent=root/'results/baseline_history/development/protocol.json';p=read(parent);audit=root/'results/round_238_audit.json';a=read(audit);assert a['passed']
    hashes=dict(p['source_sha256']);hashes.update(a['extra_source_sha256'])
    for name,value in hashes.items():assert sha(src/name)==value,name
    design=root/'outputs/ttt-pc-alm-research/239_forward_credit_envelope_protocol.md';assert sha(design)==a['report_sha256'][design.name]
    checks=Counter()
    assert model.reachable(.25,(1,))[0]==(2*(F(.25)-model.B),2*(F(.25)+model.B));checks['analytic_interval']=1
    assert model.reachable(0.,(0,))[0]==(F(0),F(0)) and model.reachable(1.,(3,))[0]==(F(0),F(0));checks['closed_boundary']=2
    assert model.reachable(.25,(3,))[0] is None;checks['empty_interval']=1
    assert model.signs((model.EPS,model.EPS),0.)==() and model.signs((F(0),F(0)),.001)==();checks['zero_deadzone']=2
    assert model.chain((1,2,1,2))==(8,-4,-2,1);checks['chain_identity']=1
    toys=[]
    for value,expected in [(.5,'nonpositive'),(.99,'positive')]:
        x=np.array([.25]);v=np.array([value]);reg=np.array([[1]],np.uint8);bb,_,_=model.build(x,v,depth=1)
        for method,bank in bb.items():
            r=hull.solve(x,v,reg,hull.RationalBank(bank));assert r['status']==expected;vv=validate(x,v,reg,bank,r);toys.append(dict(v=value,method=method,verified=vv));checks['two_sided_toys']+=1
    rng=np.random.default_rng(5900001);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=base.forward(xx,teacher)+np.random.default_rng(24900001).uniform(-.001,.001,24);x=xx[:4];v=vv[:4]
    cross=crosscheck_intervals(x);print(json.dumps(cross),flush=True);banks,metadata,details=model.build(x,v)
    old=read(root/'results/exact_credit_hull/primitive/records.json');keys=list(dict.fromkeys(r['pattern'] for r in old));ids=sorted({0,1,len(keys)//2,len(keys)-1});records=[];statuses={}
    for method,bank in banks.items():
        rb=hull.RationalBank(bank);counter=Counter()
        for index in ids:
            key=keys[index];reg=np.frombuffer(bytes.fromhex(key),np.uint8).reshape(4,4);r=hull.solve(x,v,reg,rb);verified=validate(x,v,reg,bank,r)
            records.append(dict(method=method,index=index,pattern=key,result=r,verified=verified));counter[r['status']]+=1
        statuses[method]=dict(counter);print(json.dumps(dict(method=method,metadata=metadata[method],statuses=counter)),flush=True)
    for name in ['forward_credit_envelope.py',Path(__file__).name]:hashes[name]=sha(src/name)
    out=root/'results/forward_credit_envelope/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=dict(checks),toys=toys,interval_crosscheck=cross,preflight_pool_regions=len(keys),preflight_indices=ids,statuses=statuses,metadata=metadata,
        source_sha256=hashes,parent_protocol_sha256=sha(parent),parent_audit_sha256=sha(audit),design_sha256=sha(design),records_sha256=sha(out/'records.json'),
        scope='Exact real-arithmetic outer-cone primitive; no claim that positive envelope directions are jointly realizable')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)

if __name__=='__main__':main()
