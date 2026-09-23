"""Exact smooth local-minimum certificate for the support-selected witness."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
import multiplier_fixed_point_exact as core
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/multiplier_fixed_point';out=base/'witness/stationary_basin.json';assert not out.exists()
    selection=json.loads((base/'audit/witness_selection.json').read_text());assert len(selection)==1;seed=selection[0]['seed'];rid=selection[0]['restart']
    c=next(t for t in json.loads((base/'exact/certificates.json').read_text()) if (t['seed'],t['restart'])==(seed,rid));b,_=core.decode(c['state'])
    a=np.load(base/'screen'/f'{seed}.npz');x=[F(float(t)) for t in a['x']];v=[F(float(t)) for t in a['v']];d=len(b);n=len(x)
    values=x.copy();jac=[[F(0)]*d for _ in x];constraints=[];radii=[core.B-abs(t) for t in b]
    for j,bb in enumerate(b):
        for i in range(n):
            z=values[i]+bb;jac[i][j]+=1;norm=sum(abs(t) for t in jac[i]);distance=min(abs(z-k) for k in core.K)
            assert distance>0
            if norm:radii.append(distance/norm)
            constraints.append(dict(layer=j,observation=i,preactivation=core.pack(z),coefficient=[core.pack(t) for t in jac[i]],margin=core.pack(distance)))
            s=core.S[core.branch(z)];values[i]=core.g(z);jac[i]=[s*t for t in jac[i]]
    raw=[p-t for p,t in zip(values,v)];res=[]
    for i,r in enumerate(raw):
        assert abs(r)>core.EPS
        norm=sum(abs(t) for t in jac[i]);margin=abs(r)-core.EPS
        if norm:radii.append(margin/norm)
        res.append((1 if r>0 else -1)*margin)
    grad=[sum((row[j]*rr for row,rr in zip(jac,res)),F(0))/n for j in range(d)]
    assert not any(grad);radius=min(radii)/2;assert radius>0
    gram=[[sum((row[j]*row[k] for row in jac),F(0))/n for k in range(d)] for j in range(d)]
    result=dict(passed=True,seed=seed,restart=rid,source_sha256=sha(Path(__file__)),certificates_sha256=sha(base/'exact/certificates.json'),
        exact_smooth_interior=True,radius_infinity=core.pack(radius),radius_float=float(radius),jacobian=[[core.pack(t) for t in row] for row in jac],
        residual=[core.pack(t) for t in res],gradient=[core.pack(t) for t in grad],hessian_gram=[[core.pack(t) for t in row] for row in gram],
        preactivation_constraints=constraints,identity='For ||delta||_infinity < radius: L(b+delta)-L(b)=||J delta||^2/(2n)>=0',
        optimizer_scope='Exact zero-moment Adam, ordinary gradient descent and positive-damping GN remain fixed; does not exclude perturbation, restart, different loss, or global search')
    dump(out,result);print(json.dumps(dict(passed=True,radius=float(radius),jacobian=[[int(t) for t in row] for row in jac])),flush=True)

if __name__=='__main__':main()
