"""Independent reduced-variable LP and convex-concave consistency checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import joint_credit_minimax as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def reduced_lp(x,v,reg,bank):
    k,d,n=bank.shape;m=d*n;dim=d+m+1
    zl,zh,hl,hh=[a[0] for a in model.original.screen.boxes(v,reg[None])]
    residual=np.zeros((m,dim));constant=np.zeros(m);branch=[];targets=[]
    for j in range(d):
        for i in range(n):
            t=j*n+i;s=model.original.base.SLOPES[reg[j,i]];c=model.original.base.INTERCEPTS[reg[j,i]]
            z=np.zeros(dim);z[j]=1;offset=x[i] if j==0 else 0
            if j:z[d+(j-1)*n+i]=1
            residual[t,d+t]=1;residual[t]-=s*z;constant[t]=-s*offset-c
            branch.extend([z,-z]);targets.extend([zh[j,i]-offset,offset-zl[j,i]])
    aa=bank.reshape(k,m);credit=aa@residual;credit[:,-1]=-1
    objective=np.zeros(dim);objective[-1]=1
    bounds=[(-.12,.12)]*d+list(zip(hl.ravel(),hh.ravel()))+[(None,None)]
    return linprog(objective,A_ub=np.r_[credit,np.array(branch)],b_ub=np.r_[-aa@constant,targets],bounds=bounds,
        options=dict(dual_feasibility_tolerance=1e-9,primal_feasibility_tolerance=1e-9))


def verify():
    rng=np.random.default_rng(5900001);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=model.original.base.forward(xx,teacher)+np.random.default_rng(24900001).uniform(-.001,.001,24)
    checks=dict(lp_pairs=0,single_direction_exact=0,convex_endpoint_dominance=0,exact_positive_mixtures=0,repetition_identical=0)
    maxgap=0.;maxform=0.
    for n in [4,8]:
        x=xx[:n];v=vv[:n]
        regs=np.array([model.original.base.pattern(x,b) for b in np.r_[teacher[None],rng.uniform(-.12,.12,(7,4))]],np.uint8)
        bank=rng.normal(size=(9,4,n));bank/=abs(bank).max((1,2),keepdims=True)
        for reg in regs:
            data,meta=model.solve(x,v,reg,bank);ref=reduced_lp(x,v,reg,bank)
            assert meta['success']==ref.success
            if not ref.success:assert ref.status==meta['status']==2;continue
            assert abs(meta['numerical_value']-ref.fun)<1e-8;maxform=max(maxform,abs(meta['numerical_value']-ref.fun));checks['lp_pairs']+=1
            maximum=-np.inf
            for a in bank:
                sd,sm=model.solve(x,v,reg,a[None]);exact=model.original.exact_optimum(x,v,reg,a)
                assert sm['success'] and abs(sm['numerical_value']-exact['value'])<1e-8
                maximum=max(maximum,exact['value']);checks['single_direction_exact']+=1
            assert meta['numerical_value']>=maximum-1e-8;checks['convex_endpoint_dominance']+=1
            assert abs(meta['duality_gap'])<1e-8;maxgap=max(maxgap,abs(meta['duality_gap']))
            if meta['positive']:assert meta['proof']['exact_convex_lower']['value']>0;checks['exact_positive_mixtures']+=1
            again,am=model.solve(x,v,reg,bank)
            assert again['credit'].tobytes()==data['credit'].tobytes() and am['proof']==meta['proof'];checks['repetition_identical']+=1
    # Independent exact algebraic counterexample: min_{t in [0,1]} max(t,1-t)=1/2,
    # while each separate directional minimum is 0.
    toy=linprog([0.,1.],A_ub=[[1.,-1.],[-1.,-1.]],b_ub=[0.,-1.],bounds=[(0.,1.),(None,None)])
    assert toy.success and toy.fun==.5 and np.array_equal(-toy.ineqlin.marginals,[.5,.5])
    return dict(passed=True,checks=checks,max_formulation_error=maxform,max_duality_gap=maxgap,
        exact_toy_value=.5,scope='Old preflight seed and algebraic toy only; no claim of native ALM superiority')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/online_primal_gate/development_v2/protocol.json';hashes=dict(json.loads(parent.read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    assert json.loads((root/'results/round_219_audit.json').read_text())['passed']
    result=verify()
    additions=['analyze_online_primal_gate.py','audit_online_primal_skips.py','audit_online_primal_resources.py','plot_primal_gate_components.py',
        'plot_online_primal_contrasts.py','audit_research_round_219.py','joint_credit_minimax.py',Path(__file__).name]
    for name in additions:hashes[name]=sha(Path(__file__).with_name(name))
    result.update(source_sha256=hashes,parent_protocol_sha256=sha(parent),parent_audit_sha256=sha(root/'results/round_219_audit.json'))
    out=root/'results/joint_credit_minimax/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=True,sources=len(hashes),checks=result['checks'])),flush=True)


if __name__=='__main__':main()
