"""Old-seed and small exact controls before the common-pool hull experiment."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np
import exact_credit_hull as model
import credit_history_capture as history
from audit_joint_credit_minimax import exact_mixture,rational_optimum
from verify_joint_credit_minimax import reduced_lp


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def independent_witness(x,v,reg,bank,witness):
    """Independent Fraction arithmetic, without the integer-scaled bank helper."""
    y=[F(int(n),int(d)) for n,d in witness['y']];depth,n=reg.shape;bias=y[:depth]
    h=np.array(y[depth:],dtype=object).reshape(depth,n);previous=[F(float(t)) for t in x];r=[]
    zl,zh,hl,hh=[arr[0] for arr in model.joint.original.screen.boxes(v,reg[None])]
    assert all(-F(.12)<=b<=F(.12) for b in bias)
    for layer in range(depth):
        for i in range(n):
            z=previous[i]+bias[layer]
            assert F(float(zl[layer,i]))<=z<=F(float(zh[layer,i]))
            assert F(float(hl[layer,i]))<=h[layer,i]<=F(float(hh[layer,i]))
            r.append(h[layer,i]-int(model.joint.original.base.SLOPES[reg[layer,i]])*z-int(model.joint.original.base.INTERCEPTS[reg[layer,i]]))
        previous=h[layer]
    vals=[sum((F(float(a))*b for a,b in zip(row,r)),F(0)) for row in bank.reshape(len(bank),-1)]
    mx=max(vals);assert mx<=0
    assert [str(mx.numerator),str(mx.denominator)]==witness['proof']['max_credit']
    return dict(directions=len(vals),max_credit=float(mx))


def validate(x,v,reg,bank,row):
    ref=reduced_lp(x,v,reg,bank)
    assert ref.success==row['dual']['success']
    if ref.success:assert abs(ref.fun-row['dual']['numerical_value'])<1e-8
    if row['status']=='positive':
        w=np.array(row['weights']);assert np.all(w>=0) and w.sum()>0
        a=exact_mixture(bank,w);value=rational_optimum(x,v,reg,a)
        lower=model.unpack([row['dual']['proof']['exact_convex_lower'][k] for k in ['numerator','denominator']])
        assert value>=lower>0
        return dict(status='positive',exact=float(value))
    if row['status']=='nonpositive':return dict(status='nonpositive',**independent_witness(x,v,reg,bank,row['witness']))
    assert row['status']=='unknown';return dict(status='unknown')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/credit_wall_budget/development/protocol.json';p=json.loads(parent.read_text());hashes=dict(p['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    audit=root/'results/round_234_audit.json';assert json.loads(audit.read_text())['passed']
    checks=Counter();toys=[]
    for value,label,expected in [(.5,'overlap','nonpositive'),(.99,'separated','positive'),(.741001+1e-10,'tiny_positive','positive'),(.741001-1e-10,'tiny_negative','nonpositive')]:
        x=np.array([.25]);v=np.array([value]);reg=np.array([[1]],np.uint8);bank=np.ones((1,1,1));rb=model.RationalBank(bank)
        row=model.solve(x,v,reg,rb);assert row['status']==expected,(label,row);result=validate(x,v,reg,bank,row);toys.append(dict(label=label,**result));checks['toy_cases']+=1
        if expected=='nonpositive':
            y=[model.unpack(t) for t in row['witness']['y']];bad=y.copy();bad[0]=F(1)
            assert not model.check_witness(x,v,reg,rb,bad)['accepted'];checks['bad_witness_rejected']+=1
    # A common witness at a boundary: b=.12, h=.74 exactly in rational arithmetic.
    x=np.array([.25]);v=np.array([.74]);reg=np.array([[1]],np.uint8);bank=np.array([[[1.]],[[-1.]]])
    y=[F(.12),2*(F(.25)+F(.12))];proof=model.check_witness(x,v,reg,model.RationalBank(bank),y);assert proof['accepted']
    independent_witness(x,v,reg,bank,dict(y=[model.pack(q) for q in y],proof=proof));checks['boundary_witness']=1
    # Exact toy: r(t)=(t,1-t), each endpoint direction has min=0, half mixture has min=1/2.
    eq=model.exact_linear_solve([[1,-1],[-1,-1]],[0,-1]);assert eq==[F(1,2),F(1,2)]
    assert min(F(0),F(1))==0 and (F(1,2)*F(0)+F(1,2)*F(1))==F(1,2);checks['convex_synergy_toy']=1
    rng=np.random.default_rng(5900001);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=history.light.base.forward(xx,teacher)+np.random.default_rng(24900001).uniform(-.001,.001,24);x=xx[:4];v=vv[:4]
    captured=history.capture(x,v,True);regs=captured[2];banks,_=history.build(captured[6]);statuses={};records=[]
    for method in ['native',*history.VARIANTS]:
        bank=banks[method];rb=model.RationalBank(bank);status=Counter()
        for i,reg in enumerate(regs):
            row=model.solve(x,v,reg,rb);verified=validate(x,v,reg,bank,row);status[row['status']]+=1;checks['old_seed_regions']+=1
            if row['status']=='nonpositive':checks['exact_primal_directions']+=len(bank)
            records.append(dict(method=method,index=i,pattern=reg.tobytes().hex(),verified=verified,result=row))
        statuses[method]=dict(status);print(json.dumps(dict(method=method,statuses=status)),flush=True)
    assert sum(r['verified']['status']=='positive' for r in records)>0 and sum(r['verified']['status']=='nonpositive' for r in records)>0
    for name in ['audit_credit_wall_budget.py','audit_credit_wall_resources.py','plot_credit_wall_budget.py','audit_research_round_234.py',
                 'exact_credit_hull.py',Path(__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    out=root/'results/exact_credit_hull/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=dict(checks),toys=toys,statuses=statuses,source_sha256=hashes,parent_protocol_sha256=sha(parent),
        parent_audit_sha256=sha(audit),design_sha256=sha(root/'outputs/ttt-pc-alm-research/235_exact_credit_hull_separation_protocol.md'),records_sha256=sha(out/'records.json'))
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
