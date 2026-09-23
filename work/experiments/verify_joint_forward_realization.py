"""Inverse construction and bytewise original forward/credit agreement."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import numpy as np
import joint_forward_realization as model
import light_h2_credit as light
import baseline_history_capture as history
import exact_credit_hull as hull
from verify_exact_credit_hull import validate

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def validate_generation(x,v,arrays,records):
    """Direct rational forward, original floating forward and parent BP observer."""
    checks=Counter();points=[[model.unpack(q) for q in p] for p in records['exact_points']]
    assert len(points)==len(arrays['points'])
    for rec in records['sources']:
        i=rec['observation'];bias=points[rec['point']];h=F(float(x[i]));assert all(-F(.12)<=b<=F(.12) for b in bias)
        for j,(b,k) in enumerate(zip(bias,rec['pattern'])):
            z=h+b;bounds=[(None,F(0)),(F(0),F(1,2)),(F(1,2),F(1)),(F(1),None)][k]
            assert (bounds[0] is None or z>=bounds[0]) and (bounds[1] is None or z<=bounds[1]);h=max(F(0),1-abs(2*z-1))
        assert h==model.unpack(rec['target']);vf=F(float(v[i]));assert h<vf-F(.001) if rec['sign']==1 else h>vf+F(.001);checks['exact_source_chains']+=1
    for i,bias in enumerate(points):
        point=np.clip(np.array([float(b) for b in bias]),-.12,.12);assert point.tobytes()==arrays['points'][i].tobytes()
        original=light.base.forward(x,point);assert original.tobytes()==arrays['outputs'][i].tobytes();checks['original_shared_forwards']+=1
    observer=light.Snapshots(x,v,light.config('adam60','native'));out=observer.evaluate(arrays['points'],x,v,with_jacobian=False)
    assert out[1].tobytes()==arrays['loss_residual'].tobytes() and out[3].tobytes()==arrays['raw_error'].tobytes()
    assert observer.previous.tobytes()==arrays['current_bp'].tobytes();checks['original_bp_credit_arrays']=1
    assert arrays['raw'].tobytes()==np.concatenate([arrays['current_bp'],arrays['output_credit']]).tobytes()
    for rec in records['sources']:
        p=rec['point'];i=rec['observation'];wanted=np.array([0,2,-2,0])[rec['pattern']]
        assert rec['actual_sign_preserved']==(int(np.sign(observer.previous[p,-1,i]))==rec['sign'])
        assert rec['actual_slopes_preserved']==np.array_equal(arrays['slopes'][p,:,i],wanted)
    return dict(checks)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    parent=root/'results/forward_credit_envelope/development/protocol.json';p=read(parent);audit=root/'results/round_240_audit.json';a=read(audit);assert a['passed']
    hashes=dict(p['source_sha256']);hashes.update(a['extra_source_sha256'])
    for name,value in hashes.items():assert sha(src/name)==value,name
    design=root/'outputs/ttt-pc-alm-research/241_joint_forward_realization_protocol.md';assert sha(design)==a['report_sha256'][design.name]
    toys=[];counts=Counter()
    # Includes flat branches, turning points, and exactly zero output targets.
    for x,v in [(np.array([0.,.5,1.]),np.array([.2,.8,.2])),(np.array([.25]),np.array([.99]))]:
        arrays,records,meta=model.generate(x,v,depth=1);check=validate_generation(x,v,arrays,records);counts.update(check);toys.append(dict(meta=meta,checks=check))
    rng=np.random.default_rng(5900001);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=light.base.forward(xx,teacher)+np.random.default_rng(24900001).uniform(-.001,.001,24);x=xx[:4];v=vv[:4]
    arrays,records,meta=model.generate(x,v);check=validate_generation(x,v,arrays,records);counts.update(check)
    old,oldarrays,_=history.capture(x,v,'adam60',True);banks=dict(constructed_joint=arrays['bank'],old_plus_constructed_joint=np.concatenate([oldarrays['bank'],arrays['bank']]))
    prior=read(root/'results/exact_credit_hull/primitive/records.json');keys=list(dict.fromkeys(r['pattern'] for r in prior));ids=sorted({0,1,len(keys)//2,len(keys)-1});rows=[];statuses={}
    for method,bank in banks.items():
        rb=hull.RationalBank(bank);counter=Counter()
        for i in ids:
            reg=np.frombuffer(bytes.fromhex(keys[i]),np.uint8).reshape(4,4);r=hull.solve(x,v,reg,rb);q=validate(x,v,reg,bank,r);counter[r['status']]+=1
            rows.append(dict(method=method,index=i,pattern=keys[i],result=r,verified=q))
        statuses[method]=dict(counter);print(json.dumps(dict(method=method,statuses=counter)),flush=True)
    for name in ['joint_forward_realization.py',Path(__file__).name]:hashes[name]=sha(src/name)
    out=root/'results/joint_forward_realization/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'records.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'generation.json').write_text(json.dumps(records,indent=2),encoding='utf-8');np.savez_compressed(out/'arrays.npz',**arrays)
    result=dict(passed=True,checks=dict(counts),toys=toys,metadata=meta,preflight_indices=ids,statuses=statuses,source_sha256=hashes,parent_protocol_sha256=sha(parent),parent_audit_sha256=sha(audit),
        design_sha256=sha(design),output_sha256={name:sha(out/name) for name in ['records.json','generation.json','arrays.npz']})
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','toys']}),flush=True)

if __name__=='__main__':main()
