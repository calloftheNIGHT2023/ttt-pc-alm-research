"""Before online efficacy runs: exact inverse chains and shared-start controls."""
import argparse
from collections import Counter
from fractions import Fraction as F
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import band_conditioned_online as model
import band_conditioned_meta as meta_model
import band_conditioned_initialization as init

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def validate(x,v,points,records,meta,depth):
    exact=[tuple(init.unpack(q) for q in p) for p in records['exact_points']];assert len(set(exact))==len(exact)==len(points)
    expected=np.clip(np.array([[float(q) for q in p] for p in exact]).reshape(-1,depth),-.12,.12)
    assert expected.tobytes()==points.tobytes();seen=set();counts=Counter()
    for rec in records['sources']:
        i=rec['observation'];pattern=rec['pattern'];assert (i,tuple(pattern)) not in seen;seen.add((i,tuple(pattern)))
        h=F(float(x[i]));b=exact[rec['point']]
        for bias,k in zip(b,pattern):
            assert -F(.12)<=bias<=F(.12);z=h+bias
            lo,hi=init.envelope.INTERVALS[k];assert (lo is None or z>=lo) and (hi is None or z<=hi)
            h=max(F(0),1-abs(2*z-1))
        assert h==init.unpack(rec['target']) and abs(h-F(float(v[i])))<=F(.001)
        assert rec['actual_output']==float(model.base.forward(x[i:i+1],points[rec['point']])[0])
        counts['exact_nonempty_chains']+=1
    # Exhaust every closed pattern, not only returned sources. Exact interval
    # reachability was independently LP-audited in round240; use its frozen API.
    for i,(xx,vv) in enumerate(zip(x,v)):
        for p in product(range(4),repeat=depth):
            interval,_=init.envelope.reachable(xx,p)
            nonempty=interval is not None and max(F(0),interval[0],F(float(vv))-F(.001))<=min(F(1),interval[1],F(float(vv))+F(.001))
            assert nonempty==((i,p) in seen);counts['all_single_observation_patterns']+=1
    assert meta['empty_intersections']+len(seen)==len(x)*4**depth
    assert meta['max_float_band_excess']<1e-12
    return counts

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    parent=read(root/'results/joint_forward_realization/development/protocol.json');audit=read(root/'results/round_242_audit.json');assert audit['passed']
    sources=dict(parent['source_sha256']);sources.update(audit['extra_source_sha256'])
    for name,h in sources.items():assert sha(src/name)==h,name
    extra=['band_conditioned_initialization.py','band_conditioned_online.py','band_conditioned_meta.py','matched_shifted_meta_models.py','run_matched_shifted_meta.py',Path(__file__).name]
    sources.update({name:sha(src/name) for name in extra})
    design=root/'outputs/ttt-pc-alm-research/243_band_conditioned_online_protocol.md';assert sha(design)==audit['report_sha256'][design.name]
    out=root/'results/band_conditioned_online/primitive_v2';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    xx,vv=model.original.previous.olddriver.observations(5900001);checks=Counter();generated=[]
    for name,x,v,depth in [('knots',np.array([0.,.5,1.]),np.array([0.,1.,0.]),1),('unreachable',np.array([0.]),np.array([1.]),1),('old5900001',xx[:4],vv[:4],4)]:
        points,records,gm=init.generate(x,v,depth);checks.update(validate(x,v,points,records,gm,depth));np.savez_compressed(out/f'{name}.npz',x=x,v=v,points=points)
        (out/f'{name}.json').write_text(json.dumps(dict(records=records,metadata=gm),indent=2),encoding='utf-8');generated.append(dict(name=name,metadata=gm))
    print(json.dumps(dict(phase='inverse',checks=checks)),flush=True)
    loaded,manifest=meta_model.load(root);results={};start_hashes={};q=np.linspace(0,1,61);configs=model.configs();rng_seed=484341
    core=model.neighbor.previous.core
    def forbidden(*a,**k):raise AssertionError('global BP entered a local candidate')
    for cfg in configs:
        state=None;trajectory=[]
        local=cfg.get('learner') in ['alm','pc','nodual','direct64','direct4096']
        jac=model.base.forward_jacobian;refine=core.batched.refine;grad=torch.autograd.grad
        try:
            if local:model.base.forward_jacobian=forbidden;core.batched.refine=forbidden;torch.autograd.grad=forbidden
            for n in [4,8,16,24]:
                if cfg['family']=='meta':
                    pred,new,mm=meta_model.fit(loaded[cfg['name']],xx[:n],vv[:n],state)
                    with torch.no_grad():
                        ref=loaded[cfg['name']].adapt(torch.as_tensor(xx[None,:n]),torch.as_tensor(vv[None,:n]),state)
                        expected=loaded[cfg['name']].predict(ref,torch.as_tensor(q[None])).clamp(0,1)[0].numpy()
                    assert pred(q).tobytes()==expected.tobytes();checks['frozen_meta_prediction_stages']+=1
                    arr=meta_model.arrays(new)
                elif cfg['family']=='regression':
                    pred,new,mm=model.original.previous.regression(xx[:n],vv[:n],cfg['name'],state);arr={}
                    if new is not None:arr.update(rls_w=new['w'],rls_p=new['p'])
                else:
                    pred,new,mm=model.fit(xx[:n],vv[:n],state,cfg,np.random.default_rng(rng_seed+n),64);arr=dict(points=new.samples,anchor=new.anchor)
                    assert model.original.memory.prior.state_digest(state)==mm['previous_state_digest']
                    if local:checks['local_no_global_bp_stages']+=1
                    if n==4 and cfg.get('band_pool'):
                        bm=mm['discovery']['band_initialization'];start_hashes.setdefault(cfg['band_pool'],set()).add(bm['starts_sha256'])
                        assert bm['pool_candidates']<=1280;checks['shared_pool_initializations']+=1
                    if n>4:assert 'band_initialization' not in mm['discovery']
                arr=dict(arr,prediction=pred(q));assert np.isfinite(arr['prediction']).all();trajectory.append(arr);state=new;checks['online_stages']+=1
        finally:model.base.forward_jacobian=jac;core.batched.refine=refine;torch.autograd.grad=grad
        results[cfg['name']]=trajectory
        print(json.dumps(dict(phase='old_stream',method=cfg['name'],stages=4)),flush=True)
    assert all(len(v)==1 for v in start_hashes.values())
    # Native-credit screening cannot change the posterior state or prediction.
    for cfg in [c for c in configs if c.get('primal_upper_gate')]:
        control=f'{cfg["band_pool"]}_{cfg["learner"]}_c5'
        for a,b in zip(results[cfg['name']],results[control]):
            assert all(a[k].tobytes()==b[k].tobytes() for k in a);checks['gated_c5_states_bytewise']+=1
    # Original prior256 execution remains unchanged, including the new stronger
    # Adam/GN C5 settings. No inverse-generation hook used by the reference.
    for cfg in [c for c in configs if c.get('band_pool')=='prior256']:
        refcfg={k:v for k,v in cfg.items() if k!='band_pool'};state=None
        for n,actual in zip([4,8,16,24],results[cfg['name']]):
            pred,state,_=model.original.fit(xx[:n],vv[:n],state,refcfg,np.random.default_rng(rng_seed+n),64)
            assert actual['points'].tobytes()==state.samples.tobytes() and actual['prediction'].tobytes()==pred(q).tobytes();checks['prior256_original_stages_bytewise']+=1
    # The fixed strong prior control must not pay or call inverse construction.
    old_generate=init.generate
    try:
        init.generate=forbidden
        for kind in ['prior256','prior1280']:
            cfg=next(c for c in configs if c['name']==f'{kind}_direct64_c5')
            model.fit(xx[:4],vv[:4],None,cfg,np.random.default_rng(rng_seed+4),64);checks['prior_controls_no_inverse_generation']+=1
    finally:init.generate=old_generate
    ridge=results['residual_linear_ridge'];rls=results['residual_linear_rls']
    for a,b in zip(ridge,rls):assert np.max(abs(a['prediction']-b['prediction']))<1e-9;checks['ridge_rls_stages_equivalent']+=1
    result=dict(passed=True,checks=dict(checks),generated=generated,shared_starts_sha256={k:next(iter(v)) for k,v in start_hashes.items()},
        source_sha256=sources,checkpoint_manifest=manifest,design_sha256=sha(design),parent_audit_sha256=sha(root/'results/round_242_audit.json'),
        scope='Old seed preflight and boundary toys only; no query truth or efficacy selection',
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=True,checks=checks,sources=len(sources))),flush=True)

if __name__=='__main__':main()
