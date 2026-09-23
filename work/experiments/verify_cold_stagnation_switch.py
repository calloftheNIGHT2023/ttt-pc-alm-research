"""Before efficacy: unchanged paths, causal trigger, no global BP, meta provenance."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
import cold_stagnation_switch as cold
import band_conditioned_meta as meta
import run_recovered_online_comparison as regression
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/cold_stagnation_switch/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);parent=json.loads((root/'results/round_246_audit.json').read_text());hashes=dict(parent['source_sha256'])
    for n,h in hashes.items():assert sha(src/n)==h,n
    design=root/'outputs/ttt-pc-alm-research/247_cold_start_stagnation_switch_protocol.md';assert sha(design)==parent['report_sha256'][design.name]
    hashes.update({n:sha(src/n) for n in ['cold_stagnation_switch.py',Path(__file__).name]});checks=Counter();rng=np.random.default_rng(247731)
    with cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for n in [4,8,24]:
            x=rng.uniform(0,1,n);v=rng.uniform(0,1,n);starts=rng.uniform(-.12,.12,(8,4));h=[];prev=x
            for j in range(4):h.append(cold.base.g(prev+starts[:,j,None]));prev=h[-1]
            for method in ['alm','nodual','pc']:
                best,arrays,_=cold.run_local(starts,x,v,method,128)
                expected=cold.frozen.local(starts,np.array(h),starts,x,v,method,128)
                for k in ['b','h','u','best']:assert arrays[k].tobytes()==expected[k].tobytes();checks['pure128_path_arrays']+=1
                simple,_,_=cold.run_local(starts,x,v,method,128,False);assert best.tobytes()==simple.tobytes();checks['trace_toggle_best']+=1
        # Boundary tests for the exact fixed numerical trigger predicate.
        cases=[(1e-12,1e-12,1.00001e-6,.00100101,False,True),(1.00001e-12,0.,1.,1.,False,False),
            (0.,1.00001e-12,1.,1.,False,False),(0.,0.,1e-6,1.,False,False),(0.,0.,1.,.001001,False,False),(0.,0.,1.,1.,True,False)]
        for a,b,r,e,active,want in cases:
            assert bool(cold.trigger(np.array([a]),np.array([b]),np.array([r]),np.array([e]),np.array([active]))[0])==want;checks['trigger_boundaries']+=1
        x,v=observations(5900001);x=x[:4];v=v[:4];starts=cold.starts();saved=(cold.bp.evaluate,cold.base.forward_jacobian,torch.autograd.grad)
        def forbidden(*a,**k):raise AssertionError('global BP entered local switch')
        try:
            cold.bp.evaluate=cold.base.forward_jacobian=torch.autograd.grad=forbidden
            best,a,mm=cold.run_local(starts,x,v,'switch',128)
            other,_,_=cold.run_local(starts,x,v,'switch',128,False);assert best.tobytes()==other.tobytes();checks['no_global_bp_switch']+=1
        finally:cold.bp.evaluate,cold.base.forward_jacobian,torch.autograd.grad=saved
        # Causal batch equivalence to17 independent restarts (no cross-restart state).
        for i in range(17):
            bi,ai,mi=cold.run_local(starts[i:i+1],x,v,'switch',128)
            for key in ['b','best']:assert ai[key][:,0].tobytes()==a[key][:,i].tobytes()
            for key in ['h','u']:assert ai[key][:,:,0].tobytes()==a[key][:,:,i].tobytes()
            assert mi['first_trigger']==[mm['first_trigger'][i]];checks['independent_restart_replays']+=1
        for method,steps in [('adam',240),('gauss_newton',40)]:
            b,a,_=cold.run_bp(starts,x,v,method,steps);ref,_=cold.bp.refine(starts,x,v,np.zeros(4),solver=method,steps=steps)
            assert b.tobytes()==ref.tobytes();checks['bp_observer_bitwise']+=1
    loaded,manifest=meta.load(root);oldp=json.loads((root/'results/scalar_matched_batched/meta_prior/protocol.json').read_text())
    assert oldp['train_seed']==874600 and oldp['initial_seed']==874601 and oldp['steps']==2000 and oldp['batch_size']==16
    assert set(oldp['validation_seeds']).isdisjoint(range(5900000,5900016))
    # Evaluation used these development tasks historically, but the training
    # program selects checkpoints only using its distinct validation set.
    names=['meta_ridge64','meta_ridge128','meta_shallow64_5','meta_shallow64_20']
    q=np.linspace(0,1,257)
    for name in names:
        pred,state,m=meta.fit(loaded[name],x,v,None)
        with torch.no_grad():expected=loaded[name].predict(loaded[name].adapt(torch.as_tensor(x[None]),torch.as_tensor(v[None])),torch.as_tensor(q[None])).clamp(0,1)[0].numpy()
        assert pred(q).tobytes()==expected.tobytes();checks['frozen_meta_replays']+=1
    ridge,_,_=regression.regression(x,v,'residual_linear_ridge',None);rls,_,_=regression.regression(x,v,'residual_linear_rls',None)
    assert np.max(abs(ridge(q)-rls(q)))<1e-10;checks['ridge_rls_equivalence']+=1
    result=dict(passed=True,source_sha256=hashes,parent_audit_sha256=sha(root/'results/round_246_audit.json'),design_sha256=sha(design),checks=checks,
        checkpoint_manifest={name:manifest[name] for name in names},meta_protocol_sha256=sha(root/'results/scalar_matched_batched/meta_prior/protocol.json'),
        meta_training_scope='fixed prior, independent synthetic meta training; validation5900032-47; current tasks historically evaluated but not checkpoint-selected; no new training',query_targets_accessed=False)
    dump(out/'summary.json',result);print(json.dumps(dict(passed=True,checks=checks,source_count=len(hashes))),flush=True)

if __name__=='__main__':main()
