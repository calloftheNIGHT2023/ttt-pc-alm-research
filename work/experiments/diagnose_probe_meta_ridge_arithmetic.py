"""Arithmetic-stage and high-precision diagnosis; no latent teacher access."""
import argparse
import json
import os
from pathlib import Path
import time
import mpmath as mp
import numpy as np
import torch
import probe_credit_confirmation_suite as suite
from probe_confirmation_independent_heads import values
from analyze_recovered_online_comparison import forward
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def mpmatrix(a):
    a=np.asarray(a)
    if a.ndim==1: return mp.matrix([mp.mpf(float(t)) for t in a])
    return mp.matrix([[mp.mpf(float(t)) for t in row] for row in a])


def raw_features(z,bank):
    answer=mp.matrix(len(z),len(bank))
    for i,x in enumerate(z):
        for j,bias in enumerate(bank):
            h=mp.mpf(float(x))
            for b in bias: h=max(mp.mpf(0),1-abs(2*(h+mp.mpf(float(b)))-1))
            answer[i,j]=h
    return answer


def means_centered(raw):
    mu=mp.matrix([mp.fsum(raw[i,j] for j in range(raw.cols))/raw.cols for i in range(raw.rows)])
    return mu,mp.matrix([[raw[i,j]-mu[i] for j in range(raw.cols)] for i in range(raw.rows)])


def high_precision(model,x,v,q,dps):
    with mp.workdps(dps):
        projection=mpmatrix(values(model.projection)); weights=mpmatrix(values(model.mean_weights))
        bank=values(model.bank); base,centered=means_centered(raw_features(x,bank))
        phi=centered*projection; mu=base+centered*weights
        ridge=min(mp.mpf(.1),max(mp.mpf(1e-9),mp.exp(mp.mpf(float(values(model.log_ridge)[0])))))
        kernel=phi*phi.T+ridge*mp.eye(len(x)); alpha=mp.lu_solve(kernel,mpmatrix(v)-mu)
        beta=phi.T*alpha
        # Associate the same exact linear maps differently, in high precision.
        read_weights=weights+projection*beta
        query_mean,query_centered=means_centered(raw_features(q,bank))
        prediction=query_mean+query_centered*read_weights
        clipped=np.array([float(min(1,max(0,t))) for t in prediction])
        return clipped,np.array([float(t) for t in beta]),dict(ridge=str(ridge),dps=dps,
            maximum_dual_residual=str(max(abs(t) for t in kernel*alpha-(mpmatrix(v)-mu))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();base=root/'results/probe_credit_confirmation'
    inp=base/'predictions';diagnosis=base/'head_discrepancy_diagnosis';out=base/'meta_ridge_arithmetic_diagnosis'
    assert not out.exists() and not (base/'evaluation_v2').exists()
    ds=json.loads((diagnosis/'summary.json').read_text());assert not ds['query_targets_accessed']
    for n,h in ds['outputs_sha256'].items():assert sha(diagnosis/n)==h
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    loaded,manifest=suite.resources.legacy.oldfit.meta.load(root)
    assert manifest==json.loads((inp/'protocol.json').read_text())['checkpoint_manifest']
    records=json.loads((diagnosis/'heads.json').read_text());results=[];arrays={};started=time.perf_counter()
    for row in records:
        if not row['method'].startswith('cold__meta_ridge'):continue
        name=row['method'];model=loaded[name.removeprefix('cold__')]
        assert sha(inp/row['saved_file'])==row['saved_sha256']
        with np.load(inp/row['saved_file']) as z:a={k:z[k].copy() for k in z.files}
        x,v,q=a['x_observed'],a['v_observed'],a['q_observed'];saved=a['fast_0'][0,:,0]
        bank=values(model.bank);projection=values(model.projection);weights=values(model.mean_weights)
        def npfeatures(z):
            raw=forward(z,bank).T;mu=raw.mean(1);centered=raw-mu[:,None]
            return centered@projection,mu+centered@weights
        phi,mu=npfeatures(x);qp,qmu=npfeatures(q)
        ridge=float(np.clip(np.exp(values(model.log_ridge)[0]),1e-9,.1))
        kernel=phi@phi.T+ridge*np.eye(len(x));alpha=np.linalg.solve(kernel,v-mu);beta=phi.T@alpha
        with torch.no_grad():
            tp,tm=model.features(torch.as_tensor(x[None],dtype=torch.float64))
            tq,tqm=model.features(torch.as_tensor(q[None],dtype=torch.float64))
            tr=model.log_ridge[0].exp().clamp(1e-9,.1)
            tk=tp@tp.transpose(-2,-1)+tr*torch.eye(len(x),dtype=torch.float64)
            ta=torch.linalg.solve(tk,torch.as_tensor(v[None,:,None])-tm[...,None])
            tb=(tp.transpose(-2,-1)@ta).numpy()[0,:,0]
        assert tb.tobytes()==saved.tobytes()
        np_prediction=np.clip(qmu+qp@beta,0,1)
        fixed_state=np.clip(qmu+qp@saved,0,1)
        hp70,b70,n70=high_precision(model,x,v,q,70)
        hp110,b110,n110=high_precision(model,x,v,q,110)
        assert hp70.tobytes()==hp110.tobytes() and b70.tobytes()==b110.tobytes()
        answer=dict(method=name,seed=ds['seed'],ridge_numpy=ridge,ridge_torch=float(tr),
            kernel_condition_2=float(np.linalg.cond(kernel)),kernel_eigenvalues=np.linalg.eigvalsh(kernel).tolist(),
            support_feature_gap=float(np.max(abs(phi-values(tp)[0]))),
            support_mean_gap=float(np.max(abs(mu-values(tm)[0]))),
            kernel_gap=float(np.max(abs(kernel-values(tk)[0]))),
            dual_solution_gap=float(np.max(abs(alpha-values(ta)[0,:,0]))),
            query_feature_gap=float(np.max(abs(qp-values(tq)[0]))),
            saved_state_independent_readout_gap=float(np.max(abs(fixed_state-a['prediction']))),
            numpy_full_replay_gap=float(np.max(abs(np_prediction-a['prediction']))),
            original_high_precision_gap=float(np.max(abs(hp110-a['prediction']))),
            numpy_high_precision_gap=float(np.max(abs(hp110-np_prediction))),
            original_state_high_precision_gap=float(np.max(abs(b110-saved))),
            numpy_state_high_precision_gap=float(np.max(abs(b110-beta))),
            precision_repeat_identical_after_float_conversion=True,precision_runs=[n70,n110],
            original_replay_state_bytewise_equal=True)
        arrays[name+'__hp_prediction']=hp110;arrays[name+'__hp_beta']=b110
        arrays[name+'__np_prediction']=np_prediction;arrays[name+'__fixed_state_readout']=fixed_state
        results.append(answer);print(json.dumps(answer),flush=True)
    out.mkdir();exclusive_json(out/'diagnostics.json',results)
    with (out/'arrays.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    result=dict(diagnosis_complete=True,audit_gate_passed=False,query_targets_accessed=False,
        seed=ds['seed'],methods=2,seconds=time.perf_counter()-started,source_sha256=sha(Path(__file__)),
        input_diagnosis_sha256=sha(diagnosis/'summary.json'),
        outputs_sha256={n:sha(out/n) for n in ['diagnostics.json','arrays.npz']})
    exclusive_json(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
