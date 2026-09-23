"""262 old-task compatibility, nested priors and frozen failure policy."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import matched_budget_suite as suite
from run_independent_hybrid_memory import observations
from analyze_recovered_online_comparison import forward
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/matched_budget_confirmation/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    parent=root/'results/round_261_audit.json';old=json.loads(parent.read_text());hashes=dict(old['source_sha256'])
    for n,h in hashes.items():assert sha(src/n)==h,n
    hashes.update({n:sha(src/n) for n in ['expanded_cold_readout.py','matched_budget_suite.py',Path(__file__).name]});design=root/'outputs/ttt-pc-alm-research/262_matched_budget_confirmation_protocol.md';assert sha(design)==old['report_sha256'][design.name]
    loaded,manifest=suite.oldfit.meta.load(root);cfgs=suite.catalogue();p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=sha(design),configs=cfgs,checkpoint_manifest=manifest,prior_sha256=suite.hashes_of_priors(),seed=5900001,query_targets_accessed=False)
    dump(out/'protocol.json',p);counts=Counter();x,v=observations(p['seed']);x=x[:4];v=v[:4];q=np.linspace(0,1,257);begin=time.perf_counter();rows=[]
    for small,large in [(17,33),(33,65)]:assert suite.expanded.starts(small).tobytes()==suite.expanded.starts(large)[:small].tobytes();counts['nested_start_pairs']+=1
    for small,large in [(4096,16384),(16384,65536)]:assert suite.prior_bank(small).tobytes()==suite.prior_bank(large)[:small].tobytes();counts['nested_feature_pairs']+=1
    with suite.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for cfg in cfgs:
            a,m=suite.guarded_fit(cfg,x,v,q,p['seed'],loaded);assert not m['execution_failed'];counts['finite_catalogue_calls']+=1;counts['existing_bytewise_predictors']+=suite.check_existing(root,p['seed'],cfg,a,m)
            path=out/f"{cfg['name']}.npz";np.savez_compressed(path,**a);rows.append(dict(method=cfg['name'],file=path.name,sha256=sha(path),metadata=m));print(json.dumps(dict(completed=len(rows),total=len(cfgs),method=cfg['name'])),flush=True)
            if cfg['family']=='cold':
                aa,mm=suite.expanded.fit({**cfg['config'],'restarts':17},x,v,q,p['seed'])
                for k in ['points','allocation','prediction','selected_b','best_bank']:assert aa[k].tobytes()==a[k].tobytes();counts['expanded_17_reference_arrays']+=1
        for method,n in [('adam',240),('gauss_newton',40)]:
            a,_,_=suite.cold.run_bp(suite.expanded.starts(17),x,v,method,n,False);b,_,_=suite.cold.run_bp(suite.expanded.starts(65),x,v,method,n,False)
            assert a.tobytes()==b[:17].tobytes();counts['independent_bp_restart_prefixes']+=1
    maxgap=0.
    for n in [4096,16384,65536]:
        a,m=suite.prior_fit(n,x,v,q);bank=suite.prior_bank(n);phi=forward(x,bank);mu=phi.mean(0);cov=phi.T@phi/n-np.outer(mu,mu)+np.eye(len(x))*suite.prior.base.EPS**2/3;alpha=np.linalg.solve(cov,v-mu)
        z=q[[0,50,128,256]];phiq=forward(z,bank);expected=phiq.mean(0)+(phiq.T@phi/n-np.outer(phiq.mean(0),mu))@alpha;gap=float(np.max(abs(expected-a['prediction'][[0,50,128,256]])));maxgap=max(maxgap,gap);assert gap<1e-9;counts['independent_prior_moment_checks']+=1
        if n==4096:
            cfg=next(c for c in cfgs if c['name']=='cold__prior4096_ridge');aa,_=suite.original.fit(cfg,x,v,q,p['seed'],loaded);assert a['prediction'].tobytes()==aa['prediction'].tobytes();counts['prior4096_bytewise_predictors']+=1
    cfg=cfgs[0];zero=suite.cold.base.forward(q,np.zeros(4))
    for exc in [FloatingPointError,AssertionError,ValueError,RuntimeError]:
        def fail(*args):raise exc('preflight injected numerical failure')
        a,m=suite.guarded_fit(cfg,x,v,q,p['seed'],loaded,runner=fail);assert m['execution_failed'] and a['prediction'].tobytes()==zero.tobytes() and m['charged_complete_seconds']>0;counts['numeric_failure_fallbacks']+=1
    def nan(*args):return dict(prediction=np.full_like(q,np.nan)),{}
    a,m=suite.guarded_fit(cfg,x,v,q,p['seed'],loaded,runner=nan);assert m['execution_failed'] and a['prediction'].tobytes()==zero.tobytes();counts['nonfinite_fallbacks']+=1
    def bug(*args):raise KeyError('injected fatal coding error')
    try:suite.guarded_fit(cfg,x,v,q,p['seed'],loaded,runner=bug);raise AssertionError('Coding error must not be swallowed.')
    except KeyError:counts['coding_errors_propagated']+=1
    dump(out/'rows.json',rows);ans=dict(passed=True,counts=counts,max_independent_prior_prediction_gap=maxgap,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),query_targets_accessed=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()
