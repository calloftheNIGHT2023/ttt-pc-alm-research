"""285 four-context fixed25 preflight; no query targets or posterior moments."""
import argparse,json,os,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import probe_credit_resource_suite as suite
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from run_multiplier_fixed_point_screen import sha,dump
cold=suite.probe.cold


def local_reference(cfg,x,v,a,counts):
    starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(32,4))];prep=cold.Local(starts,x,v,'nodual')
    for _ in range(16):prep.step()
    atoms=[suite.probe.transfer.run(prep.b[r],prep.h[:,r],prep.best[r],x,v,None,cfg['atomic'])[0] for r in range(33)]
    b=np.array([z['b'] for z in atoms]);h=np.stack([z['h'] for z in atoms],axis=1);u=np.stack([z['u'] for z in atoms],axis=1);best=np.array([z['best'] for z in atoms])
    for key,value in [('b',b),('h',h),('u',u),('best',best)]:assert value.tobytes()==a['initial_'+key].tobytes();counts['local_initial_arrays']+=1
    state=cold.Local(b,x,v,cfg['solver']);state.h=h;state.u=u;state.best=best;cold.frozen.retain(state.best,b,x,v,np.zeros(4));state.errors,state.moves=cold.base.score(state.best,x,v,np.zeros(4))
    history={k:[val] for k,val in state.arrays().items()}
    prefix=cfg['prefix'];extra=cfg['extra']
    for _ in range(prefix+extra):
        state.step()
        for k,val in state.arrays().items():history[k].append(val)
    for key,values in history.items():
        values=np.array(values);assert values[:prefix+1].tobytes()==a['prefix_'+key].tobytes();counts['local_prefix_arrays']+=1
        expected=values[prefix:prefix+extra+1,:,:1] if key in ['h','u'] else values[prefix:prefix+extra+1,:1]
        assert expected.tobytes()==a['anchor_'+key].tobytes();counts['local_anchor_arrays']+=1
    counts['local_reference_batches']+=1


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/probe_credit_resources/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    parent=root/'results/round_284_audit.json';old=json.loads(parent.read_text());assert old['passed'];hashes=dict(old['source_sha256'])
    for n in ['probe_credit_budget_local.py','probe_credit_resource_suite.py',Path(__file__).name]:hashes[n]=sha(src/n)
    for n,h in hashes.items():assert sha(src/n)==h,n
    design=root/'outputs/ttt-pc-alm-research/285_probe_credit_resource_protocol.md';assert sha(design)==old['report_sha256'][design.name]
    start=time.perf_counter();loaded,checkpoints=suite.legacy.oldfit.meta.load(root);loading=time.perf_counter()-start;cfgs=suite.catalogue();index=suite.frozen_inputs(root)
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=sha(design),seeds=[5910000,5910001,5910053,5910063],configs=cfgs,checkpoint_manifest=checkpoints,model_loading_seconds=loading,phase_accesses_query_targets=False,
        scope='Trace-off/trace-on predictor identity, original fixed-prefix golden replay, variable horizons vs original Local')
    dump(out/'protocol.json',p);counts=Counter();rows=[];begin=time.perf_counter()
    with discovery_box(.12):
        for seed in p['seeds']:
            directory,row=index[seed,suite.PRIMARY];assert sha(directory/row['file'])==row['sha256']
            with np.load(directory/row['file']) as z:x=z['x_observed'].copy();v=z['v_observed'].copy();q=z['q_observed'].copy()
            for cfg in cfgs:
                repairs=[]
                def runner(c,x,v,q,seed,loaded):return suite.fit(c,x,v,q,seed,loaded,trace=False)
                with conditioned.geometry_scope(repairs):a,m=suite.legacy.guarded_fit(cfg,x,v,q,seed,loaded,runner=runner)
                if not m['execution_failed']:
                    counts['frozen_gold_arrays']+=suite.check_frozen(root,seed,cfg,a,m,index)
                    with conditioned.geometry_scope([]):traced,tm=suite.fit(cfg,x,v,q,seed,loaded,trace=True)
                    for k,value in a.items():assert value.tobytes()==traced[k].tobytes(),(seed,cfg['name'],k);counts['trace_off_on_arrays']+=1
                    if cfg['family']=='local':
                        local_reference(cfg['config'],x,v,traced,counts)
                        expected=33*cfg['config']['prefix']+cfg['config']['extra'];assert m['total_restart_sweeps']==expected and m['scans']['scanned_states']==0
                    if cfg['family'] in ['local','probe']:assert m['positive_modes']==tm['positive_modes']
                else:counts['retained_numerical_failures']+=1
                fn=f'{seed}_{cfg["name"]}.npz';np.savez_compressed(out/fn,x_observed=x,v_observed=v,q_observed=q,**a);rows.append(dict(seed=seed,method=cfg['name'],file=fn,sha256=sha(out/fn),metadata=m,repairs=repairs));counts['predictors']+=1
            print(json.dumps(dict(seed=seed,predictors=len(rows),total=100,seconds=time.perf_counter()-begin)),flush=True)
    dump(out/'rows.json',rows);ans=dict(passed=True,counts=counts,seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),phase_accesses_query_targets=False,next='600 randomized full untraced timings and50 separate memory calls')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
