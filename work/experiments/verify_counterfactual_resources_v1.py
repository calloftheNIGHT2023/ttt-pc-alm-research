"""293 full37 OLD4 functional preflight; strict checks outside failure guard."""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
import counterfactual_resource_suite_v1 as suite
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from verify_probe_credit_resources import local_reference
from diagnose_gradient_flat_split_states_v1 import read,save,sha


def run(root,out):
    hashes=suite.gate(root)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfgs=suite.catalogue(root)
    index=suite.frozen_inputs(root)
    inputs=suite.observed_inputs(root,index,suite.PREFLIGHT_SEEDS)
    begin=time.perf_counter()
    loaded,checkpoints=suite.resources.legacy.oldfit.meta.load(root)
    loading=time.perf_counter()-begin
    old=read(root/'results/probe_credit_confirmation/head_preflight/protocol.json')
    assert checkpoints==old['checkpoint_manifest']
    save(out/'protocol.json',dict(source_sha256=hashes,configs=cfgs,seeds=suite.PREFLIGHT_SEEDS,
        checkpoint_manifest=checkpoints,model_loading_seconds=loading,query_targets_accessed=False,
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/293_counterfactual_resource_protocol.md'),
        scope='All37 projected implementations; trace on/off; new horizon golden paths; no quality'))
    counts=Counter();rows=[];files={};start=time.perf_counter()
    with discovery_box(.12):
        for seed in suite.PREFLIGHT_SEEDS:
            x,v,q=inputs[seed]
            for cfg in cfgs:
                repairs=[]
                # No failure guard here: implementation identities cannot be hidden by fallback.
                with conditioned.geometry_scope(repairs):
                    a,m=suite.fit(cfg,x,v,q,seed,loaded,trace=False)
                for field in suite.FIELDS:
                    assert a[field].shape==(257,) and np.isfinite(a[field]).all()
                    assert np.all((a[field]>=0)&(a[field]<=1))
                for value in a.values():
                    if np.issubdtype(value.dtype,np.number):assert np.isfinite(value).all()
                counts['frozen_arrays']+=suite.check_frozen(seed,cfg,a,m,index)
                if cfg['name'] not in suite.NEW_LOCAL:
                    assert (seed,cfg['name']) in index,(seed,cfg['name'])
                with conditioned.geometry_scope([]):
                    traced,tm=suite.fit(cfg,x,v,q,seed,loaded,trace=True)
                for key,value in a.items():
                    assert value.shape==traced[key].shape and value.dtype==traced[key].dtype
                    assert value.tobytes()==traced[key].tobytes(),(seed,cfg['name'],key)
                    counts['trace_off_on_arrays']+=1
                if 'positive_modes' in m:assert m['positive_modes']==tm['positive_modes']
                if cfg['name'] in suite.NEW_LOCAL:
                    local_reference(cfg['config'],x,v,traced,counts)
                    assert m['total_restart_sweeps']==34*cfg['config']['prefix']
                if cfg['name']==suite.PRIMARY:
                    counts['candidate_proposals']+=m['counterfactual_state_steps']
                    assert m['global_bp_used'] is False
                fn=f'{seed}_{cfg["name"]}.npz'
                with (out/fn).open('xb') as stream:np.savez_compressed(stream,**a)
                files[fn]=sha(out/fn)
                row=dict(seed=seed,method=cfg['name'],file=fn,sha256=files[fn],metadata=m,repairs=repairs)
                rows.append(row)
                save(out/(fn+'.record.json'),row)
                counts['predictors']+=1
            print(dict(tasks=suite.PREFLIGHT_SEEDS.index(seed)+1,predictors=len(rows),total=148,seconds=time.perf_counter()-start),flush=True)
    assert len(rows)==148 and counts['local_reference_batches']==12
    save(out/'rows.json',rows);save(out/'files.json',files)
    assert suite.gate(root)==hashes
    result=dict(passed=True,tasks=4,methods=37,predictors=148,counts=dict(counts),query_targets_accessed=False,
        seconds=time.perf_counter()-start,core_research_goal_complete=False,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','rows.json','files.json']},
        next='888 randomized complete untraced calls and74 separate memory probes')
    save(out/'summary.json',result);print(result,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    root=p.parse_args().project.resolve();out=root/'results/counterfactual_resources/preflight_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise


if __name__=='__main__':main()
