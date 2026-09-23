"""Frozen39-method own-state experiment; query truth evaluated only after save."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import band_conditioned_online as model
import band_conditioned_execution as execution
import band_conditioned_meta as meta_model

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dump(p,obj):p.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def trial(xx,vv,q,cfg,seed,rep,p,loaded,out=None):
    state=None;rows=[];artifacts=[];failure=None;ever_recovered=False
    for n in p['stages']:
        rng=np.random.default_rng(np.random.SeedSequence([p['rng_seed'],seed,rep,n,p['posterior_samples']]))
        start=time.perf_counter()
        try:
            if cfg['family']=='meta':pred,new,meta=meta_model.fit(loaded[cfg['name']],xx[:n],vv[:n],state)
            elif cfg['family']=='regression':pred,new,meta=model.original.previous.regression(xx[:n],vv[:n],cfg['name'],state)
            else:pred,new,meta=execution.fit(xx[:n],vv[:n],state,cfg,rng,p['posterior_samples'])
        except Exception as exc:
            failure=dict(seed=seed,method=cfg['name'],repetition=rep,failed_n=n,failed_seconds=time.perf_counter()-start,
                traceback=traceback.format_exc(),recovery_attempts=getattr(exc,'events',None),conditioning_attempts=getattr(exc,'conditioning_attempts',None));break
        write=time.perf_counter()-start;start=time.perf_counter();prediction=pred(q);read_seconds=time.perf_counter()-start
        assert np.isfinite(prediction).all();arrays=dict(x=xx[:n],v=vv[:n],q=q,prediction=prediction)
        support_error=float(np.max(abs(pred(xx[:n])-vv[:n])))
        if cfg['family']=='h2':
            assert model.original.memory.prior.state_digest(state)==meta['previous_state_digest'];arrays.update(points=new.samples,anchor=new.anchor)
            _,h=model.original.memory.prior.capture.model.light.forward_many(xx[:n],new.samples)
            particle_error=float(np.max(abs(h[:,-1]-vv[:n])));assert particle_error<=.001+1e-8
            ever_recovered|=meta['recovery']['triggered'];assert sum(e['seconds'] for e in meta['recovery']['attempts'])<=write+1e-6
            attempts=meta['conditioning_attempts']
            if n==4 and cfg.get('band_pool'):assert len(attempts)==1
            if n>4:assert not attempts
        elif cfg['family']=='meta':arrays.update(meta_model.arrays(new));particle_error=None
        else:
            particle_error=None
            if new is not None:arrays.update(rls_w=new['w'],rls_p=new['p'])
        row=dict(seed=seed,method=cfg['name'],family=cfg['family'],learner=cfg.get('learner'),band_pool=cfg.get('band_pool'),repetition=rep,n=n,
            write_seconds=write,read_seconds=read_seconds,total_seconds=write+read_seconds,state_bytes=meta['persistent_state_bytes'],shared_model_bytes=meta.get('shared_model_bytes',0),
            state_digest=meta.get('new_state_digest'),previous_state_digest=meta.get('previous_state_digest'),support_max_error=support_error,
            support_feasible=bool(support_error<=.001+1e-6),particle_support_max_error=particle_error,recovery=meta.get('recovery'),ever_recovered=ever_recovered,
            geometry_calls=meta.get('geometry_calls'),positive_modes=len(meta.get('positive_mode_keys',[])) if cfg['family']=='h2' else None,
            selected_length=meta.get('selected_length'),selected_ridge=meta.get('selected_ridge'),support_press=meta.get('support_press'))
        if cfg['family']=='h2' and meta['conditioning_attempts']:
            bm=meta['conditioning_attempts'][0];arrays['initial_starts']=np.array(bm['selected_starts'])
            row['initialization']={k:v for k,v in bm.items() if k not in ['selected_starts','inverse_records']}
        if out is not None:
            file=out/f'state_{seed}_{cfg["name"]}_{rep}_{n}.npz';assert not file.exists();np.savez_compressed(file,**arrays);row.update(state_file=file.name,state_sha256=sha(file))
            if n==4 and cfg['family']=='h2':
                detail=out/f'detail_{seed}_{cfg["name"]}_{rep}.json';dump(detail,meta);row.update(detail_file=detail.name,detail_sha256=sha(detail))
        rows.append(row);artifacts.append(arrays);state=new
    return rows,artifacts,failure

def verify_execution():
    x,v=model.original.previous.olddriver.observations(5900001);cfg=next(c for c in model.configs() if c['name']==model.PRIMARY)
    ref,rs,_=model.fit(x[:4],v[:4],None,cfg,np.random.default_rng(484351),64)
    actual,state,meta=execution.fit(x[:4],v[:4],None,cfg,np.random.default_rng(484351),64)
    assert rs.samples.tobytes()==state.samples.tobytes() and ref(x).tobytes()==actual(x).tobytes();assert len(meta['conditioning_attempts'])==1
    # Fail only after original initialization/discovery; retain its evidence
    # while the identical common prior recovery runs without band conditioning.
    recovery=model.original.recovery;saved=recovery.memory.fit
    def force_original(x,v,state,cfg,rng,count):
        answer=saved(x,v,state,cfg,rng,count)
        if not cfg.get('recovery_role'):raise RuntimeError(recovery.EMPTY)
        return answer
    try:
        recovery.memory.fit=force_original
        _,_,mm=execution.fit(x[:4],v[:4],None,cfg,np.random.default_rng(484351),64)
    finally:recovery.memory.fit=saved
    assert mm['recovery']['triggered'] and len(mm['conditioning_attempts'])==1
    assert [a['role'] for a in mm['recovery']['attempts']]==['original','recovery']
    return dict(passed=True,observer_states_bytewise=True,failed_original_initialization_preserved=True,common_recovery_unchanged=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    primpath=root/'results/band_conditioned_online/primitive_v2/summary.json';prim=read(primpath);assert prim['passed'];hashes=dict(prim['source_sha256'])
    for name,h in hashes.items():assert sha(src/name)==h,name
    hashes.update({name:sha(src/name) for name in ['band_conditioned_execution.py',Path(__file__).name]})
    design=root/'outputs/ttt-pc-alm-research/243_band_conditioned_online_protocol.md';assert sha(design)==prim['design_sha256']
    loaded,manifest=meta_model.load(root);assert manifest==prim['checkpoint_manifest'];execution_check=verify_execution()
    # Common one-stage warmup for all39 configurations on the old preflight task,
    # with no query targets. Its wall time is disclosed separately, not a new fit.
    configs=model.configs();x,v=model.original.previous.olddriver.observations(5900001);start=time.perf_counter()
    warm=dict(stages=[4],rng_seed=481853,posterior_samples=64)
    for c in configs:
        rr,_,fail=trial(x,v,np.linspace(0,1,61),c,5900001,0,warm,loaded)
        assert fail is None and len(rr)==1
    warm_seconds=time.perf_counter()-start
    p=dict(source_sha256=hashes,primitive_sha256=sha(primpath),design_sha256=sha(design),checkpoint_manifest=manifest,execution_check=execution_check,
        configs=configs,seeds=list(range(5920000,5920016)),repetitions=2,stages=[4,8,16,24],posterior_samples=2048,rng_seed=481853,order_seed=484331,
        query_points=257,primary=model.PRIMARY,bootstrap_repeats=20000,bootstrap_seed=484333,threads=dict(blas=1,torch=torch.get_num_threads(),torch_interop=torch.get_num_interop_threads()),
        common_old_seed_warmup_seconds=warm_seconds,expected_episodes=1248,expected_stages=4992,
        scope='39 controls, old16 development tasks, shared starts and new full timings; not independent confirmation or official TTT',
        query_policy='Only observed x,v passed to adaptation; truth evaluated after every method and repetition for a task has saved state and predictions',
        clipping_policy='Keep frozen regression predictions; meta controls keep their original [0,1] clipping; posterior outputs naturally bounded')
    out=root/'results/band_conditioned_online/development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();dump(out/'protocol.json',p)
    rows=[];episodes=[];failures=[];q=np.linspace(0,1,p['query_points']);order=np.random.default_rng(p['order_seed']);done=0
    jobs=[(c,r) for c in configs for r in range(2)]
    for seed in p['seeds']:
        xx,vv=model.original.previous.olddriver.observations(seed);seedrows=[]
        for index in order.permutation(len(jobs)):
            cfg,rep=jobs[index];rr,_,failure=trial(xx,vv,q,cfg,seed,rep,p,loaded,out)
            if failure:failures.append(failure)
            ep=dict(seed=seed,method=cfg['name'],repetition=rep,complete=failure is None,stages=len(rr),failed_n=None if failure is None else failure['failed_n'],
                charged_seconds=sum(r['total_seconds'] for r in rr)+(0 if failure is None else failure['failed_seconds']),
                recovery_stages=sum(bool(r['recovery'] and r['recovery']['triggered']) for r in rr))
            episodes.append(ep);seedrows.extend(rr);done+=1
            print(json.dumps(dict(completed=done,total=1248,seed=seed,method=cfg['name'],rep=rep,complete=ep['complete'],recoveries=ep['recovery_stages'],seconds=ep['charged_seconds'])),flush=True)
        # This is the first access to query truth for this task. All1248 stream
        # predictions are independent of it; it never enters an adaptation API.
        truth=model.base.forward(q,np.random.default_rng(seed).uniform(-.12,.12,4))
        for row in seedrows:
            with np.load(out/row['state_file']) as z:row['query_mse']=float(np.trapezoid((z['prediction']-truth)**2,x=q))
        rows.extend(seedrows)
        for name,value in [('rows.json',rows),('episodes.json',episodes),('failures.json',failures)]:dump(out/name,value)
    for name,h in hashes.items():assert sha(src/name)==h,name
    result=dict(execution_complete=True,episodes=done,stages=len(rows),failures=len(failures),recovery_events=sum(e['recovery_stages'] for e in episodes),sources=len(hashes),
        source_sha256=sha(Path(__file__)),input_sha256={name:sha(out/name) for name in ['protocol.json','rows.json','episodes.json','failures.json']})
    dump(out/'run_audit.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
