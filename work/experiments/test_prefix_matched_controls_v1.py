"""395 all-prefix direct/independent readout and actual worker isolation tests."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import time
import traceback
import numpy as np
import prefix_matched_controls_v1 as controls
import prefix_deadline_worker_v1 as worker
from deadline_risk_io_v1 import observations

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/prefix_matched_controls/preflight_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p,v):
    p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')


def same(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert np.array_equal(a,b),float(np.max(abs(a-b)))


def main():
    import torch
    import candidate_set_readout_v1 as reader
    torch.set_num_threads(1)
    start=time.perf_counter();folder=controls.require_training(ROOT)
    assert read(ROOT/'results/low_support_coverage/audit_v1/summary.json')['passed']
    files=[Path(__file__),Path(controls.__file__),Path(worker.__file__),
           ROOT/'outputs/ttt-pc-alm-research/395_prefix_matched_controls_v1.md']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    configs=controls.cohort_configs(ROOT)
    protocol=dict(source_sha256=hashes,cohort_configs=configs,portfolios=controls.PORTFOLIOS,
        training_audit_sha256=sha(folder.parent/'training_audit_v1/summary.json'),
        seeds=[5920000,5920001],stages=list(controls.STAGES),query_targets_accessed=False)
    save(OUT/'protocol.json',protocol)
    cases={s:observations(s) for s in [5920000,5920001]};q=np.linspace(0,1,257)
    checks=Counter();references={}
    for cfg in configs:
        model,_=controls.load_model(cfg,ROOT);fingerprint=worker.parent.fingerprint(dict(model=model))
        for n in controls.STAGES:
            x,v=cases[5920000];a,m=controls.fit_model(model,x[:n],v[:n],q)
            xx,vv,qq=[torch.from_numpy(z.copy())[None] for z in [x[:n],v[:n],q]]
            with torch.no_grad():
                state=None
                for prefix in [p for p in controls.STAGES if p<=n]:
                    state=model.adapt(xx[:,:prefix],vv[:,:prefix],state)
                raw=model.predict(state,qq)[0].numpy()
            same(raw,a['raw_prediction']);checks['matched_model_prefix_raw_predictions']+=1
            other_x,other_v=cases[5920001]
            controls.fit_model(model,other_x[:n],other_v[:n],q)
            again,_=controls.fit_model(model,x[:n],v[:n],q)
            same(a['prediction'],again['prediction']);checks['a_b_a_model_isolation']+=1
            reverse,_=controls.fit_model(model,x[:n],v[:n],q[::-1])
            assert np.max(abs(reverse['prediction'][::-1]-a['prediction']))<1e-10
            checks['negative_stride_query_readonly']+=1
            assert worker.parent.fingerprint(dict(model=model))==fingerprint
            references[cfg['name'],n]=a['prediction'].copy()
        print(json.dumps(dict(stage='model_checked',name=cfg['name'])),flush=True)
    for name in controls.PORTFOLIOS:
        for n in controls.STAGES:
            x,v=cases[5920000];events=[]
            aa,mm=controls.fit_portfolio(x[:n],v[:n],q,seed=5920000,name=name,
                emit=lambda kind,p:events.append((kind,p.copy())))
            assert len(events)==len(controls.PORTFOLIOS[name])
            for i,record in enumerate(mm['stages']):
                keys=record['cumulative_keys']
                regions=np.array([np.frombuffer(bytes.fromhex(k),np.uint8).reshape(4,n) for k in keys])
                ra,rm=reader.fit(x[:n],v[:n],q,regions,seed=5920000,search_completed=False)
                assert rm['readout_available']
                for field in ['prediction','points','allocation']:
                    same(ra[field],aa[f'stage_{i}_{field}']);checks['independent_union_readout_arrays']+=1
                assert record['positive_keys']==rm['positive_modes']
                same(events[i][1],ra['prediction'])
                assert events[i][0]=='fallback'
            references[name,n]=aa['prediction'].copy()
            checks['portfolio_prefixes']+=1
            other_x,other_v=cases[5920001]
            controls.fit_portfolio(other_x[:n],other_v[:n],q,seed=5920001,name=name)
            again,_=controls.fit_portfolio(x[:n],v[:n],q,seed=5920000,name=name)
            same(again['prediction'],aa['prediction']);checks['portfolio_a_b_a_isolation']+=1
        print(json.dumps(dict(stage='portfolio_checked',name=name)),flush=True)
    selected=[next(c for c in configs if c['name']==name) for name in
        ['official_ttt_native_prior256_32_p1','cohort_meta_ridge128','cohort_meta_shallow64_20']]
    selected += [dict(name=name,kind='portfolio') for name in controls.PORTFOLIOS]
    selected += [dict(name='prior4096_ridge',kind='regression',method='prior4096_ridge'),
                 dict(name='active1024_dfs',kind='regional',method='regional_active1024',schedule='dfs',ordering='farthest_x')]
    sessions=[]
    for cfg in selected:
        session=worker.Session(cfg,ROOT);results=[]
        try:
            for i,(seed,n,budget) in enumerate([(5920000,4,20.),(5920001,8,20.),(5920000,4,20.),(5920000,4,1e-6),(5920000,4,20.)]):
                x,v=cases[seed];pred,meta,events,archive=session.run(x[:n],v[:n],q,seed=seed,budget=budget)
                assert meta['error'] is None,meta
                if budget>1:
                    assert meta['selected']=='final' and archive is not None
                    if seed==5920000:
                        if (cfg['name'],n) in references:
                            same(pred,references[cfg['name'],n])
                        results.append(pred.copy())
                    if cfg['kind']=='portfolio':
                        assert sum(e['kind']=='fallback' for e in events)==1+len(controls.PORTFOLIOS[cfg['name']])
                else:
                    assert meta['selected']=='constant' and meta['terminated_owned_worker']
                    assert archive is None
                checks['actual_worker_calls']+=1
                # Keep each prediction, full resource records and event timestamps.
                case_folder=OUT/f"worker_{cfg['name']}_{i}";case_folder.mkdir()
                np.savez_compressed(case_folder/'prediction.npz',prediction=pred)
                save(case_folder/'metadata.json',meta)
                save(case_folder/'events.json',[{k:v for k,v in e.items() if k!='prediction'} for e in events])
            same(results[0],results[1]);same(results[0],results[2])
            checks['worker_repeated_and_recreated_predictions']+=2
        finally:
            lifecycle=session.close();sessions.append(dict(name=cfg['name'],**lifecycle))
        assert len(lifecycle['setups'])==2 and len(lifecycle['closures'])==2
        print(json.dumps(dict(stage='worker_checked',name=cfg['name'])),flush=True)
    # Exact boundary rule, independent of machine timing.
    ee=[dict(kind='fallback',received_seconds=.25,prediction=np.full_like(q,.2)),
        dict(kind='fallback',received_seconds=.5,prediction=np.full_like(q,.4)),
        dict(kind='final',received_seconds=.500001,prediction=np.full_like(q,.8))]
    for budget,value,kind in [(.1,.5,'constant'),(.25,.2,'fallback'),(.5,.4,'fallback'),(.6,.8,'final')]:
        prediction,selected=worker.parent.old.choose(ee,budget,q)
        same(prediction,np.full_like(q,value));assert selected==kind;checks['exact_receipt_boundaries']+=1
    for p,digest in hashes.items():
        assert sha(ROOT/p)==digest
    save(OUT/'sessions.json',sessions)
    save(OUT/'summary.json',dict(passed=True,checks=dict(checks),seconds=time.perf_counter()-start,
        query_targets_accessed=False,source_sha256=hashes,
        outputs_sha256={'protocol.json':sha(OUT/'protocol.json'),'sessions.json':sha(OUT/'sessions.json')}))
    print(json.dumps(read(OUT/'summary.json')),flush=True)


if __name__=='__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    controls.require_training(ROOT)
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        main()
    except Exception:
        save(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
