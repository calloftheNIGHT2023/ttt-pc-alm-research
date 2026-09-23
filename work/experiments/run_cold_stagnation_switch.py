"""Round247 fixed20-way cold-start test; every prediction committed first."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import cold_stagnation_switch as cold
import band_conditioned_meta as meta
import run_recovered_online_comparison as regression
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha,dump

def fit(cfg,x,v,q,loaded,trace=True):
    name=cfg['name'];family=cfg['family'];begin=time.perf_counter();arrays={}
    if family in ['local','bp']:
        starts=cold.starts()
        if family=='local':best,history,mm=cold.run_local(starts,x,v,cfg['method'],cfg['steps'],trace)
        else:best,history,mm=cold.run_bp(starts,x,v,cfg['method'],cfg['steps'],trace)
        selected,b=cold.select(best,x,v)
        def predict(q):return cold.base.forward(q,b)
        arrays.update(best_bank=best,selected_b=b,selected_restart=np.array(selected),starts=starts)
        if history is not None:arrays.update(history)
        mm.update(selected_restart=selected,persistent_predictor_bytes=b.nbytes,shared_model_bytes=0)
    elif family=='meta':
        predict,state,mm=meta.fit(loaded[name],x,v,None);arrays.update(meta.arrays(state))
    else:
        predict,state,mm=regression.regression(x,v,name,None)
        if state is not None:arrays.update(rls_w=state['w'],rls_p=state['p'])
    write=time.perf_counter()-begin;begin=time.perf_counter();prediction=predict(q);read=time.perf_counter()-begin
    support=predict(x);assert np.isfinite(prediction).all() and np.isfinite(support).all()
    arrays.update(x=x,v=v,q=q,prediction=prediction,support_prediction=support)
    row=dict(method=name,family=family,write_seconds=write,read_seconds=read,total_seconds=write+read,support_max_error=float(np.max(abs(support-v))),
        support_feasible=bool(np.max(abs(support-v))<=.001001),metadata=mm,trace_numeric_bytes=sum(a.nbytes for k,a in arrays.items() if k in ['b','h','u','best','active','first_trigger','roles']))
    return arrays,row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    base=root/'results/cold_stagnation_switch';out=base/'development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    primitive=json.loads((base/'primitive/summary.json').read_text());assert primitive['passed'];hashes=dict(primitive['source_sha256'])
    for name,h in hashes.items():assert sha(src/name)==h,name
    hashes[Path(__file__).name]=sha(Path(__file__));loaded,manifest=meta.load(root)
    assert primitive['checkpoint_manifest']=={n:manifest[n] for n in primitive['checkpoint_manifest']}
    configs=cold.configs();q=np.linspace(0,1,257);x,v=observations(5900001);warmbegin=time.perf_counter()
    # Complete20-method warmup on the old preflight task, with no query truth.
    with cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for cfg in configs:fit(cfg,x[:4],v[:4],q,loaded,False)
    p=dict(source_sha256=hashes,primitive_sha256=sha(base/'primitive/summary.json'),design_sha256=primitive['design_sha256'],checkpoint_manifest=primitive['checkpoint_manifest'],
        configs=configs,seeds=list(range(5900000,5900016)),repetitions=1,query_points=257,order_seed=247903,primary=cold.PRIMARY,
        common_starts='zero plus seed731 prior16',n=4,total_switch_budget=128,trigger='last two primal changes <=1e-12; max residual >1e-6; retained error >.001001; activate next sweep from zero',
        warmup_seconds=time.perf_counter()-warmbegin,threads=dict(blas=1,torch=torch.get_num_threads()),scope='Old unfiltered development tasks; point prediction; geometry diagnostic only')
    dump(out/'protocol.json',p);rows=[];rng=np.random.default_rng(p['order_seed']);begin=time.perf_counter()
    with cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for seed in p['seeds']:
            x,v=observations(seed);x=x[:4];v=v[:4];cache={}
            for index in rng.permutation(len(configs)):
                cfg=configs[index];arrays,row=fit(cfg,x,v,q,loaded,True);row['seed']=seed
                file=out/f'{seed}_{cfg["name"]}.npz';np.savez_compressed(file,**arrays);row.update(file=file.name,file_sha256=sha(file))
                if cfg['family'] in ['local','bp']:
                    gb=time.perf_counter();keys=set()
                    for b in arrays['b'].reshape(-1,4):
                        key=cold.base.pattern(x,b).astype(np.uint8).tobytes().hex();keys.add(key)
                        if key not in cache:cache[key]=cold.base.branch_feasibility(x,v,b)
                    row.update(mode_keys=sorted(keys),feasible_modes=sorted(k for k in keys if cache[k]['current_branch_feasible']),geometry_diagnostic_seconds=time.perf_counter()-gb)
                rows.append(row);print(json.dumps(dict(done=len(rows),total=320,seed=seed,method=cfg['name'],support_error=row['support_max_error'],triggers=row['metadata'].get('active_restarts'))),flush=True)
            dump(out/f'{seed}_geometry.json',cache);dump(out/'support_rows.json',rows)
    manifest=dict(protocol_sha256=sha(out/'protocol.json'),support_rows_sha256=sha(out/'support_rows.json'),
        arrays={f.name:sha(f) for f in out.glob('*.npz')},geometry={f.name:sha(f) for f in out.glob('*_geometry.json')})
    dump(out/'before_query_manifest.json',manifest)
    # The first query-answer access is after all320 predictions are committed.
    query=[]
    for row in rows:
        a=np.load(out/row['file']);teacher=np.random.default_rng(row['seed']).uniform(-.12,.12,4);truth=cold.base.forward(a['q'],teacher)
        query.append(dict(seed=row['seed'],method=row['method'],mse=float(np.mean((a['prediction']-truth)**2))))
    dump(out/'query_rows.json',query)
    for n,h in hashes.items():assert sha(src/n)==h,n
    summary=dict(passed=True,complete=True,tasks=16,configs=20,predictions=320,seconds_including_geometry_and_io=time.perf_counter()-begin,
        before_query_manifest_sha256=sha(out/'before_query_manifest.json'),query_rows_sha256=sha(out/'query_rows.json'),
        methods={cfg['name']:dict(mse=float(np.mean([r['mse'] for r in query if r['method']==cfg['name']])),
            support_feasible=sum(r['support_feasible'] for r in rows if r['method']==cfg['name'])) for cfg in configs})
    dump(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
