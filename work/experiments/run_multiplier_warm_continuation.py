"""Frozen six-way continuation from certified points. Evaluate truth LAST."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import multiplier_warm_continuation as warm
import multiplier_fixed_point_exact as exact
from run_multiplier_fixed_point_screen import sha,dump

CONFIGS=[('alm64','alm',64),('nodual64','nodual',64),('pc80','pc',80),('adam60','adam',60),('adam240','adam',240),('gn20','gauss_newton',20)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    parent=root/'results/multiplier_fixed_point';inp=parent/'exact';out=parent/'continuation';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    old=json.loads((inp/'protocol.json').read_text());hashes=dict(old['source_sha256'])
    for n,h in hashes.items():assert sha(src/n)==h,n
    hashes.update({n:sha(src/n) for n in ['multiplier_warm_continuation.py',Path(__file__).name]})
    p=dict(source_sha256=hashes,exact_summary_sha256=sha(inp/'summary.json'),accepted_sha256=sha(inp/'accepted.json'),certificates_sha256=sha(inp/'certificates.json'),
        configs=CONFIGS,verification=warm.verify(),warm_state='nearest binary floats to certified rational state; zero multipliers; preserve old512 support-best',
        query_points=257,query_truth_after='all tasks and methods traces and common branch LP cache committed',preparation_cost='shared nodual512 is charged diagnostic setup; no online timing claim')
    dump(out/'protocol.json',p);certs=json.loads((inp/'certificates.json').read_text());rows=[];geometries=[];start=time.perf_counter()
    with warm.original.old.core.pipeline.discovery_box(.12):
        for seed in range(5900000,5900016):
            selected=[c for c in certs if c['seed']==seed and c['proof']['accepted']];a=np.load(parent/'screen'/f'{seed}.npz');x=a['x'];v=a['v']
            states=[exact.decode(c['state']) for c in selected];starts=np.array([[float(t) for t in b] for b,h in states])
            activities=np.array([[[float(t) for t in row] for row in h] for b,h in states]).transpose(1,0,2)
            ids=[c['restart'] for c in selected];incumbent=a['best'][ids];cache={}
            for name,method,steps in CONFIGS:
                begin=time.perf_counter();meta=None
                if method in ['alm','nodual','pc']:
                    saved=warm.base.forward_jacobian
                    def forbidden(*args,**kwargs):raise AssertionError('global BP in local continuation')
                    try:
                        warm.base.forward_jacobian=forbidden
                        arrays=warm.local(starts,activities,incumbent,x,v,method,steps)
                    finally:warm.base.forward_jacobian=saved
                else:arrays,meta=warm.baseline(starts,incumbent,x,v,method,steps)
                seconds=time.perf_counter()-begin
                arrays.update(x=x,v=v,restarts=np.array(ids),incumbent=incumbent)
                file=out/f'{seed}_{name}.npz';np.savez_compressed(file,**arrays)
                errs,_=warm.base.score(arrays['best'][-1],x,v,np.zeros(4));initerrs,_=warm.base.score(incumbent,x,v,np.zeros(4))
                for i,c in enumerate(selected):
                    initpattern=warm.base.pattern(x,starts[i]);keys=[];changed=[]
                    # The old incumbent is available to EVERY method, including geometry.
                    bank=np.r_[incumbent[i:i+1],arrays['b'][:,i]]
                    for t,b in enumerate(bank):
                        pattern=warm.base.pattern(x,b);key=pattern.astype(np.uint8).tobytes().hex()
                        if key not in cache:cache[key]=warm.base.branch_feasibility(x,v,b)
                        keys.append(key)
                        if t and np.any(pattern!=initpattern):changed.append(t-1)
                    rows.append(dict(seed=seed,restart=c['restart'],method=name,file=file.name,file_sha256=sha(file),
                        batch_seconds_including_trace_and_BP_replay=seconds,initial_best_error=float(initerrs[i]),best_error=float(errs[i]),
                        strict_band_feasible=bool(errs[i]<=.001+1e-10),first_parameter_branch_change=changed[0] if changed else None,
                        parameter_evaluation_count=len(arrays['b']),distinct_modes=len(set(keys)),mode_keys=sorted(set(keys)),
                        feasible_modes=sorted(k for k in set(keys) if cache[k]['current_branch_feasible']),baseline_metadata=meta))
                print(json.dumps(dict(seed=seed,method=name,points=len(ids),feasible=int(np.sum(errs<=.001+1e-10)))),flush=True)
            geom=dict(seed=seed,modes=cache);geometries.append(geom);dump(out/f'{seed}_geometry.json',geom)
            dump(out/'support_rows.json',rows)
    manifest=dict(protocol_sha256=sha(out/'protocol.json'),support_rows_sha256=sha(out/'support_rows.json'),
        arrays={f.name:sha(f) for f in out.glob('*.npz')},geometry={f.name:sha(f) for f in out.glob('*_geometry.json')})
    dump(out/'before_query_manifest.json',manifest)
    # Only this evaluation block reconstructs a latent teacher / query answers.
    q=np.linspace(0,1,257);query=[]
    for row in rows:
        a=np.load(out/row['file']);i=list(a['restarts']).index(row['restart']);latent=np.random.default_rng(row['seed']).uniform(-.12,.12,4)
        truth=warm.base.forward(q,latent);pred=warm.base.forward(q,a['best'][-1,i]);initial=warm.base.forward(q,a['incumbent'][i])
        query.append(dict(seed=row['seed'],restart=row['restart'],method=row['method'],mse=float(np.mean((pred-truth)**2)),initial_mse=float(np.mean((initial-truth)**2))))
    dump(out/'query_rows.json',query)
    summary=dict(passed=True,certified_points=69,method_points=len(rows),tasks=16,seconds_including_checks=time.perf_counter()-start,
        before_query_manifest_sha256=sha(out/'before_query_manifest.json'),query_rows_sha256=sha(out/'query_rows.json'),
        methods={name:dict(feasible=sum(r['strict_band_feasible'] for r in rows if r['method']==name),branch_changed=sum(r['first_parameter_branch_change'] is not None for r in rows if r['method']==name),
            support_improved=sum(r['best_error']<r['initial_best_error']-1e-10 for r in rows if r['method']==name),query_mean=float(np.mean([r['mse'] for r in query if r['method']==name]))) for name,_,_ in CONFIGS})
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
