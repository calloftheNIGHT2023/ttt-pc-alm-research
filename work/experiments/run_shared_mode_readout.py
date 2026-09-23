"""Frozen shared posterior readout of all247 pools; marginal one-mode diagnostic."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import cold_stagnation_switch as cold
import shared_mode_readout as shared
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    previous=root/'results/cold_stagnation_switch';inp=previous/'development';out=root/'results/shared_mode_readout/development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    old=json.loads((inp/'protocol.json').read_text());hashes=dict(old['source_sha256']);analysis=json.loads((previous/'analysis/summary.json').read_text());assert analysis['passed']
    for n,h in hashes.items():assert sha(src/n)==h,n
    extra=['audit_cold_stagnation_switch.py','benchmark_cold_stagnation_switch.py','analyze_cold_stagnation_switch.py','shared_mode_readout.py',Path(__file__).name]
    hashes.update({n:sha(src/n) for n in extra});methods=[c['name'] for c in old['configs'] if c['family'] in ['local','bp']];pools=methods+['other_union','all_union']
    p=dict(source_sha256=hashes,parent_protocol_sha256=sha(inp/'protocol.json'),parent_analysis_sha256=sha(previous/'analysis/summary.json'),
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/249_shared_mode_readout_protocol.md'),seeds=old['seeds'],methods=methods,pools=pools,
        sample_count=2048,repetition_seeds=list(range(249911,249916)),query_points=257,marginal_seed=5900007,marginal_per_mode_count=8192,
        primary=cold.PRIMARY,scope='Old frozen discovery pools; same geometric readout; no new discovery or official TTT claim')
    dump(out/'protocol.json',p);rows=[];geoms=[];marginal=[];q=np.linspace(0,1,257);oldrows={(r['seed'],r['method']):r for r in json.loads((inp/'support_rows.json').read_text())};start=time.perf_counter()
    with cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for seed in p['seeds']:
            keys={m:oldrows[seed,m]['feasible_modes'] for m in methods};union=set().union(*(set(k) for k in keys.values()));representatives={}
            for m in methods:
                row=oldrows[seed,m];path=inp/row['file'];assert sha(path)==row['file_sha256'];a=np.load(path);x=a['x'];v=a['v']
                for b in a['b'].reshape(-1,4):
                    key=cold.base.pattern(x,b).astype(np.uint8).tobytes().hex()
                    if key in union and key not in representatives:representatives[key]=b.copy()
            assert set(representatives)==union;polys={};notes={}
            for index,key in enumerate(sorted(union)):
                poly,note=shared.build(x,v,representatives[key]);note.update(key=key,representative=representatives[key].tolist())
                if poly is not None:
                    path=out/f'poly_{seed}_{index}.npz';np.savez_compressed(path,**poly);note.update(file=path.name,sha256=sha(path));polys[key]=poly
                notes[key]=note
            geoms.append(dict(seed=seed,regions=notes));dump(out/f'{seed}_geometry.json',geoms[-1])
            valid={m:sorted(k for k in keys[m] if k in polys) for m in methods}
            valid['other_union']=sorted(set().union(*(set(value) for m,value in valid.items() if m!=cold.PRIMARY)))
            valid['all_union']=sorted(polys);assert valid['other_union'] and valid['all_union']
            for m in pools:
                for rep,rs in enumerate(p['repetition_seeds']):
                    begin=time.perf_counter();pool=valid[m];allocation=[]
                    if pool:
                        points,allocation=shared.draw(polys,pool,p['sample_count'],np.random.default_rng(np.random.SeedSequence([rs,seed,2048])))
                    else:
                        assert m in methods;original=np.load(inp/f'{seed}_{m}.npz');points=original['selected_b'][None].copy()
                    sampling=time.perf_counter()-begin;begin=time.perf_counter();prediction=shared.geometry.make_predict(points)(q);reading=time.perf_counter()-begin
                    maximum=float(np.max(abs(np.stack([cold.base.forward(x,b) for b in points])-v)))
                    if pool:assert maximum<=.001+1e-7
                    file=out/f'state_{seed}_{m}_{rep}.npz';np.savez_compressed(file,x=x,v=v,q=q,points=points,prediction=prediction,allocation=np.array(allocation))
                    rows.append(dict(seed=seed,method=m,repetition=rep,mode_keys=pool,file=file.name,sha256=sha(file),fallback=not bool(pool),
                        volume=sum(polys[k]['volume'] for k in pool),max_particle_support_error=maximum,sampling_seconds=sampling,read_seconds=reading,
                        geometry_seconds=sum(notes[k]['seconds'] for k in keys[m]) if m in methods else sum(notes[k]['seconds'] for k in (set().union(*(set(keys[n]) for n in methods if n!=cold.PRIMARY)) if m=='other_union' else union)),
                        persistent_predictor_numeric_bytes=points.nbytes,scope='capacity diagnostic pool' if m.endswith('union') else 'same method shared readout'))
            if seed==p['marginal_seed']:
                for rep,rs in enumerate(p['repetition_seeds']):
                    values=[];pointfiles=[];begin=time.perf_counter()
                    for i,key in enumerate(valid['all_union']):
                        points=shared.geometry.sample(polys[key],p['marginal_per_mode_count'],np.random.default_rng(np.random.SeedSequence([rs,seed,i,8192])))
                        prediction=shared.geometry.make_predict(points)(q);file=out/f'marginal_points_{seed}_{rep}_{i}.npz';np.savez_compressed(file,points=points,prediction=prediction)
                        values.append(prediction);pointfiles.append(dict(key=key,file=file.name,sha256=sha(file)))
                    weights=np.array([polys[k]['volume'] for k in valid['all_union']]);oldmask=np.array([k in valid['other_union'] for k in valid['all_union']]);newmask=~oldmask
                    vv=np.array(values);prior=np.average(vv[oldmask],weights=weights[oldmask],axis=0);full=np.average(vv,weights=weights,axis=0);new=np.average(vv[newmask],weights=weights[newmask],axis=0)
                    w=float(weights[newmask].sum()/weights.sum());assert np.max(abs(full-((1-w)*prior+w*new)))<1e-14
                    file=out/f'marginal_{seed}_{rep}.npz';np.savez_compressed(file,q=q,prior=prior,full=full,new=new,w=np.array(w),mode_prediction=vv,weights=weights)
                    marginal.append(dict(seed=seed,repetition=rep,new_keys=[k for k,m in zip(valid['all_union'],newmask) if m],new_volume_fraction=w,file=file.name,sha256=sha(file),pointfiles=pointfiles,seconds=time.perf_counter()-begin))
            dump(out/'rows.json',rows);dump(out/'marginal_rows.json',marginal);print(json.dumps(dict(seed=seed,positive_modes=len(polys),candidate_modes=len(valid[cold.PRIMARY]),predictions=len(rows))),flush=True)
    before=dict(protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),marginal_rows_sha256=sha(out/'marginal_rows.json'),
        arrays={f.name:sha(f) for f in out.glob('*.npz')},geometry={f.name:sha(f) for f in out.glob('*_geometry.json')})
    dump(out/'before_query_manifest.json',before);evaluation=[];marginal_eval=[]
    # First evaluation of query answers: all960 predictions and all marginal
    # per-mode predictors already committed. No efficacy-driven reruns.
    for row in rows:
        a=np.load(out/row['file']);teacher=np.random.default_rng(row['seed']).uniform(-.12,.12,4);truth=cold.base.forward(q,teacher)
        evaluation.append(dict(seed=row['seed'],method=row['method'],repetition=row['repetition'],mse=float(np.mean((a['prediction']-truth)**2))))
    for row in marginal:
        a=np.load(out/row['file']);teacher=np.random.default_rng(row['seed']).uniform(-.12,.12,4);truth=cold.base.forward(q,teacher);w=float(a['w']);delta=a['new']-a['prior']
        cross=float(2*w*np.mean((a['prior']-truth)*delta));square=float(w*w*np.mean(delta**2));prior=float(np.mean((a['prior']-truth)**2));full=float(np.mean((a['full']-truth)**2))
        assert abs(full-prior-cross-square)<1e-14
        marginal_eval.append(dict(seed=row['seed'],repetition=row['repetition'],w=w,prior_mse=prior,full_mse=full,difference=full-prior,cross_term=cross,positive_square_term=square))
    dump(out/'query_rows.json',evaluation);dump(out/'marginal_evaluation.json',marginal_eval)
    for n,h in hashes.items():assert sha(src/n)==h,n
    result=dict(passed=True,complete=True,predictions=len(rows),positive_regions=sum(r['regions'][k]['proof']['accepted'] for r in geoms for k in r['regions'] if r['regions'][k]['proof']),
        geometric_regions=sum(len(r['regions']) for r in geoms),seconds=time.perf_counter()-start,before_query_manifest_sha256=sha(out/'before_query_manifest.json'),
        query_rows_sha256=sha(out/'query_rows.json'),marginal_evaluation_sha256=sha(out/'marginal_evaluation.json'),
        methods={m:float(np.mean([r['mse'] for r in evaluation if r['method']==m])) for m in pools},marginal_evaluation=marginal_eval)
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
