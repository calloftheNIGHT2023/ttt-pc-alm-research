"""260: all frozen point and common-mode predictions before query evaluation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import cold_stagnation_switch as cold
import shared_mode_readout as shared
from audit_local_dual_jump_modes import modes
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    inp=root/'results/minimum_sufficient_dual/continuation';out=root/'results/minimum_dual_query/development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    final=root/'results/round_259_audit.json';old=json.loads(final.read_text());hashes=dict(old['source_sha256'])
    ca=root/'results/minimum_sufficient_dual/continuation_audit';ma=root/'results/minimum_sufficient_dual/mode_audit'
    assert json.loads((ca/'summary.json').read_text())['passed'] and json.loads((ma/'summary.json').read_text())['passed']
    hashes.update(json.loads((ca/'protocol.json').read_text())['source_sha256']);hashes['audit_minimum_dual_modes.py']=sha(src/'audit_minimum_dual_modes.py')
    for n,h in hashes.items():assert sha(src/n)==h,n
    design=root/'outputs/ttt-pc-alm-research/260_minimum_dual_continuation_query_protocol.md';assert sha(design)==old['report_sha256'][design.name]
    hashes[Path(__file__).name]=sha(Path(__file__));cp=json.loads((inp/'protocol.json').read_text());methods=[c['name'] for c in cp['configs']]
    oldshared=root/'results/shared_mode_readout/development';sp=json.loads((oldshared/'protocol.json').read_text())
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(final),design_sha256=sha(design),continuation_summary_sha256=sha(inp/'summary.json'),configs=cp['configs'],methods=methods,
        seeds=cp['seeds'],sample_count=2048,repetition_seeds=list(range(249911,249916)),query_points=257,primary='minimum_dual_alm64',
        scope='3 frozen old-development trajectories; all point and mode predictors committed before query; no new search or official TTT claim')
    for key in ['seeds','sample_count','repetition_seeds','query_points']:assert p[key]==sp[key]
    dump(out/'protocol.json',p);oldrows={(r['seed'],r['method']):r for r in json.loads((inp/'support_rows.json').read_text())};q=np.linspace(0,1,257)
    rows=[];geometry_files=[];representative_files=[];begin=time.perf_counter()
    with cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for seed in p['seeds']:
            own={};selected={};keys={};union={};x=None;v=None
            for method in methods:
                row=oldrows[seed,method];path=inp/row['file'];assert sha(path)==row['sha256']
                with np.load(path) as a:xx=a['x'];vv=a['v'];bank=np.r_[a['atomic_visited'],a['b'].reshape(-1,4)];bb=a['selected_b']
                if x is not None:assert np.array_equal(x,xx) and np.array_equal(v,vv)
                x,v=xx,vv;selected[method]=bb;keys[method]=row['feasible_modes'];found=modes(x,bank);assert sorted(found)==row['visited_modes']
                own[method]={k:found[k] for k in keys[method]}
                path=out/f'representatives_{seed}_{method}.npz';np.savez_compressed(path,keys=np.array(keys[method]),b=np.array([own[method][k] for k in keys[method]]).reshape(-1,4))
                representative_files.append(dict(seed=seed,method=method,file=path.name,sha256=sha(path)))
                for k,b in own[method].items():
                    if k not in union:union[k]=b.copy()
                    else:
                        # Geometry depends only on the same actual pattern and observations.
                        left=cold.base.branch_polytope(x,v,b);right=cold.base.branch_polytope(x,v,union[k])
                        assert all(a.tobytes()==c.tobytes() for a,c in zip(left,right))
            polys={};notes={}
            for index,key in enumerate(sorted(union)):
                poly,note=shared.build(x,v,union[key]);note.update(key=key,representative=union[key].tolist())
                if poly is not None:
                    path=out/f'poly_{seed}_{index}.npz';np.savez_compressed(path,**poly);note.update(file=path.name,sha256=sha(path));polys[key]=poly
                notes[key]=note
            path=out/f'{seed}_geometry.json';dump(path,dict(seed=seed,regions=notes));geometry_files.append(dict(seed=seed,file=path.name,sha256=sha(path)))
            for method in methods:
                prediction=cold.base.forward(q,selected[method]);path=out/f'point_{seed}_{method}.npz';np.savez_compressed(path,x=x,v=v,q=q,b=selected[method],prediction=prediction)
                rows.append(dict(seed=seed,method=method,readout='point',repetition=0,file=path.name,sha256=sha(path)))
                pool=sorted(k for k in keys[method] if k in polys)
                for repetition,rs in enumerate(p['repetition_seeds']):
                    start=time.perf_counter()
                    if pool:points,allocation=shared.draw(polys,pool,2048,np.random.default_rng(np.random.SeedSequence([rs,seed,2048])))
                    else:points=selected[method][None].copy();allocation=np.empty(0,dtype=int)
                    sampling=time.perf_counter()-start;start=time.perf_counter();prediction=shared.geometry.make_predict(points)(q);reading=time.perf_counter()-start
                    path=out/f'mode_{seed}_{method}_{repetition}.npz';np.savez_compressed(path,x=x,v=v,q=q,points=points,allocation=allocation,prediction=prediction)
                    rows.append(dict(seed=seed,method=method,readout='mode',repetition=repetition,file=path.name,sha256=sha(path),mode_keys=pool,fallback=not bool(pool),
                        volume=sum(float(polys[k]['volume']) for k in pool),sampling_seconds=sampling,read_seconds=reading,persistent_predictor_numeric_bytes=points.nbytes))
            dump(out/'rows.json',rows);dump(out/'geometry_files.json',geometry_files);dump(out/'representative_files.json',representative_files)
            print(json.dumps(dict(seed=seed,positive_modes=len(polys),predictions=len(rows))),flush=True)
    # Validate all frozen reference inputs/predictors, not their risks, before evaluation.
    reference=[];coldroot=root/'results/cold_stagnation_switch/development'
    for row in json.loads((coldroot/'support_rows.json').read_text()):
        path=coldroot/row['file'];assert sha(path)==row['file_sha256']
        with np.load(path) as a:xx=a['x'];vv=a['v'];qq=a['q']
        with np.load(out/f'point_{row["seed"]}_{methods[0]}.npz') as a:assert np.array_equal(xx,a['x']) and np.array_equal(vv,a['v']) and np.array_equal(qq,q)
        reference.append(dict(seed=row['seed'],method='cold__'+row['method'],readout='point',repetition=0,file=str(path.relative_to(root)),sha256=sha(path)))
    for row in json.loads((oldshared/'rows.json').read_text()):
        if row['method'] not in sp['methods']:continue
        path=oldshared/row['file'];assert sha(path)==row['sha256']
        with np.load(path) as a:xx=a['x'];vv=a['v'];qq=a['q']
        with np.load(out/f'point_{row["seed"]}_{methods[0]}.npz') as a:assert np.array_equal(xx,a['x']) and np.array_equal(vv,a['v']) and np.array_equal(qq,q)
        reference.append(dict(seed=row['seed'],method='cold__'+row['method'],readout='mode',repetition=row['repetition'],file=str(path.relative_to(root)),sha256=sha(path)))
    oldquery=root/'results/dual_jump_query/development';oqp=json.loads((oldquery/'protocol.json').read_text());oqs=json.loads((oldquery/'summary.json').read_text())
    assert oqs['passed'] and oqs['complete']
    for key in ['seeds','sample_count','repetition_seeds','query_points']:assert p[key]==oqp[key]
    manifest=json.loads((oldquery/'before_query_manifest.json').read_text());assert sha(oldquery/'before_query_manifest.json')==oqs['before_query_manifest_sha256'];assert sha(oldquery/'rows.json')==manifest['rows_sha256']
    for row in json.loads((oldquery/'rows.json').read_text()):
        path=oldquery/row['file'];assert sha(path)==row['sha256']
        with np.load(path) as a:xx=a['x'];vv=a['v'];qq=a['q']
        with np.load(out/f'point_{row["seed"]}_{methods[0]}.npz') as a:assert np.array_equal(xx,a['x']) and np.array_equal(vv,a['v']) and np.array_equal(qq,q)
        reference.append(dict(seed=row['seed'],method=row['method'],readout=row['readout'],repetition=row['repetition'],file=str(path.relative_to(root)),sha256=sha(path)))
    for row in reference:row['reference']=True
    assert len(rows)==288 and len(reference)==2368
    dump(out/'reference_rows.json',reference)
    before=dict(protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),reference_rows_sha256=sha(out/'reference_rows.json'),
        representative_files_sha256=sha(out/'representative_files.json'),geometry_files_sha256=sha(out/'geometry_files.json'),
        arrays={p.name:sha(p) for p in out.glob('*.npz')},geometry={p.name:sha(p) for p in out.glob('*_geometry.json')})
    dump(out/'before_query_manifest.json',before)
    # First new-query target access: all288 new predictions and2368 references are fixed.
    query=[]
    for row in rows+reference:
        path=root/row['file'] if row.get('reference',False) else out/row['file']
        with np.load(path) as a:prediction=a['prediction']
        teacher=np.random.default_rng(row['seed']).uniform(-.12,.12,4);truth=cold.base.forward(q,teacher)
        query.append({**{k:row[k] for k in ['seed','method','readout','repetition']},'mse':float(np.mean((prediction-truth)**2))})
    dump(out/'query_rows.json',query)
    for n,h in hashes.items():assert sha(src/n)==h,n
    result=dict(passed=True,complete=True,new_predictions=len(rows),reference_predictions=len(reference),seconds=time.perf_counter()-begin,
        before_query_manifest_sha256=sha(out/'before_query_manifest.json'),query_rows_sha256=sha(out/'query_rows.json'),
        methods={m:{kind:float(np.mean([r['mse'] for r in query if r['method']==m and r['readout']==kind])) for kind in ['point','mode'] if any(r['method']==m and r['readout']==kind for r in query)} for m in sorted(set(r['method'] for r in query))})
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
