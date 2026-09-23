"""271 geometry gate: all positive cells, exact interiors and original draws."""
import argparse,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import shared_mode_readout as shared
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from audit_shared_mode_readout import check_cube
from audit_local_dual_jump_modes import modes
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/fixed_restart_credit';inp=base/'development';audit=base/'development_audit';out=base/'geometry_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    aa=json.loads((audit/'summary.json').read_text());assert aa['passed'];ap0=json.loads((audit/'protocol.json').read_text());assert sha(inp/'summary.json')==ap0['development_summary_sha256']
    hashes=dict(ap0['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    p=json.loads((inp/'protocol.json').read_text());before=json.loads((inp/'before_evaluation_manifest.json').read_text());assert sha(inp/'rows.json')==before['rows_sha256'];rows=json.loads((inp/'rows.json').read_text())
    reference=root/'results/confirmation_conditional_risk/reference';refaudit=reference.parent/'reference_audit';ra=json.loads((refaudit/'summary.json').read_text());assert ra['passed'] and ra['complete_up_to_certified_zero_volume']==64
    tasks={r['seed']:r for r in json.loads((reference/'coverage.json').read_text())}
    dump(out/'protocol.json',dict(source_sha256=hashes,trajectory_audit_sha256=sha(audit/'summary.json'),reference_audit_sha256=sha(refaudit/'summary.json'),
        phase_accesses_query_targets=False,scope='All256 frozen pools vs complete support reference; exact interior cubes, numerical volumes, original seeded2048 draws; audit cache not online information'))
    counts=Counter();geometries=[];coverage=[];maxrelative=0.;begin=time.perf_counter()
    with discovery_box(.12):
        for seed in p['seeds']:
            rr=tasks[seed];assert sha(reference/rr['file'])==rr['sha256'];ref=json.loads((reference/rr['file']).read_text());positive={r['pattern']:r for r in ref['reference']['positive_regions']};cache={}
            for row in [r for r in rows if r['seed']==seed]:
                assert sha(inp/row['file'])==row['sha256']==before['prediction_files'][row['file']]
                with np.load(inp/row['file']) as z:
                    x=z['x_observed'];v=z['v_observed'];assert x.tobytes()==np.array(ref['x_observed']).tobytes() and v.tobytes()==np.array(ref['v_observed']).tobytes()
                    if row['metadata']['execution_failed']:assert not z['selected_b'].any();counts['retained_failures']+=1;continue
                    seen=modes(x,z['history_b'].reshape(-1,4));assert sorted(seen)==row['metadata']['visited_modes']
                    keys=sorted(set(seen)&set(positive));assert keys==row['metadata']['positive_modes'],(seed,row['method'],'missed_positive_geometry')
                    for key in keys:
                        if key not in cache:
                            repairs=[]
                            with conditioned.geometry_scope(repairs):poly,proof=shared.build(x,v,seen[key])
                            assert poly is not None and proof['proof']['accepted'];cube=check_cube(x,v,key,proof['proof'])
                            volume=float(np.abs(np.linalg.det(poly['facets']-poly['interior'])).sum()/24*np.prod(poly['scale']))
                            assert np.isclose(volume,poly['volume'],rtol=1e-8,atol=1e-22)
                            relative=abs(poly['volume']/positive[key]['volume']-1);assert relative<1e-8,(seed,key,relative);maxrelative=max(maxrelative,relative)
                            filename=f'{seed}_{key}.npz';np.savez_compressed(out/filename,**poly);cache[key]=poly
                            geometries.append(dict(seed=seed,key=key,file=filename,sha256=sha(out/filename),proof=proof['proof'],cube=cube,reference_volume=positive[key]['volume'],relative_volume_gap=relative,repairs=repairs))
                            counts['positive_geometries']+=1;counts['exact_cube_corners']+=16
                    if keys:
                        points,allocation=shared.draw(cache,keys,2048,np.random.default_rng(np.random.SeedSequence([249911,seed,2048])))
                        assert points.tobytes()==z['points'].tobytes() and allocation.tobytes()==z['allocation'].tobytes(),(seed,row['method'],'draw_replay')
                        counts['bytewise_particle_draws']+=1
                    else:
                        assert len(z['points'])==1 and z['points'][0].tobytes()==z['selected_b'].tobytes();counts['empty_pool_fallbacks']+=1
                    coverage.append(dict(seed=seed,method=row['method'],keys=keys,numerical_mass=sum(positive[k]['volume'] for k in keys)/sum(r['volume'] for r in positive.values())))
                    counts['pools']+=1
            if (seed-p['seeds'][0]+1)%8==0:print(json.dumps(dict(tasks=seed-p['seeds'][0]+1,total=64,seconds=time.perf_counter()-begin)),flush=True)
    assert counts['pools']+counts['retained_failures']==256
    dump(out/'geometries.json',geometries);dump(out/'coverage.json',coverage)
    ans=dict(passed=True,counts=counts,maximum_relative_reference_volume_gap=maxrelative,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),geometries_sha256=sha(out/'geometries.json'),coverage_sha256=sha(out/'coverage.json'),
        phase_accesses_query_targets=False,may_evaluate_frozen_development_predictions=True,
        scope='Numerical geometry and exact positive interiors, not exact real volumes or matched resource superiority')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
