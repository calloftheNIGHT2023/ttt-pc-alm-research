"""283 all-task scalar original paths and common geometry, no quality access."""
import argparse,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import probe_continuation_reference as independent
import shared_mode_readout as shared
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from audit_shared_mode_readout import check_cube
from audit_confirmation_posterior_moments import piecewise_forward
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_continuation_credit';inp=base/'development';out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    ss=json.loads((inp/'summary.json').read_text());assert ss['passed'];assert sha(inp/'before_evaluation_manifest.json')==ss['before_evaluation_manifest_sha256']
    before=json.loads((inp/'before_evaluation_manifest.json').read_text());p=json.loads((inp/'protocol.json').read_text());hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    for n in ['protocol','rows']:assert sha(inp/f'{n}.json')==before[n+'_sha256']
    new={(r['seed'],r['method']):r for r in json.loads((inp/'rows.json').read_text())}
    old=root/'results/certificate_activity_attribution/development';oldrows={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text())}
    reference=root/'results/confirmation_conditional_risk/reference';refaudit=reference.parent/'reference_audit';ra=json.loads((refaudit/'summary.json').read_text());assert ra['passed'] and ra['complete_up_to_certified_zero_volume']==64
    refs={r['seed']:r for r in json.loads((reference/'coverage.json').read_text())}
    dump(out/'protocol.json',dict(source_sha256=hashes,development_summary_sha256=sha(inp/'summary.json'),reference_audit_sha256=sha(refaudit/'summary.json'),phase_accesses_query_targets=False,scope='All256 scalar-start trajectories and frozen probe initial states; geometry, draws and forward values'))
    counts=Counter();maxforward=0.;maxvolume=0.;geometries=[];coverage=[];begin=time.perf_counter()
    with discovery_box(.12):
        for seed in p['seeds']:
            refrow=refs[seed];assert sha(reference/refrow['file'])==refrow['sha256'];ref=json.loads((reference/refrow['file']).read_text());positive={r['pattern']:r for r in ref['reference']['positive_regions']}
            x=np.array(ref['x_observed']);v=np.array(ref['v_observed']);q=np.linspace(0,1,257);cache={}
            oldr=oldrows[seed,p['primary']];assert sha(old/oldr['file'])==oldr['sha256']
            with np.load(old/oldr['file']) as z:
                gold={k:z[k].copy() for k in ['initial_b','initial_h','initial_u','initial_best','atomic_trial_b','atomic_trial_origins']}
                first={k:z['prefix_'+k][1].copy() for k in ['b','h','best']}
            for cfg in p['configs']:
                row=new[seed,cfg['name']];assert sha(inp/row['file'])==row['sha256']==before['prediction_files'][row['file']]
                with np.load(inp/row['file']) as z:
                    assert z['x_observed'].tobytes()==x.tobytes() and z['v_observed'].tobytes()==v.tobytes() and z['q_observed'].tobytes()==q.tobytes()
                    if row['metadata']['execution_failed']:
                        assert not z['selected_b'].any();assert np.max(abs(piecewise_forward(q,np.zeros((1,4)))[0]-z['prediction']))<1e-12;counts['retained_failures']+=1;continue
                    refs0,cc=independent.reference(cfg,x,v);counts.update(cc);seen,cc=independent.verify(cfg,x,v,z,row['metadata'],refs0);counts.update(cc)
                    for k,a in gold.items():assert z[k].tobytes()==a.tobytes();counts['shared_initial_arrays']+=1
                    if cfg['solver']=='nodual':
                        for k,a in first.items():assert z['prefix_'+k][1].tobytes()==a.tobytes();counts['first_primal_step_identities']+=1
                    keys=sorted(set(seen)&set(positive));assert keys==row['metadata']['positive_modes'];selected=z['selected_b']
                    for key in keys:
                        if key not in cache:
                            repairs=[]
                            with conditioned.geometry_scope(repairs):poly,proof=shared.build(x,v,seen[key])
                            assert poly is not None and proof['proof']['accepted'];cube=check_cube(x,v,key,proof['proof']);gap=abs(float(poly['volume'])/positive[key]['volume']-1)
                            assert gap<1e-8;maxvolume=max(maxvolume,gap);cache[key]=poly;fn=f'{seed}_{key}.npz';np.savez_compressed(out/fn,**poly)
                            geometries.append(dict(seed=seed,key=key,file=fn,sha256=sha(out/fn),proof=proof['proof'],cube=cube,relative_volume_gap=gap,repairs=repairs));counts['positive_geometries']+=1;counts['exact_cube_corners']+=16
                    if keys:
                        points,allocation=shared.draw(cache,keys,2048,np.random.default_rng(np.random.SeedSequence([249911,seed,2048])))
                        assert points.tobytes()==z['points'].tobytes() and allocation.tobytes()==z['allocation'].tobytes();counts['particle_draws']+=1
                        assert np.max(abs(piecewise_forward(x,points)-v))<=.001+1e-7;counts['support_particles']+=len(points)
                    else:assert len(z['points'])==1 and z['points'][0].tobytes()==selected.tobytes();counts['empty_pools']+=1
                    gap=max(float(np.max(abs(piecewise_forward(q,z['points']).mean(0)-z['prediction']))),float(np.max(abs(piecewise_forward(q,selected[None])[0]-z['point_prediction']))));assert gap<1e-12;maxforward=max(maxforward,gap)
                    meta=row['metadata'];expected=1088 if cfg['solver'] in ['pc','nodual'] else 33*cfg.get('steps',0)
                    assert meta['atomic_calls']==33 and meta['preparation_restart_sweeps']==528 and meta['total_restart_sweeps']==expected
                    coverage.append(dict(seed=seed,method=cfg['name'],keys=keys,numerical_mass=sum(positive[k]['volume'] for k in keys)/sum(r['volume'] for r in positive.values())));counts['predictors']+=1
            counts['tasks']+=1
            if counts['tasks']%8==0:print(json.dumps(dict(tasks=counts['tasks'],total=64,seconds=time.perf_counter()-begin)),flush=True)
    assert counts['predictors']+counts['retained_failures']==256
    dump(out/'geometries.json',geometries);dump(out/'coverage.json',coverage)
    ans=dict(passed=True,counts=counts,maximum_independent_forward_gap=maxforward,maximum_relative_reference_volume_gap=maxvolume,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),geometries_sha256=sha(out/'geometries.json'),coverage_sha256=sha(out/'coverage.json'),phase_accesses_query_targets=False,may_evaluate_frozen_development_predictions=True,next='All69 quality and544 descriptive paired intervals')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
