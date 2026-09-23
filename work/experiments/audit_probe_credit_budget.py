"""Both285 budget sets: independent original-horizon paths and full geometry."""
import argparse,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import probe_continuation_reference as independent
from verify_probe_credit_resources import local_reference
from audit_local_dual_jump_modes import modes
import shared_mode_readout as shared
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from audit_shared_mode_readout import check_cube
from audit_confirmation_posterior_moments import piecewise_forward
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_budget';out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    directories={'main':base/'development','sensitivity':root/'results/probe_credit_budget_sensitivity/development'};hashes={};new={};configs=[];manifests={};inputs={};protocols={}
    for group,directory in directories.items():
        ss=json.loads((directory/'summary.json').read_text());assert ss['passed'] and ss['predictions']==256
        before=json.loads((directory/'before_evaluation_manifest.json').read_text());assert sha(directory/'before_evaluation_manifest.json')==ss['before_evaluation_manifest_sha256'];p=json.loads((directory/'protocol.json').read_text());protocols[group]=p
        for name,h in p['source_sha256'].items():
            if name in hashes:assert hashes[name]==h
            hashes[name]=h
        for n in ['protocol','rows']:assert sha(directory/(n+'.json'))==before[n+'_sha256']
        for row in json.loads((directory/'rows.json').read_text()):new[row['seed'],row['method']]=row;inputs[row['method']]=directory
        configs.extend(p['configs']);manifests[group]=before
    assert len(configs)==len({c['name'] for c in configs})==8
    hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    old=root/'results/certificate_activity_attribution/development';oldrows={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text())}
    reference=root/'results/confirmation_conditional_risk/reference';refaudit=reference.parent/'reference_audit';ra=json.loads((refaudit/'summary.json').read_text());assert ra['passed'] and ra['complete_up_to_certified_zero_volume']==64
    refs={r['seed']:r for r in json.loads((reference/'coverage.json').read_text())};seeds=list(range(5910000,5910064))
    dump(out/'protocol.json',dict(source_sha256=hashes,development_summaries={group:sha(d/'summary.json') for group,d in directories.items()},configs=configs,seeds=seeds,reference_audit_sha256=sha(refaudit/'summary.json'),phase_accesses_query_targets=False,
        scope='512 predictors, fixed1.00 main and separate1.10 sensitivity; independent original Local/Adam, full trials and common geometry'))
    counts=Counter();maxforward=0.;maxvolume=0.;geometries=[];coverage=[];begin=time.perf_counter()
    with discovery_box(.12):
        for seed in seeds:
            refrow=refs[seed];assert sha(reference/refrow['file'])==refrow['sha256'];ref=json.loads((reference/refrow['file']).read_text());positive={r['pattern']:r for r in ref['reference']['positive_regions']}
            x=np.array(ref['x_observed']);v=np.array(ref['v_observed']);q=np.linspace(0,1,257);cache={}
            for cfg in configs:
                directory=inputs[cfg['name']];group=next(k for k,d in directories.items() if d==directory);row=new[seed,cfg['name']];assert sha(directory/row['file'])==row['sha256']==manifests[group]['prediction_files'][row['file']]
                with np.load(directory/row['file']) as z:
                    assert z['x_observed'].tobytes()==x.tobytes() and z['v_observed'].tobytes()==v.tobytes() and z['q_observed'].tobytes()==q.tobytes()
                    if row['metadata']['execution_failed']:
                        assert not z['selected_b'].any();assert np.max(abs(piecewise_forward(q,np.zeros((1,4)))[0]-z['prediction']))<1e-12;counts['retained_failures']+=1;continue
                    goldname='credit_control_alm33' if cfg['group']=='plain_alm' else 'credit_control_probe33';goldrow=oldrows[seed,goldname];assert sha(old/goldrow['file'])==goldrow['sha256']
                    with np.load(old/goldrow['file']) as gold:
                        for key in ['initial_b','initial_h','initial_u','initial_best','atomic_trial_b','atomic_trial_origins']:
                            assert z[key].tobytes()==gold[key].tobytes();counts['shared_initial_arrays']+=1
                    assert row['metadata']['atomic_work']==goldrow['metadata']['atomic_work']
                    if cfg['family']=='local':
                        local_reference(cfg['config'],x,v,z,counts)
                        bank=z['prefix_best'][-1].copy();bank[0]=z['anchor_best'][-1,0]
                        assert bank.tobytes()==z['best_bank'].tobytes();_,selected0=independent.cold.select(bank,x,v);assert selected0.tobytes()==z['selected_b'].tobytes()
                        seen=modes(x,z['atomic_trial_b']);seen.update(modes(x,z['prefix_b'].reshape(-1,4)));seen.update(modes(x,z['anchor_b'].reshape(-1,4)))
                        assert sorted(seen)==row['metadata']['visited_modes']
                        if cfg['config']['solver'] in ['pc','nodual']:
                            assert not z['prefix_u'].any() and not z['anchor_u'].any() and not z['prefix_active'].any() and not z['anchor_active'].any();counts['zero_dual_predictors']+=1
                    else:
                        assert cfg['family']=='probe' and cfg['config']['solver']=='adam'
                        states,cc=independent.reference(cfg['config'],x,v);counts.update(cc);seen,cc=independent.verify(cfg['config'],x,v,z,row['metadata'],states);counts.update(cc)
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
                    meta=row['metadata'];cc=cfg['config'];expected=33*cc['prefix']+cc['extra'] if cfg['family']=='local' else 33*cc['steps']
                    assert meta['atomic_calls']==33 and meta['preparation_restart_sweeps']==528 and meta['total_restart_sweeps']==expected
                    coverage.append(dict(seed=seed,method=cfg['name'],budget_group=group,keys=keys,numerical_mass=sum(positive[k]['volume'] for k in keys)/sum(r['volume'] for r in positive.values())));counts['predictors']+=1
            counts['tasks']+=1
            if counts['tasks']%8==0:print(json.dumps(dict(tasks=counts['tasks'],total=64,seconds=time.perf_counter()-begin)),flush=True)
    assert counts['predictors']+counts['retained_failures']==512
    dump(out/'geometries.json',geometries);dump(out/'coverage.json',coverage)
    ans=dict(passed=True,counts=counts,maximum_independent_forward_gap=maxforward,maximum_relative_reference_volume_gap=maxvolume,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),geometries_sha256=sha(out/'geometries.json'),coverage_sha256=sha(out/'coverage.json'),phase_accesses_query_targets=False,may_evaluate_frozen_development_predictions=True,next='All77 quality and608 descriptive paired intervals, main/sensitivity resources separate')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
