"""281 independent originals, actual solvers, all trials and geometry audit."""
import argparse,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import cold_stagnation_switch as cold
import local_dual_jump_transfer as transfer
import shared_mode_readout as shared
import conditioned_mode_geometry as conditioned
import local_dual_jump_short as jump
import minimum_sufficient_dual as minimum
from posterior_confirmation_pipeline import discovery_box
from audit_local_dual_jump_modes import modes
from audit_shared_mode_readout import check_cube
from audit_confirmation_posterior_moments import piecewise_forward
from run_multiplier_fixed_point_screen import sha,dump


def pure_history(b,h,u,best,x,v,steps,solver):
    state=cold.Local(b,x,v,solver);state.h=h.copy();state.u=u.copy();state.best=best.copy()
    cold.frozen.retain(state.best,b,x,v,np.zeros(4));state.errors,state.moves=cold.base.score(state.best,x,v,np.zeros(4))
    history={k:[value] for k,value in state.arrays().items()}
    for _ in range(steps):
        state.step()
        for k,value in state.arrays().items():history[k].append(value)
    return {k:np.array(value) for k,value in history.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/certificate_activity_attribution';inp=base/'development';out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    ss=json.loads((inp/'summary.json').read_text());assert ss['passed'];assert sha(inp/'before_evaluation_manifest.json')==ss['before_evaluation_manifest_sha256']
    before=json.loads((inp/'before_evaluation_manifest.json').read_text());p=json.loads((inp/'protocol.json').read_text());hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    for n in ['protocol','rows']:assert sha(inp/f'{n}.json')==before[n+'_sha256']
    new={(r['seed'],r['method']):r for r in json.loads((inp/'rows.json').read_text())}
    old=root/'results/anchor_preserving_fork/development';oldrows={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text())}
    reference=root/'results/confirmation_conditional_risk/reference';refaudit=reference.parent/'reference_audit';ra=json.loads((refaudit/'summary.json').read_text());assert ra['passed'] and ra['complete_up_to_certified_zero_volume']==64
    refs={r['seed']:r for r in json.loads((reference/'coverage.json').read_text())}
    dump(out/'protocol.json',dict(source_sha256=hashes,development_summary_sha256=sha(inp/'summary.json'),reference_audit_sha256=sha(refaudit/'summary.json'),phase_accesses_query_targets=False,
        scope='All384 control predictors and64 independent A33 golden trajectories; actual solvers, all atomic trials, exact geometry and original draws'))
    counts=Counter();maxforward=0.;maxvolume=0.;geometries=[];coverage=[];begin=time.perf_counter()
    with discovery_box(.12):
        for seed in p['seeds']:
            refrow=refs[seed];assert sha(reference/refrow['file'])==refrow['sha256'];ref=json.loads((reference/refrow['file']).read_text());positive={r['pattern']:r for r in ref['reference']['positive_regions']}
            x=np.array(ref['x_observed']);v=np.array(ref['v_observed']);q=np.linspace(0,1,257)
            starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(32,4))]
            prep=cold.Local(starts,x,v,'nodual')
            for _ in range(16):prep.step()
            events=[]
            for r in range(33):
                record=jump.scan(x,prep.b[r],prep.h[:,r]);events.append(minimum.minimum_event(x,prep.b[r],prep.h[:,r],record['selected'])['selected'])
            def reference_batch(cfg):
                bb=[];hh=[];uu=[];best=[];trials=[];origins=[];work=Counter()
                for r in range(33):
                    event=events[r] if cfg['certificate'] else None
                    atom,note=transfer.run(prep.b[r],prep.h[:,r],prep.best[r],x,v,event,cfg['atomic'])
                    bb.append(atom['b']);hh.append(atom['h']);uu.append(atom['u']);best.append(atom['best'])
                    trials.extend(atom['trial_b']);origins.extend([r]*len(atom['trial_b']))
                    for key in ['activity_blocks','bias_blocks','dual_updates','activity_branch_proposals','dual_writes']:work[key]+=note.get(key,0)
                b=np.array(bb);h=np.stack(hh,axis=1);u=np.stack(uu,axis=1);best=np.array(best)
                history=pure_history(b,h,u,best,x,v,64,cfg['solver'])
                return dict(b=b,h=h,u=u,best=best,history=history,trials=np.array(trials),origins=origins,work=dict(work))
            gold=reference_batch(dict(atomic='activity_only',certificate=True,solver='alm'));golden=oldrows[seed,'anchor_A33_32_A64'];assert sha(old/golden['file'])==golden['sha256']
            with np.load(old/golden['file']) as z:
                for key in ['b','h','u','best']:
                    assert gold[key].tobytes()==z['initial_'+key].tobytes();counts['golden_arrays']+=1
                for key in ['b','best','active','first_trigger','h','u']:
                    assert gold['history'][key][:33].tobytes()==z['prefix_'+key].tobytes();counts['golden_arrays']+=1
            cache={}
            for cfg in p['configs']:
                rr=reference_batch(cfg);counts['independent_control_batches']+=1
                row=new[seed,cfg['name']];assert sha(inp/row['file'])==row['sha256']==before['prediction_files'][row['file']]
                with np.load(inp/row['file']) as z:
                    assert z['x_observed'].tobytes()==x.tobytes() and z['v_observed'].tobytes()==v.tobytes() and z['q_observed'].tobytes()==q.tobytes()
                    if row['metadata']['execution_failed']:
                        assert not z['selected_b'].any();check=piecewise_forward(q,np.zeros((1,4)))[0];assert np.max(abs(check-z['prediction']))<1e-12;counts['retained_failures']+=1;continue
                    assert z['origins'].tolist()==list(range(33)) and z['assigned_actions'].tolist()==[cfg['action']]*33
                    for key in ['b','h','u','best']:
                        assert z['initial_'+key].tobytes()==rr[key].tobytes();counts['initial_arrays']+=1
                    assert z['effective_initial_u'].tobytes()==rr['u'].tobytes()
                    assert z['atomic_trial_b'].tobytes()==rr['trials'].tobytes() and z['atomic_trial_origins'].tolist()==rr['origins']
                    assert row['metadata']['atomic_work']==rr['work'] and row['metadata']['atomic_trial_points']==len(rr['trials'])
                    counts['all_atomic_trial_points']+=len(rr['trials']);history=rr['history']
                    for key in ['b','best','active','first_trigger','h','u']:
                        assert z['prefix_'+key].tobytes()==history[key][:33].tobytes();counts['prefix_batch_arrays']+=1
                    for key in ['b','best','active','first_trigger']:
                        assert z['anchor_'+key][:,0].tobytes()==history[key][32:65,0].tobytes();counts['anchor_arrays']+=1
                    for key in ['h','u']:
                        assert z['anchor_'+key][:,:,0].tobytes()==history[key][32:65,:,0].tobytes();counts['anchor_arrays']+=1
                    if cfg['solver'] in ['pc','nodual']:
                        assert not z['prefix_u'].any() and not z['anchor_u'].any() and not z['prefix_active'].any() and not z['anchor_active'].any();counts['zero_dual_solver_predictors']+=1
                    seen={}
                    for bank in [rr['trials'],history['b'][:33].reshape(-1,4),history['b'][32:65,0]]:
                        for key,b in modes(x,bank).items():seen[key]=b
                    best=history['best'][32].copy();best[0]=history['best'][64,0]
                    assert best.tobytes()==z['best_bank'].tobytes();_,selected=cold.select(best,x,v);assert selected.tobytes()==z['selected_b'].tobytes()
                    assert sorted(seen)==row['metadata']['visited_modes']
                    keys=sorted(set(seen)&set(positive));assert keys==row['metadata']['positive_modes']
                    for key in keys:
                        if key not in cache:
                            repairs=[]
                            with conditioned.geometry_scope(repairs):poly,proof=shared.build(x,v,seen[key])
                            assert poly is not None and proof['proof']['accepted'];cube=check_cube(x,v,key,proof['proof']);gap=abs(float(poly['volume'])/positive[key]['volume']-1)
                            assert gap<1e-8;maxvolume=max(maxvolume,gap);cache[key]=poly;filename=f'{seed}_{key}.npz';np.savez_compressed(out/filename,**poly)
                            geometries.append(dict(seed=seed,key=key,file=filename,sha256=sha(out/filename),proof=proof['proof'],cube=cube,relative_volume_gap=gap,repairs=repairs));counts['positive_geometries']+=1;counts['exact_cube_corners']+=16
                    if keys:
                        points,allocation=shared.draw(cache,keys,2048,np.random.default_rng(np.random.SeedSequence([249911,seed,2048])))
                        assert points.tobytes()==z['points'].tobytes() and allocation.tobytes()==z['allocation'].tobytes();counts['particle_draws']+=1
                        assert np.max(abs(piecewise_forward(x,points)-v))<=.001+1e-7;counts['support_particles']+=len(points)
                    else:assert len(z['points'])==1 and z['points'][0].tobytes()==selected.tobytes();counts['empty_pools']+=1
                    expected=piecewise_forward(q,z['points']).mean(0);point=piecewise_forward(q,selected[None])[0]
                    gap=max(float(np.max(abs(expected-z['prediction']))),float(np.max(abs(point-z['point_prediction']))));assert gap<1e-12;maxforward=max(maxforward,gap)
                    meta=row['metadata'];assert meta['atomic_calls']==33 and meta['prefix_restart_sweeps']==1056 and meta['anchor_extra_sweeps']==32 and meta['total_restart_sweeps']==1088
                    assert meta['preparation_restart_sweeps']==33*16
                    assert meta['continuation_solver']==cfg['solver'] and meta['scans']['scanned_states']==(33 if cfg['certificate'] else 0)
                    coverage.append(dict(seed=seed,method=cfg['name'],keys=keys,numerical_mass=sum(positive[k]['volume'] for k in keys)/sum(r['volume'] for r in positive.values())))
                    counts['predictors']+=1
            counts['tasks']+=1
            if counts['tasks']%8==0:print(json.dumps(dict(tasks=counts['tasks'],total=64,seconds=time.perf_counter()-begin)),flush=True)
    assert counts['predictors']+counts['retained_failures']==384
    dump(out/'geometries.json',geometries);dump(out/'coverage.json',coverage)
    ans=dict(passed=True,counts=counts,maximum_independent_forward_gap=maxforward,maximum_relative_reference_volume_gap=maxvolume,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),geometries_sha256=sha(out/'geometries.json'),coverage_sha256=sha(out/'coverage.json'),
        phase_accesses_query_targets=False,may_evaluate_frozen_development_predictions=True,
        next='Development query/conditional evaluation of all65 methods; then actual resource matching if competitive, no blind claim')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
