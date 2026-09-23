"""281 scalar-start preflight, golden A33 and genuine PC/nodual trajectories."""
import argparse,json,time,os
from pathlib import Path
from collections import Counter
import numpy as np
import certificate_activity_attribution as candidate
import cold_stagnation_switch as cold
import local_dual_jump_short as jump
import minimum_sufficient_dual as minimum
import local_dual_jump_transfer as transfer
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from audit_local_dual_jump_modes import modes
from audit_confirmation_posterior_moments import piecewise_forward
from run_multiplier_fixed_point_screen import sha,dump


def scalar_reference(initial,cfg,x,v,steps):
    prep=cold.Local(initial[None],x,v,'nodual')
    for _ in range(16):prep.step()
    event=None
    if cfg['certificate']:
        record=jump.scan(x,prep.b[0],prep.h[:,0]);event=minimum.minimum_event(x,prep.b[0],prep.h[:,0],record['selected'])['selected']
    atom,note=transfer.run(prep.b[0],prep.h[:,0],prep.best[0],x,v,event,cfg['atomic'])
    state=cold.Local(atom['b'][None],x,v,cfg['solver']);state.h=atom['h'][:,None].copy();state.u=atom['u'][:,None].copy();state.best=atom['best'][None].copy()
    cold.frozen.retain(state.best,state.b,x,v,np.zeros(4));state.errors,state.moves=cold.base.score(state.best,x,v,np.zeros(4))
    history={key:[value] for key,value in state.arrays().items()}
    for _ in range(steps):
        state.step()
        for key,value in state.arrays().items():history[key].append(value)
    return atom,note,{key:np.array(value) for key,value in history.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/certificate_activity_attribution/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    parent=root/'results/round_280_audit.json';pa=json.loads(parent.read_text());assert pa['passed'];hashes=dict(pa['source_sha256'])
    for n in ['certificate_activity_attribution.py',Path(__file__).name]:hashes[n]=sha(src/n)
    for n,h in hashes.items():assert sha(src/n)==h,n
    design='281_certificate_activity_attribution_protocol.md';assert sha(root/'outputs/ttt-pc-alm-research'/design)==pa['report_sha256'][design]
    old=root/'results/anchor_preserving_fork/development';oldrows={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text())}
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=pa['report_sha256'][design],seeds=[5910000,5910001,5910053,5910063],configs=candidate.CONFIGS,golden=candidate.GOLD,
        phase_accesses_query_targets=False,scope='33 independent scalar preparation/atomic/continuations per control; all probe trials, actual PC/nodual solver, exact oldA golden replay, measured scan calls')
    dump(out/'protocol.json',p);rows=[];counts=Counter();begin=time.perf_counter();maxforward=0.
    starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(32,4))]
    with discovery_box(.12):
        for seed in p['seeds']:
            rr=oldrows[seed,candidate.PRIMARY];assert sha(old/rr['file'])==rr['sha256']
            with np.load(old/rr['file']) as z:x=z['x_observed'].copy();v=z['v_observed'].copy();q=z['q_observed'].copy();gold={k:z[k].copy() for k in z.files if k not in ['x_observed','v_observed','q_observed']}
            for cfg in [candidate.GOLD]+candidate.CONFIGS:
                reference=[scalar_reference(starts[r],cfg,x,v,64 if r==0 else 32) for r in range(33)];counts['scalar_reference_trajectories']+=33
                repairs=[];scan_calls=[];original_scan=jump.scan
                def measured_scan(*args,**kwargs):scan_calls.append(1);return original_scan(*args,**kwargs)
                jump.scan=measured_scan
                try:
                    with conditioned.geometry_scope(repairs):a,meta=candidate.fit(cfg,x,v,q,seed,trace=True)
                finally:jump.scan=original_scan
                assert len(scan_calls)==meta['scans']['scanned_states']==(33 if cfg['certificate'] else 0)
                assert meta['continuation_solver']==cfg['solver'] and meta['preparation_restart_sweeps']==528 and meta['total_restart_sweeps']==1088 and meta['atomic_calls']==33
                assert a['origins'].tolist()==list(range(33)) and a['assigned_actions'].tolist()==[cfg['action']]*33
                seen={};trials=[];trial_origins=[];work=Counter();best=[]
                for r,(atom,note,history) in enumerate(reference):
                    for key in ['b','best','active','first_trigger']:
                        assert a['prefix_'+key][:,r].tobytes()==history[key][:33,0].tobytes(),(seed,cfg['name'],r,key);counts['prefix_arrays']+=1
                    for key in ['h','u']:
                        assert a['prefix_'+key][:,:,r].tobytes()==history[key][:33,:,0].tobytes(),(seed,cfg['name'],r,key);counts['prefix_arrays']+=1
                    for key in ['b','h','u','best']:
                        saved=a['initial_'+key][r] if key in ['b','best'] else a['initial_'+key][:,r];assert saved.tobytes()==atom[key].tobytes();counts['initial_arrays']+=1
                    trials.extend(atom['trial_b']);trial_origins.extend([r]*len(atom['trial_b']));best.append(history['best'][-1,0])
                    for bank in [atom['trial_b'],history['b'][:,0]]:
                        for key,b in modes(x,bank).items():seen[key]=b
                    for key in ['activity_blocks','bias_blocks','dual_updates','activity_branch_proposals','dual_writes']:work[key]+=note.get(key,0)
                assert dict(work)==meta['atomic_work'];assert np.array(trials).tobytes()==a['atomic_trial_b'].tobytes();assert trial_origins==a['atomic_trial_origins'].tolist()
                assert len(trials)==meta['atomic_trial_points'];counts['all_atomic_trial_points']+=len(trials)
                anchor=reference[0][2]
                for key in ['b','best','active','first_trigger']:
                    assert a['anchor_'+key][:,0].tobytes()==anchor[key][32:65,0].tobytes();counts['anchor_arrays']+=1
                for key in ['h','u']:
                    assert a['anchor_'+key][:,:,0].tobytes()==anchor[key][32:65,:,0].tobytes();counts['anchor_arrays']+=1
                assert np.array(best).tobytes()==a['best_bank'].tobytes();assert sorted(seen)==meta['visited_modes']
                if cfg['solver'] in ['pc','nodual']:
                    assert not a['prefix_u'].any() and not a['anchor_u'].any() and not a['prefix_active'].any() and not a['anchor_active'].any();counts['zero_dual_solver_predictors']+=1
                if cfg['name']==candidate.PRIMARY:
                    for key,value in gold.items():assert a[key].tobytes()==value.tobytes(),(seed,key);counts['golden_arrays']+=1
                    assert meta['positive_modes']==rr['metadata']['positive_modes'];counts['golden_predictors']+=1
                gap=float(np.max(abs(piecewise_forward(q,a['points']).mean(0)-a['prediction'])));assert gap<1e-12;maxforward=max(maxforward,gap)
                if not meta['fallback']:assert np.max(abs(piecewise_forward(x,a['points'])-v))<=.001+1e-7
                fn=f'{seed}_{cfg["name"]}.npz';np.savez_compressed(out/fn,x_observed=x,v_observed=v,q_observed=q,**a);rows.append(dict(seed=seed,method=cfg['name'],file=fn,sha256=sha(out/fn),metadata=meta,repairs=repairs));counts['predictors']+=1
            print(json.dumps(dict(seed=seed,predictors=len(rows),total=28,seconds=time.perf_counter()-begin)),flush=True)
    dump(out/'rows.json',rows);ans=dict(passed=True,counts=counts,maximum_independent_forward_gap=maxforward,seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),phase_accesses_query_targets=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
