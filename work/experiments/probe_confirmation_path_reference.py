"""Original-block/scalar-start audit references for all287 mode-pool methods."""
from collections import Counter
import numpy as np
import cold_stagnation_switch as cold
import local_dual_jump_transfer as transfer
import expanded_cold_readout as expanded
import probe_continuation_reference as probe_reference
from audit_certificate_activity_attribution import pure_history
from audit_local_dual_jump_modes import modes


def reference(cfg,x,v,arrays,meta):
    count=Counter();family=cfg['family'];seen={}
    if family=='probe':
        states,cc=probe_reference.reference(cfg['config'],x,v);count.update(cc)
        seen,cc=probe_reference.verify(cfg['config'],x,v,arrays,meta,states);count.update(cc)
        return seen,count
    if family=='local':
        c=cfg['config'];starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(32,4))]
        prep=cold.Local(starts,x,v,'nodual')
        for _ in range(16):prep.step()
        atoms=[];trials=[];origins=[];work=Counter()
        for r in range(33):
            atom,note=transfer.run(prep.b[r],prep.h[:,r],prep.best[r],x,v,None,c['atomic']);atoms.append(atom)
            trials.extend(atom['trial_b']);origins.extend([r]*len(atom['trial_b']))
            for k in ['activity_blocks','bias_blocks','dual_updates','activity_branch_proposals','dual_writes']:work[k]+=note.get(k,0)
        initial={k:np.stack([a[k] for a in atoms],axis=1 if k in ['h','u'] else 0) for k in ['b','h','u','best']}
        for k,val in initial.items():assert val.tobytes()==arrays['initial_'+k].tobytes();count['initial_arrays']+=1
        assert np.array(trials).tobytes()==arrays['atomic_trial_b'].tobytes() and origins==arrays['atomic_trial_origins'].tolist()
        assert dict(work)==meta['atomic_work'] and len(trials)==meta['atomic_trial_points'];count['atomic_trial_points']+=len(trials)
        assert initial['u'].tobytes()==arrays['effective_initial_u'].tobytes()
        # Zero-dual branch probes must start at zero. The plain-ALM control
        # deliberately performs an alm1 atomic update and may already have u.
        if c['atomic']=='branch_probe':
            assert not initial['u'].any();count['zero_dual_probe_initializations']+=1
        prefix=c['prefix'];extra=c['extra'];history=pure_history(initial['b'],initial['h'],initial['u'],initial['best'],x,v,prefix+extra,c['solver'])
        for k,val in history.items():
            assert val[:prefix+1].tobytes()==arrays['prefix_'+k].tobytes();count['prefix_arrays']+=1
            tail=val[prefix:prefix+extra+1,:,:1] if k in ['h','u'] else val[prefix:prefix+extra+1,:1]
            assert tail.tobytes()==arrays['anchor_'+k].tobytes();count['anchor_arrays']+=1
        best=history['best'][prefix].copy();best[0]=history['best'][prefix+extra,0]
        assert best.tobytes()==arrays['best_bank'].tobytes();_,selected=cold.select(best,x,v);assert selected.tobytes()==arrays['selected_b'].tobytes()
        for bank in [np.array(trials),history['b'][:prefix+1].reshape(-1,4),history['b'][prefix:prefix+extra+1,0]]:seen.update(modes(x,bank))
        assert sorted(seen)==meta['visited_modes'];count['original_local_batches']+=1
        if c['solver'] in ['pc','nodual']:
            assert not history['u'].any() and not history['active'].any();count['zero_dual_predictors']+=1
        assert meta['total_restart_sweeps']==33*prefix+extra and meta['preparation_restart_sweeps']==528 and meta['atomic_calls']==33
        return seen,count
    assert family=='legacy'
    if cfg['group']=='cold_adam':
        c=cfg['config']['config'];starts=expanded.starts(c['restarts']);assert starts.tobytes()==arrays['starts'].tobytes();best=[]
        for initial in starts:
            saved=cold.bp.evaluate
            def record(b,xx,vv,with_jacobian=True):
                seen.update(modes(x,b));count['original_bp_evaluations']+=1;return saved(b,xx,vv,with_jacobian)
            cold.bp.evaluate=record
            try:bb,_=cold.bp.refine(initial[None],x,v,np.zeros(4),solver=c['method'],steps=c['steps'])
            finally:cold.bp.evaluate=saved
            best.append(bb[0]);count['original_cold_bp_paths']+=1
        assert np.array(best).tobytes()==arrays['best_bank'].tobytes();assert sorted(seen)==meta['visited_modes']
        _,selected=cold.select(np.array(best),x,v);assert selected.tobytes()==arrays['selected_b'].tobytes()
        return seen,count
    assert cfg['group']=='full_probe' and cfg['name']=='probe_all_alm64'
    c=cfg['config']['config'];assert c['all_trials'] and c['initial']=='branch_probe' and c['steps']==64 and c['method']=='alm' and not c.get('bp')
    starts=cold.starts();prep=cold.Local(starts,x,v,'nodual')
    for _ in range(16):prep.step()
    best=[];ntrials=0
    for r in range(len(starts)):
        atom,_=transfer.run(prep.b[r],prep.h[:,r],prep.best[r],x,v,None,c['initial']);seen.update(modes(x,atom['trial_b']));ntrials+=len(atom['trial_b'])
        for j,b in enumerate(atom['trial_b']):
            h=atom['trial_h'][:,j:j+1];u=atom['trial_u'][:,j:j+1]
            history=pure_history(b[None],h,u,atom['best'][None],x,v,64,'alm')
            seen.update(modes(x,history['b'][:,0]));best.append(history['best'][-1,0]);count['original_full_probe_paths']+=1
    assert np.array(best).tobytes()==arrays['best_bank'].tobytes() and ntrials==meta['atomic_trial_points']==meta['restarts']
    assert sorted(seen)==meta['visited_modes'];_,selected=cold.select(np.array(best),x,v);assert selected.tobytes()==arrays['selected_b'].tobytes()
    count['atomic_trial_points']+=ntrials;return seen,count
