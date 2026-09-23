"""Independent original scalar-start references for283, without new fit."""
from collections import Counter
import numpy as np
import cold_stagnation_switch as cold
import local_dual_jump_transfer as transfer
from audit_local_dual_jump_modes import modes


def reference(cfg,x,v):
    starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(32,4))]
    states=[];counts=Counter()
    for r,initial in enumerate(starts):
        prep=cold.Local(initial[None],x,v,'nodual')
        for _ in range(16):prep.step()
        atom,note=transfer.run(prep.b[0],prep.h[:,0],prep.best[0],x,v,None,'branch_probe')
        best=atom['best'][None].copy();cold.frozen.retain(best,atom['b'][None],x,v,np.zeros(4));seen=modes(x,atom['trial_b']);hist=None;roles=None
        if cfg['solver'] in ['alm','pc','nodual']:
            local=cold.Local(atom['b'][None],x,v,cfg['solver']);local.h=atom['h'][:,None].copy();local.u=atom['u'][:,None].copy();local.best=best.copy();local.errors,local.moves=cold.base.score(best,x,v,np.zeros(4))
            history={k:[value] for k,value in local.arrays().items()}
            for _ in range(64 if r==0 else 32):
                local.step()
                for k,value in local.arrays().items():history[k].append(value)
            hist={k:np.array(value) for k,value in history.items()};best=local.best.copy();seen.update(modes(x,hist['b'][:,0]))
        elif cfg['solver']=='adam':
            saved=cold.bp.evaluate;history=[];roles=[]
            def record(bs,xx,vv,with_jacobian=True):
                history.append(bs.copy());roles.append(with_jacobian);return saved(bs,xx,vv,with_jacobian)
            cold.bp.evaluate=record
            try:bpbest,bpmeta=cold.bp.refine(atom['b'][None],x,v,np.zeros(4),solver='adam',steps=cfg['steps'],lr=.003)
            finally:cold.bp.evaluate=saved
            hist=np.array(history);cold.frozen.retain(best,bpbest,x,v,np.zeros(4));seen.update(modes(x,hist[:,0]));counts['bp_evaluations']+=len(hist)
        else:assert cfg['solver']=='archive';seen.update(modes(x,atom['b'][None]))
        states.append(dict(atom=atom,note=note,history=hist,roles=roles,best=best[0],seen=seen));counts['scalar_paths']+=1
    return states,counts


def verify(cfg,x,v,a,meta,states):
    count=Counter();trials=[];origins=[];seen={};work=Counter();best=[]
    assert meta['continuation_solver']==cfg['solver'] and meta['scans']['scanned_states']==0
    assert a['origins'].tolist()==list(range(33)) and a['assigned_actions'].tolist()==['Q']*33
    for r,state in enumerate(states):
        atom=state['atom'];history=state['history'];best.append(state['best']);seen.update(state['seen']);trials.extend(atom['trial_b']);origins.extend([r]*len(atom['trial_b']))
        for key in ['b','h','u','best']:
            aa=a['initial_'+key][r] if key in ['b','best'] else a['initial_'+key][:,r]
            assert aa.tobytes()==atom[key].tobytes(),(cfg['name'],r,key);count['initial_arrays']+=1
        for key in ['activity_blocks','bias_blocks','dual_updates','activity_branch_proposals','dual_writes']:work[key]+=state['note'].get(key,0)
        if cfg['solver'] in ['alm','pc','nodual']:
            for key in ['b','best','active','first_trigger']:
                assert a['prefix_'+key][:,r].tobytes()==history[key][:33,0].tobytes();count['prefix_arrays']+=1
                if r==0:assert a['anchor_'+key][:,0].tobytes()==history[key][32:65,0].tobytes();count['anchor_arrays']+=1
            for key in ['h','u']:
                assert a['prefix_'+key][:,:,r].tobytes()==history[key][:33,:,0].tobytes();count['prefix_arrays']+=1
                if r==0:assert a['anchor_'+key][:,:,0].tobytes()==history[key][32:65,:,0].tobytes();count['anchor_arrays']+=1
        elif cfg['solver']=='adam':
            assert a['bp_b'][:,r].tobytes()==history[:,0].tobytes();assert a['bp_roles'].tolist()==state['roles'];count['bp_histories']+=1
    assert a['atomic_trial_b'].tobytes()==np.array(trials).tobytes() and a['atomic_trial_origins'].tolist()==origins
    assert not a['effective_initial_u'].any();assert meta['atomic_work']==dict(work) and meta['atomic_trial_points']==len(trials)
    assert np.array(best).tobytes()==a['best_bank'].tobytes();_,selected=cold.select(np.array(best),x,v);assert selected.tobytes()==a['selected_b'].tobytes()
    assert sorted(seen)==meta['visited_modes'];count['all_atomic_trial_points']+=len(trials)
    if cfg['solver'] in ['pc','nodual']:
        assert not a['prefix_u'].any() and not a['anchor_u'].any();assert not a['prefix_active'].any() and not a['anchor_active'].any();count['zero_dual_predictors']+=1
    return seen,count
