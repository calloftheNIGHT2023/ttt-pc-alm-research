"""273 preflight: exact protected anchor, pure prefixes, and new prior starts."""
import argparse,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import anchor_preserving_fork as candidate
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from audit_local_dual_jump_modes import modes
from audit_confirmation_posterior_moments import piecewise_forward
from run_multiplier_fixed_point_screen import sha,dump


def extra_reference(initial,x,v):
    cold=candidate.cold;prep=cold.Local(initial[None],x,v,'nodual')
    for _ in range(16):prep.step()
    record=candidate.jump.scan(x,prep.b[0],prep.h[:,0]);event=candidate.minimum.minimum_event(x,prep.b[0],prep.h[:,0],record['selected'])['selected']
    a,_=candidate.transfer.run(prep.b[0],prep.h[:,0],prep.best[0],x,v,event,'activity_only')
    state=cold.Local(a['b'][None],x,v,'alm');state.h=a['h'][:,None].copy();state.u=a['u'][:,None].copy();state.best=a['best'][None].copy()
    cold.frozen.retain(state.best,state.b,x,v,np.zeros(4));state.errors,state.moves=cold.base.score(state.best,x,v,np.zeros(4))
    history={k:[val] for k,val in state.arrays().items()}
    for _ in range(32):
        state.step()
        for k,val in state.arrays().items():history[k].append(val)
    return {k:np.array(val) for k,val in history.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/anchor_preserving_fork/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    parent=root/'results/round_272_audit.json';pa=json.loads(parent.read_text());assert pa['passed'];hashes=dict(pa['source_sha256'])
    for n in ['anchor_preserving_fork.py',Path(__file__).name]:hashes[n]=sha(src/n)
    for n,h in hashes.items():assert sha(src/n)==h,n
    design='273_anchor_preserving_fork_protocol.md';assert sha(root/'outputs/ttt-pc-alm-research'/design)==pa['report_sha256'][design]
    pure=root/'results/fixed_restart_credit/primitive_v2';pure_rows={(r['seed'],r['method']):r for r in json.loads((pure/'rows.json').read_text())}
    old=root/'results/matched_budget_confirmation/conditioned_confirmation';oldrows={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text())}
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=pa['report_sha256'][design],seeds=[5910000,5910001,5910053,5910063],configs=candidate.CONFIGS,
        discovery_bound=.12,phase_accesses_query_targets=False,scope='Four contexts, exact pure prefixes and64-step anchor, including independently prepared new16 prior points; no quality screening')
    dump(out/'protocol.json',p);counts=Counter();rows=[];begin=time.perf_counter();maxforward=0.
    with discovery_box(.12):
        for seed in p['seeds']:
            r=oldrows[seed,'minimum_dual_alm64'];assert sha(old/r['file'])==r['sha256']
            with np.load(old/r['file']) as z:x=z['x_observed'].copy();v=z['v_observed'].copy();q=z['q_observed'].copy()
            refs={}
            for action in ['D','A','R','B']:
                r=pure_rows[seed,'pure_'+action];assert sha(pure/r['file'])==r['sha256']
                with np.load(pure/r['file']) as z:refs[action]={k.replace('history_',''):z[k].copy() for k in z.files if k.startswith('history_')}
            extended=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(32,4))]
            extra={i:extra_reference(extended[i],x,v) for i in range(17,33)};counts['independent_extra_prior_trajectories']+=16
            for cfg in candidate.CONFIGS:
                repairs=[]
                with conditioned.geometry_scope(repairs):a,meta=candidate.fit(cfg,x,v,q,seed,trace=True)
                assert meta['atomic_calls']==33 and meta['prefix_restart_sweeps']==1056 and meta['anchor_extra_sweeps']==32 and meta['total_restart_sweeps']==1088
                seen={}
                for i,(r,action) in enumerate(zip(a['origins'],a['assigned_actions'])):
                    ref=refs[str(action)] if r<17 else extra[int(r)];rr=int(r) if r<17 else 0
                    for key in ['b','best','active','first_trigger']:
                        assert a['prefix_'+key][:,i].tobytes()==ref[key][:33,rr].tobytes(),(seed,cfg['name'],i,key);counts['prefix_arrays']+=1
                    for key in ['h','u']:
                        assert a['prefix_'+key][:,:,i].tobytes()==ref[key][:33,:,rr].tobytes(),(seed,cfg['name'],i,key);counts['prefix_arrays']+=1
                    for key,b in modes(x,ref['b'][:33,rr]).items():seen[key]=b
                for key in ['b','best','active','first_trigger']:
                    assert a['anchor_'+key][:,0].tobytes()==refs['A'][key][32:65,0].tobytes(),(seed,cfg['name'],'anchor',key);counts['anchor_arrays']+=1
                for key in ['h','u']:
                    assert a['anchor_'+key][:,:,0].tobytes()==refs['A'][key][32:65,:,0].tobytes(),(seed,cfg['name'],'anchor',key);counts['anchor_arrays']+=1
                for key,b in modes(x,refs['A']['b'][32:65,0]).items():seen[key]=b
                assert sorted(seen)==meta['visited_modes'];assert set(modes(x,refs['A']['b'][:,0]))<=set(seen)
                expected=piecewise_forward(q,a['points']).mean(0);gap=float(np.max(abs(expected-a['prediction'])));assert gap<1e-12;maxforward=max(maxforward,gap)
                if not meta['fallback']:assert np.max(abs(piecewise_forward(x,a['points'])-v))<=.001+1e-7
                filename=f'{seed}_{cfg["name"]}.npz';np.savez_compressed(out/filename,x_observed=x,v_observed=v,q_observed=q,**a)
                rows.append(dict(seed=seed,method=cfg['name'],file=filename,sha256=sha(out/filename),metadata=meta,repairs=repairs));counts['candidate_predictors']+=1
            print(json.dumps(dict(seed=seed,completed=len(rows),total=16,seconds=time.perf_counter()-begin)),flush=True)
    dump(out/'rows.json',rows);ans=dict(passed=True,counts=counts,maximum_independent_forward_gap=maxforward,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),phase_accesses_query_targets=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
