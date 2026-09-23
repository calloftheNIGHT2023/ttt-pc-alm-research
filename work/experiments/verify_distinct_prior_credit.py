"""275 scalar prior-by-prior preflight, no candidate fit in reference paths."""
import argparse,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import distinct_prior_credit as candidate
import cold_stagnation_switch as cold
import local_dual_jump_short as jump
import minimum_sufficient_dual as minimum
import local_dual_jump_transfer as transfer
import conditioned_mode_geometry as conditioned
from audit_anchor_preserving_fork import pure_history
from posterior_confirmation_pipeline import discovery_box
from audit_local_dual_jump_modes import modes
from audit_confirmation_posterior_moments import piecewise_forward
from run_multiplier_fixed_point_screen import sha,dump


def scalar_reference(initial,action,x,v,steps):
    prep=cold.Local(initial[None],x,v,'nodual')
    for _ in range(16):prep.step()
    record=jump.scan(x,prep.b[0],prep.h[:,0]);event=minimum.minimum_event(x,prep.b[0],prep.h[:,0],record['selected'])['selected']
    method={'D':'dual_jump','R':'dual_jump','A':'activity_only','B':'bias_only'}[action]
    a,_=transfer.run(prep.b[0],prep.h[:,0],prep.best[0],x,v,event,method);assert len(a['trial_b'])==1
    u=a['u'].copy()
    if action=='R':u[:]=0.
    history=pure_history(a['b'][None],a['h'][:,None],u[:,None],a['best'][None],x,v,steps)
    return history


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/distinct_prior_credit/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    parent=root/'results/round_274_audit.json';pa=json.loads(parent.read_text());assert pa['passed'];hashes=dict(pa['source_sha256'])
    for n in ['distinct_prior_credit.py',Path(__file__).name]:hashes[n]=sha(src/n)
    for n,h in hashes.items():assert sha(src/n)==h,n
    design='275_distinct_prior_credit_protocol.md';assert sha(root/'outputs/ttt-pc-alm-research'/design)==pa['report_sha256'][design]
    old=root/'results/anchor_preserving_fork/development';oldrows={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text())}
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=pa['report_sha256'][design],seeds=[5910000,5910001,5910053,5910063],configs=candidate.CONFIGS,
        discovery_bound=.12,phase_accesses_query_targets=False,scope='Scalar preparation and selected atomic action per33 prior points; exact prefix and anchor, allA golden replay; no query-quality selection')
    dump(out/'protocol.json',p);counts=Counter();rows=[];begin=time.perf_counter();maxforward=0.
    starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(32,4))]
    with discovery_box(.12):
        for seed in p['seeds']:
            oldrow=oldrows[seed,'anchor_A33_32_A64'];assert sha(old/oldrow['file'])==oldrow['sha256']
            with np.load(old/oldrow['file']) as z:x=z['x_observed'].copy();v=z['v_observed'].copy();q=z['q_observed'].copy();gold={k:z[k].copy() for k in z.files if k not in ['x_observed','v_observed','q_observed']}
            refs={(r,action):scalar_reference(starts[r],action,x,v,64 if r==0 else 32) for r in range(33) for action in (['A'] if r==0 else ['D','A','R','B'])};counts['independent_scalar_trajectories']+=len(refs)
            primary_arrays=None
            for cfg in candidate.CONFIGS+[dict(name='allA_golden_replay',pattern='allA')]:
                repairs=[]
                with conditioned.geometry_scope(repairs):a,meta=candidate.fit(cfg,x,v,q,seed,trace=True)
                assert meta['atomic_calls']==33 and meta['preparation_restart_sweeps']==528 and meta['total_restart_sweeps']==1088
                assert a['origins'].tolist()==list(range(33));seen={}
                for r,action in enumerate(a['assigned_actions']):
                    ref=refs[r,str(action)]
                    for key in ['b','best','active','first_trigger']:
                        assert a['prefix_'+key][:,r].tobytes()==ref[key][:33,0].tobytes(),(seed,cfg['name'],r,key);counts['prefix_arrays']+=1
                    for key in ['h','u']:
                        assert a['prefix_'+key][:,:,r].tobytes()==ref[key][:33,:,0].tobytes(),(seed,cfg['name'],r,key);counts['prefix_arrays']+=1
                    for key,b in modes(x,ref['b'][:33,0]).items():seen[key]=b
                for key in ['b','best','active','first_trigger']:
                    assert a['anchor_'+key][:,0].tobytes()==refs[0,'A'][key][32:65,0].tobytes();counts['anchor_arrays']+=1
                for key in ['h','u']:
                    assert a['anchor_'+key][:,:,0].tobytes()==refs[0,'A'][key][32:65,:,0].tobytes();counts['anchor_arrays']+=1
                for key,b in modes(x,refs[0,'A']['b'][:,0]).items():seen[key]=b
                assert sorted(seen)==meta['visited_modes']
                if cfg['name']==candidate.PRIMARY:primary_arrays=a
                if cfg['pattern']=='oddR':
                    for key in ['initial_b','initial_h','initial_u','initial_best']:assert a[key].tobytes()==primary_arrays[key].tobytes();counts['causal_initial_arrays']+=1
                    assert not a['effective_initial_u'][:,1::2].any()
                if cfg['pattern']=='allA':
                    for key,value in gold.items():assert a[key].tobytes()==value.tobytes(),(seed,key);counts['golden_arrays']+=1
                    assert meta['positive_modes']==oldrow['metadata']['positive_modes'];counts['golden_predictors']+=1
                gap=float(np.max(abs(piecewise_forward(q,a['points']).mean(0)-a['prediction'])));assert gap<1e-12;maxforward=max(maxforward,gap)
                if not meta['fallback']:assert np.max(abs(piecewise_forward(x,a['points'])-v))<=.001+1e-7
                filename=f'{seed}_{cfg["name"]}.npz';np.savez_compressed(out/filename,x_observed=x,v_observed=v,q_observed=q,**a)
                rows.append(dict(seed=seed,method=cfg['name'],file=filename,sha256=sha(out/filename),metadata=meta,repairs=repairs));counts['candidate_predictors']+=1
            print(json.dumps(dict(seed=seed,completed=len(rows),total=24,seconds=time.perf_counter()-begin)),flush=True)
    dump(out/'rows.json',rows);ans=dict(passed=True,counts=counts,maximum_independent_forward_gap=maxforward,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),phase_accesses_query_targets=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
