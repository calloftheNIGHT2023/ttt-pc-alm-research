"""All64 trajectory selection and independent forward audit before new evaluation."""
import argparse,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import cold_stagnation_switch as cold
import local_dual_jump_transfer as transfer
from posterior_confirmation_pipeline import discovery_box
from audit_local_dual_jump_modes import modes
from audit_confirmation_posterior_moments import piecewise_forward
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/fixed_restart_credit';inp=base/'development';out=base/'development_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    ss=json.loads((inp/'summary.json').read_text());assert ss['passed'];assert sha(inp/'before_evaluation_manifest.json')==ss['before_evaluation_manifest_sha256']
    manifest=json.loads((inp/'before_evaluation_manifest.json').read_text());p=json.loads((inp/'protocol.json').read_text())
    for key in ['protocol','rows']:assert sha(inp/(key+'.json'))==manifest[key+'_sha256']
    hashes=dict(p['source_sha256']);parent=root/'results/round_270_audit.json';pa=json.loads(parent.read_text());assert pa['passed']
    for n,h in pa['source_sha256'].items():
        if n in hashes:assert hashes[n]==h
        hashes[n]=h
    hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'protocol.json',dict(source_sha256=hashes,parent_audit_sha256=sha(parent),development_summary_sha256=sha(inp/'summary.json'),
        phase_accesses_query_targets=False,scope='Frozen pure initial states plus independent original Local trajectory replay; no mixed implementation calls; all256 predictors'))
    new={(r['seed'],r['method']):r for r in json.loads((inp/'rows.json').read_text())}
    old=root/'results/matched_budget_confirmation/conditioned_confirmation';oldrows={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text())}
    counts=Counter();maxforward=0.;begin=time.perf_counter()
    with discovery_box(.12):
        for seed in p['seeds']:
            first=new[seed,p['primary']];assert sha(inp/first['file'])==first['sha256']
            with np.load(inp/first['file']) as z:x=z['x_observed'].copy();v=z['v_observed'].copy();q=z['q_observed'].copy()
            prep=cold.Local(cold.starts(),x,v,'nodual')
            for _ in range(16):prep.step()
            pure={}
            for action in ['D','A','R','B']:
                if action!='B':
                    name={'D':'minimum_dual_alm64','A':'minimum_activity_alm64','R':'minimum_reset_alm64'}[action];ref=oldrows[seed,name];assert sha(old/ref['file'])==ref['sha256']
                    with np.load(old/ref['file']) as z:b=z['initial_b'].copy();h=z['initial_h'].copy();u=z['initial_u'].copy()
                else:
                    h=prep.h.copy();u=np.zeros_like(h);b=transfer.bias_pass(prep.b,h,u,x)
                original_u=u.copy()
                if action=='R':u[:]=0.
                state=cold.Local(b,x,v,'alm');state.h=h.copy();state.u=u.copy();state.best=prep.best.copy()
                cold.frozen.retain(state.best,prep.b,x,v,np.zeros(4));cold.frozen.retain(state.best,b,x,v,np.zeros(4))
                state.errors,state.moves=cold.base.score(state.best,x,v,np.zeros(4))
                history={k:[value] for k,value in state.arrays().items()}
                for _ in range(64):
                    state.step()
                    for k,value in state.arrays().items():history[k].append(value)
                pure[action]={k:np.array(value) for k,value in history.items()};pure[action]['atomic_u']=original_u
                counts['independent_pure_trajectories']+=1
            for cfg in p['configs']:
                r=new[seed,cfg['name']];assert sha(inp/r['file'])==r['sha256']==manifest['prediction_files'][r['file']]
                with np.load(inp/r['file']) as z:
                    assert z['x_observed'].tobytes()==x.tobytes() and z['v_observed'].tobytes()==v.tobytes() and z['q_observed'].tobytes()==q.tobytes()
                    if r['metadata']['execution_failed']:
                        assert not z['selected_b'].any();expected=piecewise_forward(q,np.zeros((1,4)))[0];assert np.max(abs(expected-z['prediction']))<1e-12;counts['retained_failures']+=1;continue
                    seen={}
                    actions=[cfg['even'] if i%2==0 else cfg['odd'] for i in range(17)];assert z['assigned_actions'].tolist()==actions==r['metadata']['assigned_actions']
                    for i,action in enumerate(actions):
                        ref=pure[action]
                        for key in ['b','best','active','first_trigger']:
                            assert z['history_'+key][:,i].tobytes()==ref[key][:,i].tobytes(),(seed,cfg['name'],i,key);counts['selected_trajectory_arrays']+=1
                        for key in ['h','u']:
                            assert z['history_'+key][:,:,i].tobytes()==ref[key][:,:,i].tobytes(),(seed,cfg['name'],i,key);counts['selected_trajectory_arrays']+=1
                        assert z['initial_u'][:,i].tobytes()==ref['atomic_u'][:,i].tobytes()
                        assert z['effective_initial_u'][:,i].tobytes()==ref['u'][0,:,i].tobytes()
                        for key,b in modes(x,ref['b'][:,i]).items():seen[key]=b
                    assert sorted(seen)==r['metadata']['visited_modes']
                    bank=np.stack([pure[action]['best'][-1,i] for i,action in enumerate(actions)])
                    assert bank.tobytes()==z['best_bank'].tobytes();_,best=cold.select(bank,x,v);assert best.tobytes()==z['selected_b'].tobytes()
                    predicted=piecewise_forward(q,z['points']).mean(0);point=piecewise_forward(q,z['selected_b'][None])[0]
                    gap=max(float(np.max(abs(predicted-z['prediction']))),float(np.max(abs(point-z['point_prediction']))));assert gap<1e-12;maxforward=max(maxforward,gap)
                    if not r['metadata']['fallback']:
                        assert len(z['points'])==2048 and np.max(abs(piecewise_forward(x,z['points'])-v))<=.001+1e-7
                        assert set(modes(x,z['points']))<=set(r['metadata']['positive_modes'])<=set(seen);counts['support_particles']+=len(z['points'])
                    assert r['metadata']['atomic_calls']==17 and r['metadata']['total_restart_sweeps']==1088;counts['mixed_predictors']+=1
            counts['tasks']+=1
            if counts['tasks']%8==0:print(json.dumps(dict(tasks=counts['tasks'],total=64,seconds=time.perf_counter()-begin)),flush=True)
    assert counts['mixed_predictors']+counts['retained_failures']==256
    ans=dict(passed=True,counts=counts,maximum_independent_forward_gap=maxforward,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),phase_accesses_query_targets=False,
        next='Geometric identity/interior checks and complete development evaluation with all46 old comparators; resource-matched confirmation not yet done')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
