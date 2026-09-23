"""271 primitive: old pure predictors and exact selected per-restart trajectories."""
import argparse
from collections import Counter
import json,os,time
from pathlib import Path
import numpy as np
import mixed_restart_credit as mixed
from posterior_confirmation_pipeline import discovery_box
import conditioned_mode_geometry as conditioned
from audit_local_dual_jump_modes import modes
from analyze_recovered_online_comparison import forward
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/fixed_restart_credit/primitive_v2';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    parent=root/'results/round_268_audit.json';pa=json.loads(parent.read_text());assert pa['passed'];hashes=dict(pa['source_sha256'])
    for n in ['mixed_restart_credit.py',Path(__file__).name]:hashes[n]=sha(src/n)
    for n,h in hashes.items():assert sha(src/n)==h,n
    old=root/'results/matched_budget_confirmation/conditioned_confirmation';oldrows={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text())}
    seeds=[5910000,5910001,5910053,5910063]
    design=root/'outputs/ttt-pc-alm-research/271_fixed_restart_credit_diversity_protocol.md'
    dump(out/'protocol.json',dict(source_sha256=hashes,design_sha256=sha(design),parent_audit_sha256=sha(parent),seeds=seeds,
        configs=mixed.CONFIGS,discovery_bound=.12,failed_attempt_protocol_sha256=sha(root/'results/fixed_restart_credit/primitive/protocol.json'),phase_accesses_query_targets=False,scope='Four preflight contexts including the observed empty-pool case; no query labels, not a quality screen'))
    rows=[];counts=Counter();begin=time.perf_counter()
    for seed in seeds:
        r=oldrows[seed,'minimum_dual_alm64'];assert sha(old/r['file'])==r['sha256']
        with np.load(old/r['file']) as z:x=z['x_observed'].copy();v=z['v_observed'].copy();q=z['q_observed'].copy()
        pure={}
        for action in ['D','A','R','B']:
            repairs=[]
            with conditioned.geometry_scope(repairs):a,meta=mixed.fit(dict(even=action,odd=action),x,v,q,seed,trace=True)
            pure[action]=a
            if action!='B':
                original={'D':'minimum_dual_alm64','A':'minimum_activity_alm64','R':'minimum_reset_alm64'}[action];r=oldrows[seed,original];assert sha(old/r['file'])==r['sha256']
                with np.load(old/r['file']) as z:
                    for key in ['points','allocation','prediction','point_prediction','selected_b','best_bank','initial_b','initial_h','initial_u']:
                        assert a[key].tobytes()==z[key].tobytes(),(seed,action,key);counts['pure_frozen_arrays']+=1
            filename=f'{seed}_pure_{action}.npz';np.savez_compressed(out/filename,**a);rows.append(dict(seed=seed,method='pure_'+action,file=filename,sha256=sha(out/filename),metadata=meta,repairs=repairs));counts['fresh_pure_predictors']+=1
        for cfg in mixed.CONFIGS:
            repairs=[]
            with conditioned.geometry_scope(repairs):a,meta=mixed.fit(cfg,x,v,q,seed,trace=True)
            seen={}
            for r,action in enumerate(a['assigned_actions']):
                ref=pure[str(action)]
                for key in ['history_b','history_best','history_active','history_first_trigger']:
                    assert a[key][:,r].tobytes()==ref[key][:,r].tobytes(),(seed,cfg['name'],r,key);counts['selected_trajectory_arrays']+=1
                for key in ['history_h','history_u']:
                    assert a[key][:,:,r].tobytes()==ref[key][:,:,r].tobytes(),(seed,cfg['name'],r,key);counts['selected_trajectory_arrays']+=1
                assert a['initial_b'][r].tobytes()==ref['initial_b'][r].tobytes()
                for key in ['initial_h','initial_u','effective_initial_u']:assert a[key][:,r].tobytes()==ref[key][:,r].tobytes()
                for key,b in modes(x,ref['history_b'][:,r]).items():seen[key]=b
            assert sorted(seen)==meta['visited_modes'];assert meta['atomic_calls']==17 and meta['total_restart_sweeps']==1088
            assert np.max(abs(forward(x,a['points'])-v))<=.001+1e-7 if not meta['fallback'] else True
            filename=f'{seed}_{cfg["name"]}.npz';np.savez_compressed(out/filename,**a);rows.append(dict(seed=seed,method=cfg['name'],file=filename,sha256=sha(out/filename),metadata=meta,repairs=repairs));counts['mixed_predictors']+=1
        print(json.dumps(dict(seed=seed,completed_contexts=counts['mixed_predictors']//4,seconds=time.perf_counter()-begin)),flush=True)
    dump(out/'rows.json',rows);ans=dict(passed=True,counts=counts,seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),phase_accesses_query_targets=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':
    with discovery_box(.12):
        assert mixed.live.cold.base.BOUND==.12
        main()
