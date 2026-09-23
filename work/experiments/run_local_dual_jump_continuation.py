"""254: all-state fixed continuation, no query targets or predictor selection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import cold_stagnation_switch as cold
from run_local_dual_jump_v2 import dump
from run_multiplier_fixed_point_screen import sha

CONFIGS=[dict(name='dual_alm64',initial='dual_jump',method='alm',steps=64),
    dict(name='activity_alm64',initial='activity_only',method='alm',steps=64),
    dict(name='reorder_alm64',initial='reorder_no_dual',method='alm',steps=64),
    dict(name='probe_alm64',initial='branch_probe',method='alm',steps=64),
    dict(name='dual_reset_alm64',initial='dual_jump',method='alm',steps=64,reset=True)]
CONFIGS += [dict(name=f'{method}65',initial='unchanged',method=method,steps=65) for method in ['alm','nodual','pc']]
CONFIGS += [dict(name=name,initial='unchanged',method=method,steps=steps,bp=True) for name,method,steps in [('adam60','adam',60),('adam240','adam',240),('gn20','gauss_newton',20),('gn40','gauss_newton',40)]]
CONFIGS += [dict(name='probe_all_alm64',initial='branch_probe',method='alm',steps=64,all_trials=True)]

def initial(parent,lookup,seed,cfg):
    bb=[];hh=[];uu=[];best=[];origin=[];visited=[]
    for restart in range(17):
        row=lookup[seed,restart,cfg['initial']];assert sha(parent/row['file'])==row['sha256']
        with np.load(parent/row['file']) as a:
            x=a['x'];v=a['v'];visited.extend(a['trial_b'])
            if cfg.get('all_trials'):
                n=len(a['trial_b']);bb.extend(a['trial_b']);hh.extend(a['trial_h'].transpose(1,0,2));uu.extend(a['trial_u'].transpose(1,0,2));best.extend(np.tile(a['best'],(n,1)));origin.extend([restart]*n)
            else:bb.append(a['b']);hh.append(a['h']);uu.append(a['u']);best.append(a['best']);origin.append(restart)
    return x,v,np.array(bb),np.stack(hh,axis=1),np.stack(uu,axis=1),np.array(best),np.array(origin),np.array(visited)

def local_run(b,h,u,best,x,v,cfg):
    state=cold.Local(b,x,v,cfg['method']);state.h=h.copy();state.u=np.zeros_like(u) if cfg.get('reset') else u.copy();state.best=best.copy()
    state.errors,state.moves=cold.base.score(state.best,x,v,np.zeros(4));cold.frozen.retain(state.best,b,x,v,np.zeros(4));state.errors,state.moves=cold.base.score(state.best,x,v,np.zeros(4))
    history={key:[value] for key,value in state.arrays().items()}
    for _ in range(cfg['steps']):
        state.step()
        for key,value in state.arrays().items():history[key].append(value)
    return state.best.copy(),{key:np.array(value) for key,value in history.items()},dict(solver_numeric_bytes=state.numeric_state_bytes())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/local_dual_jump';parent=base/'transfer';out=base/'continuation';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    audit=json.loads((base/'transfer_audit/summary.json').read_text());assert audit['passed'];hashes=json.loads((base/'transfer_audit/protocol.json').read_text())['source_sha256']
    extras=['analyze_local_dual_jump_transfer.py','prove_local_dual_parameter_credit.py',Path(__file__).name];hashes.update({name:sha(src/name) for name in extras})
    for name,h in hashes.items():assert sha(src/name)==h,name
    p=dict(source_sha256=hashes,transfer_audit_sha256=sha(base/'transfer_audit/summary.json'),credit_proof_sha256=sha(base/'credit_proof/summary.json'),
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/254_local_dual_jump_continuation_protocol.md'),configs=CONFIGS,seeds=list(range(5900000,5900016)),
        primary='dual_alm64',query_targets_accessed=False,shared_preparation='nodual16 from17 cold starts; must be charged together with search and atomic step')
    dump(out/'protocol.json',p);lookup={(r['seed'],r['restart'],r['method']):r for r in json.loads((parent/'support_rows.json').read_text())};rows=[];begin=time.perf_counter()
    with cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for seed in p['seeds']:
            cache={}
            for cfg in CONFIGS:
                x,v,b,h,u,best,origin,atomic_visited=initial(parent,lookup,seed,cfg);incumbent=best.copy();start=time.perf_counter()
                if cfg.get('bp'):
                    answer,history,meta=cold.run_bp(b,x,v,cfg['method'],cfg['steps'],True);cold.frozen.retain(best,answer,x,v,np.zeros(4));answer=best.copy();history['best']=best[None]
                else:answer,history,meta=local_run(b,h,u,best,x,v,cfg)
                elapsed=time.perf_counter()-start;selected,bb=cold.select(answer,x,v);err,_=cold.base.score(answer,x,v,np.zeros(4));keys=set();geometry_start=time.perf_counter()
                for trial in np.r_[atomic_visited,history['b'].reshape(-1,4)]:
                    key=cold.base.pattern(x,trial).astype(np.uint8).tobytes().hex();keys.add(key)
                    if key not in cache:cache[key]=cold.base.branch_feasibility(x,v,trial)
                geometry_seconds=time.perf_counter()-geometry_start;path=out/f'{seed}_{cfg["name"]}.npz'
                np.savez_compressed(path,x=x,v=v,initial_b=b,initial_h=h,initial_u=u,incumbent=incumbent,
                    origin=origin,atomic_visited=atomic_visited,best_bank=answer,selected_b=bb,selected_restart=np.array(selected),**history)
                row=dict(seed=seed,method=cfg['name'],file=path.name,sha256=sha(path),restarts=len(b),support_max_error=float(err[selected]),support_feasible=bool(err[selected]<=.001001),
                    restart_feasible=int(np.sum(err<=.001001)),original_restarts_with_feasible=len(set(origin[err<=.001001].tolist())),
                    visited_modes=sorted(keys),feasible_modes=sorted(k for k in keys if cache[k]['current_branch_feasible']),
                    continuation_seconds_excluding_preparation_search_atomic_geometry_io=elapsed,geometry_diagnostic_seconds=geometry_seconds,metadata=meta,
                    max_abs_u=float(np.max(abs(history['u']))) if 'u' in history else None,finite=all(np.isfinite(a).all() for a in history.values()))
                assert row['finite'];rows.append(row);dump(out/'support_rows.json',rows)
                print(json.dumps(dict(seed=seed,method=cfg['name'],support_error=row['support_max_error'],feasible_restarts=row['original_restarts_with_feasible'],modes=len(row['feasible_modes']),seconds=elapsed)),flush=True)
            dump(out/f'{seed}_geometry.json',cache)
    result=dict(passed=True,configs=len(CONFIGS),task_methods=len(rows),seconds=time.perf_counter()-begin,query_targets_accessed=False,
        methods={cfg['name']:dict(feasible_tasks=sum(r['support_feasible'] for r in rows if r['method']==cfg['name']),feasible_original_restarts=sum(r['original_restarts_with_feasible'] for r in rows if r['method']==cfg['name']),
            feasible_modes=sum(len(r['feasible_modes']) for r in rows if r['method']==cfg['name']),seconds_continuation_only=sum(r['continuation_seconds_excluding_preparation_search_atomic_geometry_io'] for r in rows if r['method']==cfg['name'])) for cfg in CONFIGS},
        protocol_sha256=sha(out/'protocol.json'),support_rows_sha256=sha(out/'support_rows.json'),geometry_sha256={f.name:sha(f) for f in out.glob('*_geometry.json')})
    for name,h in hashes.items():assert sha(src/name)==h,name
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
