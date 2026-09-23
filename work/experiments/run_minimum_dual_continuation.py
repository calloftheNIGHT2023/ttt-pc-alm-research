"""260 phase1: frozen three-way continuation, all16 tasks, query-free."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import cold_stagnation_switch as cold
from run_local_dual_jump_continuation import initial,local_run
from run_local_dual_jump_v2 import dump
from run_multiplier_fixed_point_screen import sha

CONFIGS=[dict(name='minimum_dual_alm64',initial='minimum_dual',method='alm',steps=64),
         dict(name='minimum_activity_alm64',initial='minimum_activity',method='alm',steps=64),
         dict(name='minimum_reset_alm64',initial='minimum_dual',method='alm',steps=64,reset=True)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/minimum_sufficient_dual';parent=base/'transfer';out=base/'continuation';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    final=root/'results/round_259_audit.json';old=json.loads(final.read_text());hashes=dict(old['source_sha256'])
    for n,h in hashes.items():assert sha(src/n)==h,n
    hashes[Path(__file__).name]=sha(Path(__file__));design=root/'outputs/ttt-pc-alm-research/260_minimum_dual_continuation_query_protocol.md';assert sha(design)==old['report_sha256'][design.name]
    summary=json.loads((parent/'summary.json').read_text());assert sha(parent/'support_rows.json')==summary['support_rows_sha256']
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(final),design_sha256=sha(design),transfer_summary_sha256=sha(parent/'summary.json'),configs=CONFIGS,seeds=list(range(5900000,5900016)),
        primary='minimum_dual_alm64',query_targets_accessed=False,scope='all fixed states and original incumbents; full cold preparation and search must be charged later')
    dump(out/'protocol.json',p);lookup={(r['seed'],r['restart'],r['method']):r for r in json.loads((parent/'support_rows.json').read_text())};rows=[];begin=time.perf_counter()
    with cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for seed in p['seeds']:
            cache={}
            for cfg in CONFIGS:
                x,v,b,h,u,best,origin,atomic_visited=initial(parent,lookup,seed,cfg);incumbent=best.copy();start=time.perf_counter();answer,history,meta=local_run(b,h,u,best,x,v,cfg);elapsed=time.perf_counter()-start
                selected,bb=cold.select(answer,x,v);err,_=cold.base.score(answer,x,v,np.zeros(4));keys=set();start=time.perf_counter()
                for trial in np.r_[atomic_visited,history['b'].reshape(-1,4)]:
                    key=cold.base.pattern(x,trial).astype(np.uint8).tobytes().hex();keys.add(key)
                    if key not in cache:cache[key]=cold.base.branch_feasibility(x,v,trial)
                geometry=time.perf_counter()-start;path=out/f'{seed}_{cfg["name"]}.npz';np.savez_compressed(path,x=x,v=v,initial_b=b,initial_h=h,initial_u=u,incumbent=incumbent,origin=origin,
                    atomic_visited=atomic_visited,best_bank=answer,selected_b=bb,selected_restart=np.array(selected),**history)
                row=dict(seed=seed,method=cfg['name'],file=path.name,sha256=sha(path),restarts=len(b),support_max_error=float(err[selected]),support_feasible=bool(err[selected]<=.001001),
                    restart_feasible=int(np.sum(err<=.001001)),original_restarts_with_feasible=len(set(origin[err<=.001001].tolist())),visited_modes=sorted(keys),
                    feasible_modes=sorted(k for k in keys if cache[k]['current_branch_feasible']),continuation_seconds_excluding_preparation_search_atomic_geometry_io=elapsed,
                    geometry_diagnostic_seconds=geometry,metadata=meta,max_abs_u=float(np.max(abs(history['u']))),finite=all(np.isfinite(a).all() for a in history.values()))
                assert row['finite'];rows.append(row);print(json.dumps(dict(seed=seed,method=cfg['name'],support_error=row['support_max_error'],feasible_restarts=row['original_restarts_with_feasible'],modes=len(row['feasible_modes']))),flush=True)
            dump(out/f'{seed}_geometry.json',cache);dump(out/'support_rows.json',rows)
    ans=dict(passed=True,configs=3,task_methods=len(rows),seconds=time.perf_counter()-begin,query_targets_accessed=False,
        methods={c['name']:dict(feasible_tasks=sum(r['support_feasible'] for r in rows if r['method']==c['name']),feasible_original_restarts=sum(r['original_restarts_with_feasible'] for r in rows if r['method']==c['name']),
            feasible_modes=sum(len(r['feasible_modes']) for r in rows if r['method']==c['name']),seconds_continuation_only=sum(r['continuation_seconds_excluding_preparation_search_atomic_geometry_io'] for r in rows if r['method']==c['name'])) for c in CONFIGS},
        protocol_sha256=sha(out/'protocol.json'),support_rows_sha256=sha(out/'support_rows.json'),geometry_sha256={f.name:sha(f) for f in out.glob('*_geometry.json')})
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()
