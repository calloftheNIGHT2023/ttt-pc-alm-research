"""252 support-only transfer test: all272 states and every direct branch trial."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import local_dual_jump_transfer as transfer
from run_local_dual_jump_v2 import dump
from run_multiplier_fixed_point_screen import sha

def key(x,b):return transfer.base.pattern(x,b).astype(np.uint8).tobytes().hex()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/local_dual_jump/transfer';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    screen=root/'results/local_dual_jump/screen_v2';audit=root/'results/local_dual_jump/audit';parent=json.loads((audit/'protocol.json').read_text());assert json.loads((audit/'summary.json').read_text())['passed']
    hashes=parent['source_sha256'];hashes.update({name:sha(src/name) for name in ['local_dual_jump_transfer.py',Path(__file__).name]})
    for name,h in hashes.items():assert sha(src/name)==h,name
    design=root/'outputs/ttt-pc-alm-research/252_local_dual_jump_transfer_protocol.md'
    p=dict(source_sha256=hashes,design_sha256=sha(design),audit_sha256=sha(audit/'summary.json'),screen_summary_sha256=sha(screen/'summary.json'),methods=transfer.METHODS,
        states=272,query_targets_accessed=False,source='all247 nodual128 step16 states with original incumbents',scope='atomic parameter transfer, no full-optimizer or task-risk superiority claim')
    dump(out/'protocol.json',p);files=json.loads((screen/'files.json').read_text());rows=[];geometry_files=[];begin=time.perf_counter()
    with transfer.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for f in files:
            seed=f['seed'];source=root/'results/cold_stagnation_switch/development'/f['source_file'];assert sha(source)==f['source_sha256'];assert sha(screen/f['file'])==f['sha256']
            with np.load(source) as a:x=a['x'];v=a['v'];bank=a['b'][16];activities=a['h'][16];incumbents=a['best'][16]
            records=json.loads((screen/f['file']).read_text());cache={}
            for record in records:
                r=record['restart'];b=bank[r];h=activities[:,r];old=incumbents[r];old_error,old_move=transfer.base.score(old[None],x,v,np.zeros(4))
                for method in p['methods']:
                    start=time.perf_counter();arrays,meta=transfer.run(b,h,old,x,v,record['selected'],method);elapsed=time.perf_counter()-start
                    current=arrays['b'];best=arrays['best'];keys=[]
                    for trial in arrays['trial_b']:
                        k=key(x,trial);keys.append(k)
                        if k not in cache:cache[k]=transfer.base.branch_feasibility(x,v,trial)
                    err,mov=transfer.base.score(best[None],x,v,np.zeros(4));current_error=transfer.base.score(current[None],x,v,np.zeros(4))[0][0]
                    path=out/f'{seed}_{r}_{method}.npz';np.savez_compressed(path,x=x,v=v,initial_b=b,initial_h=h,incumbent=old,**arrays)
                    rows.append(dict(seed=seed,restart=r,method=method,triggered=record['selected'] is not None,event=record['selected'],file=path.name,sha256=sha(path),
                        parameter_changed=bool(not np.array_equal(current,b)),forward_branch_changed=key(x,current)!=key(x,b),current_error=float(current_error),
                        best_error=float(err[0]),best_improved=bool(transfer.base.better(err,mov,old_error,old_move)[0]),best_feasible=bool(err[0]<=.001001),
                        current_feasible=bool(current_error<=.001001),visited_mode_keys=sorted(set(keys)),current_key=key(x,current),
                        feasible_mode_keys=sorted(k for k in set(keys) if cache[k]['current_branch_feasible']),seconds_atomic_excludes_search_preparation_geometry=elapsed,metadata=meta))
            path=out/f'{seed}_geometry.json';dump(path,cache);geometry_files.append(dict(file=path.name,sha256=sha(path)));dump(out/'support_rows.json',rows)
            print(json.dumps(dict(seed=seed,rows=len(rows),visited_modes=len(cache))),flush=True)
    dump(out/'geometry_files.json',geometry_files)
    summary={}
    for method in p['methods']:
        subset=[r for r in rows if r['method']==method]
        summary[method]=dict(states=len(subset),**{name:sum(r[name] for r in subset) for name in ['parameter_changed','forward_branch_changed','best_improved','best_feasible','current_feasible']},
            feasible_modes=sum(len(set(k for r in subset if r['seed']==seed for k in r['feasible_mode_keys'])) for seed in range(5900000,5900016)),
            actual_trial_points=sum(len(r['metadata']['trial_roles']) for r in subset),seconds_atomic_only=sum(r['seconds_atomic_excludes_search_preparation_geometry'] for r in subset))
    result=dict(passed=True,states=272,method_states=len(rows),methods=summary,query_targets_accessed=False,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),support_rows_sha256=sha(out/'support_rows.json'),geometry_files_sha256=sha(out/'geometry_files.json'))
    for name,h in hashes.items():assert sha(src/name)==h,name
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
