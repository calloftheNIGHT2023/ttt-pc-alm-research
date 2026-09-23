"""258D: all272 atomic transfers, paired same-activity no-dual control.

The nine old frozen conditions remain explicit hashed comparators. Geometry
is diagnostic only and is never written into any optimizer's parameters.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import local_dual_jump_transfer as transfer
from run_local_dual_jump_transfer import key
from run_local_dual_jump_v2 import dump
from run_multiplier_fixed_point_screen import sha

METHODS={'minimum_dual':'dual_jump','minimum_activity':'activity_only'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    parent=root/'results/minimum_sufficient_dual/threshold';audit=root/'results/minimum_sufficient_dual/threshold_audit';out=root/'results/minimum_sufficient_dual/transfer';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert json.loads((audit/'summary.json').read_text())['passed'];p=json.loads((audit/'protocol.json').read_text());hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    olddir=root/'results/local_dual_jump/transfer';oldrows=json.loads((olddir/'support_rows.json').read_text());oldsummary=json.loads((olddir/'summary.json').read_text());assert sha(olddir/'support_rows.json')==oldsummary['support_rows_sha256']
    for r in oldrows:assert sha(olddir/r['file'])==r['sha256']
    summary=json.loads((parent/'summary.json').read_text());assert sha(parent/'files.json')==summary['files_sha256']
    protocol=dict(source_sha256=hashes,threshold_summary_sha256=sha(parent/'summary.json'),threshold_audit_sha256=sha(audit/'summary.json'),
        old_support_rows_sha256=sha(olddir/'support_rows.json'),new_methods=METHODS,old_methods=transfer.METHODS,states=272,query_targets_accessed=False,
        scope='original incumbents preserved, all old control artifacts retained; no query selection or LP writeback')
    dump(out/'protocol.json',protocol);rows=[];geometry_files=[];begin=time.perf_counter()
    with transfer.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for f in json.loads((parent/'files.json').read_text()):
            seed=f['seed'];source=root/'results/cold_stagnation_switch/development'/f['source_file'];assert sha(source)==f['source_sha256'];assert sha(parent/f['file'])==f['sha256']
            with np.load(source) as a:x=a['x'];v=a['v'];bb=a['b'][16];hh=a['h'][16];inc=a['best'][16]
            cache={}
            for record in json.loads((parent/f['file']).read_text()):
                r=record['restart'];b=bb[r];h=hh[:,r];old=inc[r];oe,om=transfer.base.score(old[None],x,v,np.zeros(4))
                for method,backend in METHODS.items():
                    start=time.perf_counter();arrays,meta=transfer.run(b,h,old,x,v,record['selected'],backend);elapsed=time.perf_counter()-start
                    current=arrays['b'];best=arrays['best'];k=key(x,current)
                    if k not in cache:cache[k]=transfer.base.branch_feasibility(x,v,current)
                    err,mov=transfer.base.score(best[None],x,v,np.zeros(4));curerr=transfer.base.score(current[None],x,v,np.zeros(4))[0][0]
                    path=out/f'{seed}_{r}_{method}.npz';np.savez_compressed(path,x=x,v=v,initial_b=b,initial_h=h,incumbent=old,**arrays)
                    rows.append(dict(seed=seed,restart=r,method=method,backend=backend,event=record['selected'],old_event=record['old_event'],triggered=record['selected'] is not None,file=path.name,sha256=sha(path),
                        parameter_changed=not np.array_equal(current,b),forward_branch_changed=k!=key(x,b),current_error=float(curerr),best_error=float(err[0]),
                        best_improved=bool(transfer.base.better(err,mov,oe,om)[0]),best_feasible=bool(err[0]<=.001001),current_feasible=bool(curerr<=.001001),
                        visited_mode_keys=[k],current_key=k,feasible_mode_keys=[k] if cache[k]['current_branch_feasible'] else [],seconds_atomic_excludes_search_preparation_geometry=elapsed,metadata=meta))
            path=out/f'{seed}_geometry.json';dump(path,cache);geometry_files.append(dict(file=path.name,sha256=sha(path)));print(json.dumps(dict(seed=seed,rows=len(rows),visited_modes=len(cache))),flush=True)
    dump(out/'support_rows.json',rows);dump(out/'geometry_files.json',geometry_files)
    methods={}
    for m in METHODS:
        subset=[r for r in rows if r['method']==m];methods[m]=dict(states=len(subset),**{k:sum(r[k] for r in subset) for k in ['parameter_changed','forward_branch_changed','best_improved','best_feasible','current_feasible']},
            feasible_modes=sum(len(set(k for r in subset if r['seed']==s for k in r['feasible_mode_keys'])) for s in range(5900000,5900016)),seconds_atomic_only=sum(r['seconds_atomic_excludes_search_preparation_geometry'] for r in subset))
    ans=dict(passed=True,states=272,new_method_states=len(rows),unchanged_control_states=len(oldrows),methods=methods,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),support_rows_sha256=sha(out/'support_rows.json'),geometry_files_sha256=sha(out/'geometry_files.json'),query_targets_accessed=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()
