"""Independent batched forward-pattern enumeration for every saved continuation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import cold_stagnation_switch as cold
from run_multiplier_fixed_point_screen import sha,dump

def modes(x,bank):
    n=len(x);h=np.broadcast_to(x,(len(bank),n));codes=np.empty((len(bank),4*n),dtype=np.uint8)
    for j in range(4):
        z=h+bank[:,j,None];codes[:,j*n:(j+1)*n]=(z>=0).astype(np.uint8)+(z>=.5).astype(np.uint8)+(z>=1).astype(np.uint8)
        h=np.maximum(0.,1.-np.abs(2.*z-1.))
    unique,indices=np.unique(codes,axis=0,return_index=True)
    return {c.tobytes().hex():bank[i] for c,i in zip(unique,indices)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/minimum_sufficient_dual';parent=base/'continuation'
    out=base/'mode_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists();rows=json.loads((parent/'support_rows.json').read_text());counts=Counter();begin=time.perf_counter();tasknotes=[]
    with cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for seed in range(5900000,5900016):
            cache={};saved=json.loads((parent/f'{seed}_geometry.json').read_text())
            for row in [r for r in rows if r['seed']==seed]:
                assert sha(parent/row['file'])==row['sha256']
                with np.load(parent/row['file']) as a:x=a['x'];v=a['v'];bank=np.r_[a['atomic_visited'],a['b'].reshape(-1,4)]
                found=modes(x,bank);assert sorted(found)==row['visited_modes'];counts['actual_parameter_points']+=len(bank);counts['method_pools']+=1
                for key,b in found.items():
                    if key not in cache:
                        cache[key]=cold.base.branch_feasibility(x,v,b);counts['independent_lp_calls']+=1
                        assert cache[key]['current_branch_feasible']==saved[key]['current_branch_feasible'],(seed,key)
                assert sorted(k for k in found if cache[k]['current_branch_feasible'])==row['feasible_modes']
            assert set(cache)==set(saved);tasknotes.append(dict(seed=seed,unique_modes=len(cache),feasible_modes=sum(bool(r['current_branch_feasible']) for r in cache.values())))
            print(json.dumps(dict(seed=seed,counts=counts)),flush=True)
    dump(out/'tasks.json',tasknotes)
    result=dict(passed=True,counts=counts,seconds=time.perf_counter()-begin,query_targets_accessed=False,source_sha256=sha(Path(__file__)),support_rows_sha256=sha(parent/'support_rows.json'),tasks_sha256=sha(out/'tasks.json'))
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
