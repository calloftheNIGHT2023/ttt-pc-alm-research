"""Support-only paired/mode coverage analysis; query answers never loaded."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/local_dual_jump/transfer';out=root/'results/local_dual_jump/transfer_analysis';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    rows=json.loads((parent/'support_rows.json').read_text());methods=json.loads((parent/'protocol.json').read_text())['methods'];lookup={(r['seed'],r['restart'],r['method']):r for r in rows}
    old=json.loads((root/'results/cold_stagnation_switch/development/support_rows.json').read_text());paired={};novel=[];task_rows=[]
    for method in methods:
        if method=='dual_jump':continue
        diffs=[r['best_error']-lookup[r['seed'],r['restart'],method]['best_error'] for r in rows if r['method']=='dual_jump']
        paired[method]=dict(lower=sum(t< -1e-12 for t in diffs),same=sum(abs(t)<=1e-12 for t in diffs),higher=sum(t>1e-12 for t in diffs),mean_error_difference=float(np.mean(diffs)))
    for seed in range(5900000,5900016):
        sets={m:set(k for r in rows if r['seed']==seed and r['method']==m for k in r['feasible_mode_keys']) for m in methods}
        rest=set().union(*(sets[m] for m in methods if m!='dual_jump'));historical=set(k for r in old if r['seed']==seed for k in r.get('feasible_modes',[]))
        row=dict(seed=seed,counts={m:len(s) for m,s in sets.items()},dual_unique_vs_activity=sorted(sets['dual_jump']-sets['activity_only']),
            dual_unique_vs_probe=sorted(sets['dual_jump']-sets['branch_probe']),dual_unique_vs_atomic_union=sorted(sets['dual_jump']-rest),
            dual_unique_vs_all_old_cold_histories=sorted(sets['dual_jump']-historical),probe_unique_vs_dual=sorted(sets['branch_probe']-sets['dual_jump']))
        task_rows.append(row)
        for k in row['dual_unique_vs_probe']:
            occurrences=[dict(restart=r['restart'],event=r['event'],current_error=r['current_error'],file=r['file']) for r in rows if r['seed']==seed and r['method']=='dual_jump' and k in r['feasible_mode_keys']]
            novel.append(dict(seed=seed,key=k,also_absent_from_atomic_union=k not in rest,also_absent_from_all_old_histories=k not in historical,occurrences=occurrences))
    dump(out/'task_rows.json',task_rows);dump(out/'dual_novel_modes.json',novel)
    result=dict(passed=True,paired_support_error=paired,mode_totals={k:sum(len(r[k]) for r in task_rows) for k in ['dual_unique_vs_activity','dual_unique_vs_probe','dual_unique_vs_atomic_union','dual_unique_vs_all_old_cold_histories','probe_unique_vs_dual']},
        query_targets_accessed=False,source_sha256=sha(Path(__file__)),support_rows_sha256=sha(parent/'support_rows.json'),
        old_support_rows_sha256=sha(root/'results/cold_stagnation_switch/development/support_rows.json'),output_sha256={p.name:sha(p) for p in out.glob('*.json')})
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
