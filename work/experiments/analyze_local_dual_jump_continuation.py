"""All-task paired continuation outcomes and preservation of six atomic cases."""
import argparse
import json
from pathlib import Path
import numpy as np
import cold_stagnation_switch as cold
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/local_dual_jump';parent=base/'continuation'
    out=base/'continuation_analysis';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    rows=json.loads((parent/'support_rows.json').read_text());lookup={(r['seed'],r['method']):r for r in rows};methods=[c['name'] for c in json.loads((parent/'protocol.json').read_text())['configs']]
    tasks=[];restart_rows=[]
    for seed in range(5900000,5900016):
        sets={m:set(lookup[seed,m]['feasible_modes']) for m in methods};dual=sets['dual_alm64'];other=set().union(*(s for m,s in sets.items() if m!='dual_alm64'))
        tasks.append(dict(seed=seed,dual_unique_vs_activity=sorted(dual-sets['activity_alm64']),dual_unique_vs_probe=sorted(dual-sets['probe_alm64']),
            dual_unique_vs_probe_all=sorted(dual-sets['probe_all_alm64']),dual_unique_vs_other_union=sorted(dual-other),
            dual_unique_vs_bp_union=sorted(dual-set().union(*(sets[m] for m in ['adam60','adam240','gn20','gn40'])))))
        for m in methods:
            a=np.load(parent/lookup[seed,m]['file']);bank=a['best_bank'];origin=a['origin']
            for r in range(17):
                subset=bank[origin==r];_,best=cold.select(subset,a['x'],a['v']);error=float(cold.base.score(best[None],a['x'],a['v'],np.zeros(4))[0][0])
                restart_rows.append(dict(seed=seed,restart=r,method=m,error=error,feasible=error<=.001001,b=best.tolist()))
    by_restart={(r['seed'],r['restart'],r['method']):r for r in restart_rows};paired={}
    for m in methods:
        if m=='dual_alm64':continue
        diffs=[lookup[s,'dual_alm64']['support_max_error']-lookup[s,m]['support_max_error'] for s in range(5900000,5900016)]
        paired[m]=dict(task_lower=sum(t< -1e-12 for t in diffs),task_same=sum(abs(t)<=1e-12 for t in diffs),task_higher=sum(t>1e-12 for t in diffs),
            primary_feasible_control_not=[s for s in range(5900000,5900016) if lookup[s,'dual_alm64']['support_feasible'] and not lookup[s,m]['support_feasible']],
            control_feasible_primary_not=[s for s in range(5900000,5900016) if not lookup[s,'dual_alm64']['support_feasible'] and lookup[s,m]['support_feasible']])
    witness_rows=[]
    for w in json.loads((base/'credit_proof/witnesses.json').read_text()):
        witness_rows.append(dict(seed=w['seed'],restart=w['restart'],atomic_gap=w['gap_float'],continuation={m:by_restart[w['seed'],w['restart'],m] for m in methods}))
    dump(out/'task_modes.json',tasks);dump(out/'restart_rows.json',restart_rows);dump(out/'six_atomic_cases.json',witness_rows)
    result=dict(passed=True,paired=paired,mode_totals={k:sum(len(r[k]) for r in tasks) for k in tasks[0] if k!='seed'},query_targets_accessed=False,
        source_sha256=sha(Path(__file__)),support_rows_sha256=sha(parent/'support_rows.json'),output_sha256={f.name:sha(f) for f in out.glob('*.json')})
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
