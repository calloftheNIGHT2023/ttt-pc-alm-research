"""All-state support comparisons; old six witnesses are tracked, not selected."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
from prove_local_dual_parameter_credit import error,pack
from run_local_dual_jump_v2 import dump
from run_multiplier_fixed_point_screen import sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/minimum_sufficient_dual';out=base/'analysis';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    audit=base/'transfer_audit';assert json.loads((audit/'summary.json').read_text())['passed'];hashes=dict(json.loads((audit/'protocol.json').read_text())['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    parent=base/'transfer';old=root/'results/local_dual_jump/transfer';newrows=json.loads((parent/'support_rows.json').read_text());oldrows=json.loads((old/'support_rows.json').read_text());rows=newrows+oldrows;index={(r['seed'],r['restart'],r['method']):r for r in rows};methods=list(dict.fromkeys(r['method'] for r in rows))
    witnesses=root/'results/local_dual_jump/credit_proof/witnesses.json'
    dump(out/'protocol.json',dict(source_sha256=hashes,transfer_audit_sha256=sha(audit/'summary.json'),support_rows_sha256=sha(parent/'support_rows.json'),old_support_rows_sha256=sha(old/'support_rows.json'),old_witnesses_sha256=sha(witnesses),query_targets_accessed=False))
    paired={};primary=[r for r in newrows if r['method']=='minimum_dual']
    for method in methods:
        if method=='minimum_dual':continue
        diff=[r['best_error']-index[r['seed'],r['restart'],method]['best_error'] for r in primary]
        paired[method]=dict(lower=sum(d< -1e-12 for d in diff),same=sum(abs(d)<=1e-12 for d in diff),higher=sum(d>1e-12 for d in diff),mean_error_difference=float(np.mean(diff)))
    taskrows=[]
    for seed in range(5900000,5900016):
        sets={m:set(k for r in rows if r['seed']==seed and r['method']==m for k in r['feasible_mode_keys']) for m in methods};allold=set().union(*(sets[m] for m in methods if m not in ['minimum_dual','minimum_activity']))
        taskrows.append(dict(seed=seed,feasible_modes={m:len(s) for m,s in sets.items()},minimum_dual_unique_vs_old_union=sorted(sets['minimum_dual']-allold),
            minimum_dual_unique_vs_probe=sorted(sets['minimum_dual']-sets['branch_probe']),minimum_dual_unique_vs_old_dual=sorted(sets['minimum_dual']-sets['dual_jump']),
            old_dual_unique_vs_minimum=sorted(sets['dual_jump']-sets['minimum_dual']),minimum_dual_unique_vs_same_activity=sorted(sets['minimum_dual']-sets['minimum_activity']),
            minimum_best_errors={m:min(r['best_error'] for r in rows if r['seed']==seed and r['method']==m) for m in methods}))
    exactrows=[];strict=[];points=0
    for row in primary:
        control=index[row['seed'],row['restart'],'branch_probe'];assert sha(parent/row['file'])==row['sha256'] and sha(old/control['file'])==control['sha256']
        with np.load(parent/row['file']) as a, np.load(old/control['file']) as c:
            x=[F(float(t)) for t in a['x']];v=[F(float(t)) for t in a['v']];ours=error(x,v,[F(float(t)) for t in a['best']]);bank=np.r_[c['incumbent'][None],c['initial_b'][None],c['trial_b']];err=[error(x,v,[F(float(t)) for t in b]) for b in bank];points+=len(bank);gap=min(err)-ours
            rec=dict(seed=row['seed'],restart=row['restart'],dual_error=pack(ours),minimum_probe_error=pack(min(err)),gap=pack(gap),gap_float=float(gap),strict_over_1e12=gap>F(1e-12),probe_points=len(bank));exactrows.append(rec)
            if gap>F(1e-12):strict.append(rec)
    strict.sort(key=lambda r:-r['gap_float']);tracked=[]
    for witness in json.loads(witnesses.read_text()):
        seed,r=witness['seed'],witness['restart'];tracked.append(dict(seed=seed,restart=r,best_errors={m:index[seed,r,m]['best_error'] for m in methods},old_exact_gap=witness['gap_float'],new_exact_gap=next(z['gap_float'] for z in exactrows if (z['seed'],z['restart'])==(seed,r))))
    threshold=json.loads((base/'threshold/rows.json').read_text());ratios=[r['norm_ratio'] for r in threshold if r['selected'] is not None]
    resource=json.loads((base/'short_resources/summary.json').read_text());full=resource['methods']['dual_alm64__full']['mean_seconds'];short=resource['methods']['dual_alm64__short']['mean_seconds'];searchfull=resource['methods']['dual_alm64__full']['mean_search_seconds'];searchshort=resource['methods']['dual_alm64__short']['mean_search_seconds'];cold=resource['methods']['cold__alm16']['mean_seconds']
    dump(out/'task_rows.json',taskrows);dump(out/'exact_probe_pairs.json',exactrows);dump(out/'strict_probe_witnesses.json',strict);dump(out/'tracked_six.json',tracked)
    ans=dict(passed=True,paired_support_error=paired,mode_totals={k:sum(len(r[k]) for r in taskrows) for k in ['minimum_dual_unique_vs_old_union','minimum_dual_unique_vs_probe','minimum_dual_unique_vs_old_dual','old_dual_unique_vs_minimum','minimum_dual_unique_vs_same_activity']},
        old_six_retained=sum(r['new_exact_gap']>1e-12 for r in tracked),strict_probe_witnesses=len(strict),exact_probe_points=points,norm_ratio_quantiles={str(q):float(np.quantile(ratios,q)) for q in [0,.25,.5,.75,1]},
        short_search_complete_time_reduction=1-short/full,short_search_time_reduction=1-searchshort/searchfull,short_vs_cold_time_ratio=short/cold,
        protocol_sha256=sha(out/'protocol.json'),output_sha256={p.name:sha(p) for p in out.glob('*.json')},query_targets_accessed=False,
        scope='all same old development tasks; support error and conditional geometry are not unseen-query risk')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()
