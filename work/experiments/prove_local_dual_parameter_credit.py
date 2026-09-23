"""Exact forward-error checks for every support-selected atomic credit witness.

Comparators are the explicit finite zero-dual branch probes, not all possible
zero-dual algorithms or all continuous activity perturbations.
"""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
from audit_local_dual_jump import g,branch
from run_multiplier_fixed_point_screen import sha,dump

def pack(x):return [str(x.numerator),str(x.denominator)]
def error(x,v,b):
    values=list(x)
    for bias in b:values=[g(t+bias) for t in values]
    return max(abs(y-z) for y,z in zip(values,v))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();parent=root/'results/local_dual_jump/transfer'
    out=root/'results/local_dual_jump/credit_proof';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    rows=json.loads((parent/'support_rows.json').read_text());index={(r['seed'],r['restart'],r['method']):r for r in rows};proofs=[];all_pairs=[]
    for row in rows:
        if row['method']!='dual_jump':continue
        control=index[row['seed'],row['restart'],'branch_probe'];a=np.load(parent/row['file']);c=np.load(parent/control['file'])
        x=list(map(lambda t:F(float(t)),a['x']));v=list(map(lambda t:F(float(t)),a['v']))
        dual_error=error(x,v,list(map(lambda t:F(float(t)),a['best'])))
        probe_errors=[error(x,v,list(map(lambda t:F(float(t)),b))) for b in np.r_[c['incumbent'][None],c['initial_b'][None],c['trial_b']]]
        minimum=min(probe_errors);strict=dual_error<minimum
        assert strict==bool(row['best_error']<control['best_error']-1e-12) or abs(float(dual_error-minimum))<1e-12
        all_pairs.append(dict(seed=row['seed'],restart=row['restart'],strict_exact=strict,strict_over_1e12=minimum-dual_error>F(1e-12),gap=pack(minimum-dual_error)))
        if minimum-dual_error<=F(1e-12):continue
        other=index[row['seed'],row['restart'],'activity_only'];ao=np.load(parent/other['file']);assert a['h'].tobytes()==ao['h'].tobytes()
        layers=[]
        for j in range(4):
            prev=a['x'] if j==0 else a['h'][j-1];prev=[F(float(t)) for t in prev]
            bc,ba=F(float(a['b'][j])),F(float(ao['b'][j]));pc=[branch(t+bc) for t in prev];pa=[branch(t+ba) for t in prev]
            common=pc==pa;record=dict(layer=j,same_bias_piece=common,actual_delta=pack(bc-ba),dual_pattern=pc,activity_pattern=pa)
            if common:
                slopes=[[0,2,-2,0][k] for k in pc];target_c=[F(float(t)) for t in a['h'][j]+a['u'][j]];target_a=[F(float(t)) for t in ao['h'][j]]
                shift=sum((s*(yc-ya) for s,yc,ya in zip(slopes,target_c,target_a)),F(0))/(sum(s*s for s in slopes)+len(x)*F(.01))
                interior=all(t+bb not in [F(0),F(.5),F(1)] for bb in [bc,ba] for t in prev) and max(abs(bc),abs(ba))<F(.12)
                record.update(unconstrained_target_shift=pack(shift),both_strictly_interior=interior,shift_error=pack(bc-ba-shift))
                if interior:assert abs(bc-ba-shift)<F(1e-12)
            layers.append(record)
        proofs.append(dict(seed=row['seed'],restart=row['restart'],selection='strict exact support error lower than every stored zero-dual branch probe and old incumbent by >1e-12',
            dual_error=pack(dual_error),best_probe_error=pack(minimum),gap=pack(minimum-dual_error),gap_float=float(minimum-dual_error),
            probe_points=len(probe_errors),dual_b=a['b'].tolist(),activity_only_b=ao['b'].tolist(),same_activity_exact=True,layers=layers,
            files={r['file']:r['sha256'] for r in [row,control,other]},event=row['event']))
    proofs.sort(key=lambda r:(-r['gap_float'],r['seed'],r['restart']))
    dump(out/'all_pairs.json',all_pairs);dump(out/'witnesses.json',proofs)
    result=dict(passed=True,pairs=len(all_pairs),strict_witnesses_over_1e12=len(proofs),exact_forward_points=sum(p['probe_points'] for p in proofs),
        strongest_support_selected=proofs[0] if proofs else None,query_targets_accessed=False,source_sha256=sha(Path(__file__)),
        support_rows_sha256=sha(parent/'support_rows.json'),output_sha256={p.name:sha(p) for p in out.glob('*.json')})
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
