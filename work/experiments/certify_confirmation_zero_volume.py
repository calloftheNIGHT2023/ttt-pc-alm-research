"""Exact certificates for unresolved boundary cells, never discard by tolerance.

If lambda>=0, lambda*A=0 and lambda*r=0, every feasible point satisfies
all positively weighted inequalities as equalities. If one weighted row has
a nonzero normal, the cell lies in a hyperplane and has zero 4D volume.
lambda*r<0 instead proves infeasibility. Floating LP proposes weights only;
binary-rational arithmetic is the sole acceptance test.
"""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import enumerate_support_modes as reference
from run_multiplier_fixed_point_screen import sha,dump


def validate(a,r,weights):
    assert all(w>=0 for w in weights)
    coeff=[sum((w*row[j] for w,row in zip(weights,a)),F(0)) for j in range(4)]
    value=sum((w*t for w,t in zip(weights,r)),F(0))
    active=[i for i,w in enumerate(weights) if w>0 and any(a[i])]
    if any(coeff) or value>0 or not active:return None
    return dict(kind='zero_volume' if value==0 else 'infeasible',
        multipliers=[dict(row=i,numerator=str(w.numerator),denominator=str(w.denominator)) for i,w in enumerate(weights) if w],
        weighted_rhs=dict(numerator=str(value.numerator),denominator=str(value.denominator)),
        nonconstant_equality_row=active[0] if value==0 else None)


def certify(a,r):
    # Directly opposed coincident halfspaces give the simplest exact witness.
    for i,row in enumerate(a):
        if not any(row):continue
        pivot=next(j for j,v in enumerate(row) if v)
        for j in range(i+1,len(a)):
            ratio=-a[j][pivot]/row[pivot]
            if ratio>0 and all(v==-ratio*u for u,v in zip(row,a[j])) and r[j]==-ratio*r[i]:
                weights=[F(0)]*len(a);weights[i]=ratio;weights[j]=F(1);answer=validate(a,r,weights)
                assert answer is not None;return {**answer,'proposal':'opposite_exact_rows'}
    af=np.array(a,float);rf=np.array(r,float);norm=np.abs(af).sum(1)
    result=linprog(np.r_[np.zeros(4),-1.],A_ub=np.c_[af,norm],b_ub=rf,bounds=[(None,None)]*5,
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if result.success:
        proposed=np.maximum(-result.ineqlin.marginals,0.)
        for bound in [None,1024,1048576]:
            weights=[F(float(w)) if bound is None else F(float(w)).limit_denominator(bound) for w in proposed]
            answer=validate(a,r,weights)
            if answer is not None:return {**answer,'proposal':'radius_LP_then_exact_validation','denominator_cap':bound}
    return dict(kind='unresolved',lp_status=int(result.status),no_strict_conclusion=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/confirmation_conditional_risk';inp=base/'reference';out=base/'boundary_certificates';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    summary=json.loads((inp/'summary.json').read_text());assert summary['passed'] and not summary['phase_accesses_query_targets'];p=json.loads((inp/'protocol.json').read_text());hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    assert sha(inp/'coverage.json')==summary['coverage_sha256'];coverage=json.loads((inp/'coverage.json').read_text())
    dump(out/'protocol.json',dict(source_sha256=hashes,reference_summary_sha256=sha(inp/'summary.json'),phase_accesses_query_targets=False,
        exact_acceptance='lambda>=0, lambda*A=0, lambda*r<=0, nonzero weighted normal; equality gives measure-zero hyperplane, strict inequality gives infeasibility',
        proposals='opposite exact rows, then fixed LP dual rationalizations with caps none/1024/1048576; no query or risk input'))
    rows=[]
    for task in coverage:
        path=inp/task['file'];assert sha(path)==task['sha256'];data=json.loads(path.read_text());x=np.array(data['x_observed']);v=np.array(data['v_observed'])
        for unknown in data['reference']['final_geometry_unresolved']:
            pattern=np.frombuffer(bytes.fromhex(unknown['pattern']),dtype=np.uint8).reshape(4,4);a,r=reference.constraints(x,v,pattern,True)
            for j in range(4):
                normal=[F(0)]*4;normal[j]=F(1);a.extend([normal,[-t for t in normal]]);r.extend([F(.12),F(.12)])
            result=certify(a,r);rows.append(dict(seed=task['seed'],pattern=unknown['pattern'],original_reason=unknown['reason'],**result))
    dump(out/'certificates.json',rows);ans=dict(passed=True,cells=len(rows),zero_volume=sum(r['kind']=='zero_volume' for r in rows),infeasible=sum(r['kind']=='infeasible' for r in rows),unresolved=sum(r['kind']=='unresolved' for r in rows),
        affected_tasks=len({r['seed'] for r in rows}),protocol_sha256=sha(out/'protocol.json'),certificates_sha256=sha(out/'certificates.json'),phase_accesses_query_targets=False,
        next='independent certificate replay and complete reference accounting before any posterior moments')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
