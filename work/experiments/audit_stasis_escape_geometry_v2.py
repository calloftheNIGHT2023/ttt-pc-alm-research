"""318 v2: also audit auxiliary point certificates kept in infeasible records."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import traceback
from audit_stasis_escape_geometry_v1 import inequalities
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    folder=root/'results/primal_stasis_escape/geometry_v1';summary=read(folder/'summary.json');assert summary['passed']
    for name,digest in summary['outputs_sha256'].items():assert sha(folder/name)==digest
    counts=Counter();tasks={};matrices={}
    for item in read(folder/'tasks.json'):
        task=read(folder/item['file']);tasks[task['seed']]=task
        for mode,result in task['geometry'].items():
            a,rhs=inequalities(task['x_observed'],task['v_observed'],mode);matrices[task['seed'],mode]=(a,rhs)
            assert result['classification']=='infeasible' and not result['positive_volume_certified']
            proof=False
            for cert in result['certificates']:
                if cert['type']=='negative_constant_row':
                    i=cert['row_index'];assert not any(a[i]) and rhs[i]==F(cert['rhs'])<0;proof=True
                elif cert['type'] in ['exact_interior_cube','exact_point_check']:
                    point=list(map(F,cert['point']))
                    slacks=[r-sum((z*b for z,b in zip(row,point)),F(0)) for row,r in zip(a,rhs)]
                    assert cert['closed_region_feasible']==all(s>=0 for s in slacks)
                    assert not cert['strict_interior'],'An interior certificate contradicts an infeasible label'
                    counts['auxiliary_point_certificates']+=1
                else:
                    assert cert['type']=='exact_box_separation';weights=[F(0)]*len(rhs)
                    for i,w in cert['weights']:weights[i]=F(w);assert weights[i]>=0
                    coeff=[sum((w*row[j] for w,row in zip(weights,a)),F(0)) for j in range(4)]
                    constant=sum((w*r for w,r in zip(weights,rhs)),F(0))
                    lower=-constant-F(.12)*sum(map(abs,coeff),F(0))
                    assert lower==F(cert['lower']) and coeff==list(map(F,cert['weighted_coefficients']))
                    assert constant==F(cert['weighted_rhs'])
                    if cert['conclusion']=='infeasible':assert lower>0;proof=True
                counts['independent_geometry_certificates']+=1
            assert proof
    pointcounts=Counter()
    for p in read(folder/'points.json'):
        task=tasks[p['seed']];b=list(map(F,p['b']));values=list(map(F,task['x_observed']));codes=[]
        for bias in b:
            zz=[z+bias for z in values];codes.extend(sum(z>=k for k in [F(0),F(1,2),F(1)]) for z in zz)
            values=[max(F(0),min(2*z,2-2*z)) for z in zz]
        assert bytes(codes).hex()==p['mode'];a,rhs=matrices[p['seed'],p['mode']]
        slacks=[r-sum((c*z for c,z in zip(row,b)),F(0)) for row,r in zip(a,rhs)]
        assert min(slacks)==F(p['minimum_slack']) and all(s>=0 for s in slacks)==p['closed_point_feasible']
        assert not p['closed_point_feasible']
        pointcounts[p['method']]+=1;counts['independent_forward_and_point_checks']+=1
    for name,item in summary['aggregate'].items():
        assert item['proposal_points']==pointcounts[name]
        assert item['positive_task_mode_pairs']==item['new_positive_task_mode_pairs']==item['actual_points_feasible']==0
        counts['aggregate_fields']+=4
    final=dict(passed=True,counts=dict(counts),input_sha256=sha(folder/'summary.json'),
               source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'audit_stasis_escape_geometry_v1.py']},
               prior_failure_preserved='results/primal_stasis_escape/geometry_audit_v1/failure.json',query_targets_accessed=False)
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
