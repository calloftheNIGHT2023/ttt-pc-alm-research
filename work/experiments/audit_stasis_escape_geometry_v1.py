"""318 independent scalar forward, inequality and separation verification."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def inequalities(x,v,mode):
    codes=list(bytes.fromhex(mode));n=len(x);d=len(codes)//n;a=[];rhs=[]
    for obs in range(n):
        coefficients=[F(0)]*d;constant=F(x[obs])
        for layer in range(d):
            coefficients=coefficients[:];coefficients[layer]+=1;code=codes[layer*n+obs]
            lower=[None,F(0),F(1,2),F(1)][code];upper=[F(0),F(1,2),F(1),None][code]
            if upper is not None:a.append(coefficients[:]);rhs.append(upper-constant)
            if lower is not None:a.append([-z for z in coefficients]);rhs.append(constant-lower)
            slope=[0,2,-2,0][code];coefficients=[slope*z for z in coefficients];constant=slope*constant+[0,0,2,0][code]
        a.extend([coefficients,[-z for z in coefficients]])
        rhs.extend([F(v[obs])+F(.001)-constant,-F(v[obs])+F(.001)+constant])
    for j in range(d):
        unit=[F(k==j) for k in range(d)];a.extend([unit,[-z for z in unit]]);rhs.extend([F(.12),F(.12)])
    return a,rhs


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
                else:
                    assert cert['type']=='exact_box_separation';weights=[F(0)]*len(rhs)
                    for i,w in cert['weights']:weights[i]=F(w);assert weights[i]>=0
                    coeff=[sum((w*row[j] for w,row in zip(weights,a)),F(0)) for j in range(4)]
                    constant=sum((w*r for w,r in zip(weights,rhs)),F(0))
                    lower=-constant-F(.12)*sum(map(abs,coeff),F(0))
                    assert lower==F(cert['lower'])
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
    save(out/'summary.json',dict(passed=True,counts=dict(counts),input_sha256=sha(folder/'summary.json'),
                                source_sha256=sha(Path(__file__)),query_targets_accessed=False))
    print(dict(counts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)
