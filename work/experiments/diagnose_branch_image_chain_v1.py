"""321 full-pool exact structural image-gap diagnosis, no candidate mutation."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def conflicts(x,v,pattern):
    n=len(x);depth=len(pattern);bound=F(.12);eps=F(.001)
    zlo=[-bound,F(0),F(1,2),F(1)];zhi=[F(0),F(1,2),F(1),1+bound]
    result=[]
    for j,row in enumerate(pattern):
        low=list(map(F,x)) if j==0 else [F(0)]*n
        high=low if j==0 else [F(0) if q in [0,3] else F(1) for q in pattern[j-1]]
        left=max([-bound]+[zlo[r]-h for r,h in zip(row,high)])
        right=min([bound]+[zhi[r]-h for r,h in zip(row,low)])
        if left>right:result.append(dict(type='empty_shared_bias',layer=j,lower=str(left),upper=str(right)))
    for i,row in enumerate(pattern[-1]):
        if row in [0,3] and abs(F(v[i]))>eps:
            result.append(dict(type='flat_output_observation_conflict',observation=i,observed=str(F(v[i])),eps=str(eps)))
    return result


def run(root,out):
    begin=time.perf_counter();folder=root/'results/factorized_dual_branch_search/geometry_v1'
    summary=read(folder/'summary.json');assert summary['passed']
    for name,digest in summary['outputs_sha256'].items():assert sha(folder/name)==digest
    assert conflicts([.1],[.5],[[0],[2]])
    assert not conflicts([.45],[.04],[[1],[2]])
    assert conflicts([.01],[.7],[[0]])
    counts=Counter();aggregate={n:Counter() for n in summary['aggregate']};records=[]
    for row in read(folder/'tasks.json'):
        task=read(folder/row['file']);n=len(task['x_observed']);rejected=set();checks={}
        for mode,result in task['geometry'].items():
            flat=list(bytes.fromhex(mode));pattern=[flat[j*n:(j+1)*n] for j in range(4)]
            certificate=conflicts(task['x_observed'],task['v_observed'],pattern)
            counts['modes']+=1
            if certificate:
                assert result['classification']=='infeasible';rejected.add(mode);counts['structurally_excluded']+=1
                for kind in set(c['type'] for c in certificate):counts[kind]+=1
            else:
                counts['structural_survivors']+=1;counts['survivors_'+result['classification']]+=1
            checks[mode]=certificate
        for name,method in task['methods'].items():
            proposals=set(method['proposal_modes']);positive=set(method['positive_modes'])
            assert not (positive&rejected)
            aggregate[name].update(dict(proposals=len(proposals),excluded=len(proposals&rejected),
                                        survivors=len(proposals-rejected),positive_preserved=len(positive)))
        records.append(dict(seed=task['seed'],certificates=checks))
    save(out/'certificates.json',records);save(out/'aggregate.json',aggregate)
    design=root/'outputs/ttt-pc-alm-research/321_branch_image_chain_design_v1.md'
    result=dict(passed=True,counts=dict(counts),aggregate=aggregate,seconds=time.perf_counter()-begin,
                input_sha256=sha(folder/'summary.json'),source_sha256=sha(Path(__file__)),design_sha256=sha(design),
                outputs_sha256={n:sha(out/n) for n in ['certificates.json','aggregate.json']},
                candidate_generation_performed=False,query_targets_accessed=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
