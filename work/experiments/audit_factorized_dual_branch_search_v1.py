"""320 independent all-breakpoint dual values and scalar shell minima.

Does not claim exhaustive K-order auditing on the 4^16 real problem;
K-order correctness is separately tested on fully enumerated small cases.
"""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import time
import traceback
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha
from audit_post_escape_geometry_v1 import certificates
from audit_stasis_escape_geometry_v1 import inequalities

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']
BOUND=F(.12);EPS=F(.001);S=[0,2,-2,0];C=[0,0,2,0]
ROWS=list(product(range(4),repeat=4))


def independent_layer(x,previous,a):
    low=list(map(F,x)) if x is not None else [F(0)]*4
    high=low if x is not None else [F(1)]*4
    intervals=[(-BOUND,F(0)),(F(0),F(1,2)),(F(1,2),F(1)),(F(1),1+BOUND)]
    answer={}
    for row in ROWS:
        zlo=[intervals[k][0] for k in row];zhi=[intervals[k][1] for k in row]
        left=max([-BOUND]+[z-h for z,h in zip(zlo,high)])
        right=min([BOUND]+[z-h for z,h in zip(zhi,low)])
        if left>right:answer[row]=None;continue
        # Both kinds of clamp breakpoint, not the generator's sign-selected set.
        choices={left,right}
        for i in range(4):
            for point in [zlo[i]-low[i],zhi[i]-high[i]]:
                if left<=point<=right:choices.add(point)
        slope=[S[r]*v for r,v in zip(row,a)];coef=[p-q for p,q in zip(previous,slope)]
        offset=-sum((v*C[r] for v,r in zip(a,row)),F(0));values=[]
        for bias in choices:
            hlow=[max(l,z-bias) for l,z in zip(low,zlo)]
            hhigh=[min(h,z-bias) for h,z in zip(high,zhi)]
            values.append(offset+sum((min(c*l,c*h)-s*bias for c,l,h,s in zip(coef,hlow,hhigh,slope)),F(0)))
        answer[row]=min(values)
    return answer


def audit_proposals(root,source,counts):
    cache={};records=[];start=time.perf_counter()
    summary=read(source/'summary.json');manifest=read(source/'before_geometry_manifest.json')
    assert summary['passed'] and sha(source/'before_geometry_manifest.json')==summary['manifest_sha256']
    for name,digest in manifest['files_sha256'].items():assert sha(source/name)==digest
    aggregate={n:Counter() for n in CHANNELS}
    for index,row in enumerate(read(source/'rows.json')):
        rec=read(source/row['file']);x=np.array(rec['x_observed']);v=np.array(rec['v_observed'])
        b,h,u=map(np.array,[rec['b'],rec['h'],rec['u']]);pred=x.copy();original=[];slopes=[];residual=[];previous=x
        for j in range(4):
            zz=pred+b[j];original.extend(int(z>=0)+int(z>=.5)+int(z>=1) for z in zz)
            slopes.append(np.where((zz>0)&(zz<.5),2.,np.where((zz>.5)&(zz<1),-2.,0.)))
            pred=np.maximum(0,1-np.abs(2*zz-1))
            residual.append(h[j]-np.maximum(0,1-np.abs(2*(previous+b[j])-1)));previous=h[j]
        assert bytes(original).hex()==rec['original_mode'];original=np.array(original).reshape(4,4)
        residual=np.array(residual);alpha=np.zeros((4,4));raw=pred-v
        alpha[-1]=-np.sign(raw)*np.maximum(abs(raw)-.001,0)
        for j in [2,1,0]:alpha[j]=slopes[j+1]*alpha[j+1]
        rng=np.random.default_rng(np.random.SeedSequence([320071,rec['seed'],*rec['location']]))
        expected=dict(dual=u,dual_plus_residual=u+residual,residual=residual,bp=alpha,
                      random_sign=rng.choice([-1.,1.],size=(4,4)),zero=np.zeros((4,4)))
        with np.load(root/'results/solver_policy_recurrence/development_v1'/rec['source_file'],allow_pickle=False) as z:
            i=rec['source_index'];assert np.array_equal(b,z['b'][0,i]) and np.array_equal(h,z['h'][0,:,i]) and np.array_equal(u,z['u'][0,:,i])
        counts['source_state_arrays']+=3
        old=read(root/'results/post_escape_continuation/geometry_v1'/f"{rec['seed']}_geometry.json")
        for name in CHANNELS:
            assert np.array_equal(expected[name],rec['credits'][name]);counts['credit_arrays']+=1
            aa=[[F(float(q)) for q in r] for r in expected[name]];tables=[]
            for j in range(4):
                xp=tuple(x) if j==0 else None;prev=tuple([F(0)]*4 if j==0 else aa[j-1]);key=(xp,prev,tuple(aa[j]))
                if key not in cache:cache[key]=independent_layer(xp,prev,aa[j]);counts['independent_layer_tables']+=1
                tables.append(cache[key])
            terminal=sum((min(a*max(F(0),F(float(y))-EPS),a*min(F(1),F(float(y))+EPS)) for a,y in zip(aa[-1],v)),F(0))
            # Independent scalar min-plus DP in reverse layer order; no K pruning.
            dp={0:F(0)}
            for j in [3,2,1,0]:
                best={}
                for r,value in tables[j].items():
                    if value is None:continue
                    m=sum(a!=b for a,b in zip(r,original[j]));best[m]=min(value,best.get(m,value))
                updated={}
                for m,a in dp.items():
                    for k,value in best.items():
                        result=a+value;updated[m+k]=min(result,updated.get(m+k,result))
                dp=updated
            result=rec['results'][name];assert {str(m):str(value+terminal) for m,value in sorted(dp.items())}==result['shell_minima']
            counts['global_shell_minima']+=len(dp)
            current=terminal+sum((tables[j][tuple(original[j])] for j in range(4)),F(0))
            assert current==F(result['current_lower']) and (current>0)==result['current_certified_infeasible']
            admissible=sorted(m for m,value in dp.items() if m>=1 and value+terminal<=0)
            minimum=admissible[0] if admissible else None
            assert minimum==result['minimum_nonexcluded_hamming']
            assert (minimum if current>0 else None)==result['necessary_hamming_lower_bound']
            if current>0:
                assert old['geometry'][rec['original_mode']]['classification']=='infeasible';counts['independent_current_infeasibility']+=1
            order=defaultdict(list);seen=set()
            for p in result['proposals']:
                pattern=np.array(list(bytes.fromhex(p['mode']))).reshape(4,4);assert p['mode'] not in seen;seen.add(p['mode'])
                m=int(np.sum(pattern!=original));assert m==p['hamming'] and minimum<=m<=minimum+2
                value=terminal+sum((tables[j][tuple(pattern[j])] for j in range(4)),F(0))
                assert value==F(p['lower'])<=0;order[m].append((value,tuple(pattern.ravel())))
                assert len(order[m])==p['rank']+1;counts['proposal_lower_bounds']+=1
            for ordered in order.values():assert ordered==sorted(ordered) and len(ordered)<=8
            counts['current_lower_bounds']+=1
            aggregate[name]['cases']+=1;aggregate[name]['current_certified_infeasible']+=int(current>0)
            aggregate[name]['proposals']+=len(result['proposals'])
        records.append(rec)
        if (index+1)%10==0:print(dict(audited_states=index+1,seconds=time.perf_counter()-start),flush=True)
    for n,fields in aggregate.items():
        for key,value in fields.items():assert summary['aggregate'][n][key]==value;counts['aggregate_fields']+=1
    return records


def run(root,out):
    begin=time.perf_counter();source=root/'results/factorized_dual_branch_search/development_v1';counts=Counter()
    records=audit_proposals(root,source,counts)
    folder=root/'results/factorized_dual_branch_search/geometry_v1';summary=read(folder/'summary.json');assert summary['passed']
    for name,digest in summary['outputs_sha256'].items():assert sha(folder/name)==digest
    byseed=defaultdict(list)
    for rec in records:byseed[rec['seed']].append(rec)
    tasks=[]
    for row in read(folder/'tasks.json'):
        task=read(folder/row['file']);props={n:set() for n in CHANNELS};positive={}
        for rec in byseed[task['seed']]:
            for n in CHANNELS:props[n].update(p['mode'] for p in rec['results'][n]['proposals'])
        for mode,result in task['geometry'].items():
            a,rhs=inequalities(np.array(task['x_observed']),np.array(task['v_observed']),mode)
            counts['independent_geometry_certificates']+=certificates(a,rhs,result)
        prior=set(task['prior_positive_modes'])
        for n in CHANNELS:
            assert sorted(props[n])==task['methods'][n]['proposal_modes']
            positive[n]={m for m in props[n] if task['geometry'][m]['positive_volume_certified']}
            assert sorted(positive[n])==task['methods'][n]['positive_modes'];counts['method_pools']+=1
        for n in CHANNELS:
            other=set().union(*(positive[k] for k in CHANNELS if k!=n))
            assert task['comparisons'][n]==dict(new_vs_prior=sorted(positive[n]-prior),new_vs_prior_and_other_channels=sorted(positive[n]-prior-other))
            counts['comparison_sets']+=2
        expected=(positive['dual']|positive['dual_plus_residual'])-prior-set().union(*(positive[n] for n in ['residual','bp','random_sign','zero']))
        assert sorted(expected)==task['combined_candidate_exclusive'];tasks.append(task)
    for n,values in summary['aggregate'].items():
        expected=dict(proposal_pairs=sum(len(t['methods'][n]['proposal_modes']) for t in tasks),
                      positive_pairs=sum(len(t['methods'][n]['positive_modes']) for t in tasks),
                      new_vs_prior=sum(len(t['comparisons'][n]['new_vs_prior']) for t in tasks),
                      new_vs_prior_and_other_channels=sum(len(t['comparisons'][n]['new_vs_prior_and_other_channels']) for t in tasks),
                      tasks_with_new_positive=sum(bool(t['comparisons'][n]['new_vs_prior']) for t in tasks))
        assert expected==values;counts['aggregate_fields']+=len(values)
    assert sum(len(t['combined_candidate_exclusive']) for t in tasks)==summary['combined_candidate_exclusive_pairs']
    result=dict(passed=True,counts=dict(counts),seconds=time.perf_counter()-begin,
                source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'audit_post_escape_geometry_v1.py','audit_stasis_escape_geometry_v1.py']},
                input_sha256={str(p.relative_to(root)):sha(p) for p in [source/'summary.json',folder/'summary.json']},
                scope='All credit inputs, scalar global shell minima, proposed dual values, geometry and sets. K-order exhaustive tests are separate small cases.',
                query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
