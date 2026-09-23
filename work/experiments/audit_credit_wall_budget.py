"""Replay recorded stopping work; independently certify all accepted outputs.

Repeated identical certificates reuse a cache keyed by the complete credit and
weights bytes, seed, family and region. Counts distinguish instances and unique
proofs. This does not re-measure the original real clock.
"""
import argparse
from collections import Counter
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import credit_wall_budget as model
import optimized_branch_dual as cert
import neighbor_mode_memory as neighbor
from audit_joint_credit_minimax import exact_mixture,rational_optimum
from run_credit_wall_budget import load_inputs,METHODS,BUDGETS


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/credit_wall_budget';inp=base/'development'
    p=read(inp/'protocol.json');s=read(inp/'summary.json');rows=read(inp/'rows.json');assert s['execution_complete'] and len(rows)==s['runs']==1152
    assert s['protocol_sha256']==sha(inp/'protocol.json') and s['rows_sha256']==sha(inp/'rows.json')
    assert p['design_sha256']==sha(root/'outputs/ttt-pc-alm-research/233_credit_wall_budget_protocol.md') and p['primitive_sha256']==sha(base/'primitive/summary.json')
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    inputs,hashes=load_inputs(root);assert json.loads(json.dumps(hashes))==p['input_hashes']
    checks=dict(runs=0,replayed_arrays=0,old_certificate_instances=0,unique_old_certificates=0,new_certificate_instances=0,unique_new_certificates=0,
        exact_convex_unique=0,exact_rounding_unique=0,unique_parameter_infeasible=0,late_old=0,late_new=0,clock_events=0,cooperative_stops=0,step_caps=0)
    old_cache={};new_cache={};lp_cache={};proofrows=[];runrows=[]
    for ri,row in enumerate(rows):
        seed=row['seed'];method=row['method'];x,v,regs,banks=inputs[seed];bank=banks[method];budget=row['budget_seconds']
        assert sha(inp/row['arrays_file'])==row['arrays_sha256'] and sha(inp/row['meta_file'])==row['meta_sha256']
        with np.load(inp/row['arrays_file']) as z:arrays={key:z[key] for key in z.files}
        meta=read(inp/row['meta_file']);assert meta['time_source']=='wall' and meta['budget_seconds']==budget
        assert all(a['seconds']<=b['seconds'] for a,b in zip(meta['events'],meta['events'][1:]))
        assert row['total_seconds']==meta['events'][-1]['seconds']==meta['total_seconds'] and row['external_guard_seconds']>=row['total_seconds']
        assert row['overrun_seconds']==max(0.,row['total_seconds']-budget)
        again,am=model.guarded_solve(x,v,regs,bank,budget=budget,max_steps=4096,replay_events=meta['events'])
        for key,value in arrays.items():assert value.tobytes()==again[key].tobytes(),(ri,key);checks['replayed_arrays']+=1
        for key,value in meta.items():
            if key!='time_source':assert value==am[key],(ri,key)
        assert am['time_source']=='replay';indices=arrays['indices'];assert np.array_equal(indices,np.flatnonzero(~arrays['old_positive']))
        assert meta['new_count']==int(np.sum(arrays['first_step']>0)) and np.array_equal(arrays['accepted'][indices],arrays['first_step']>0)
        assert meta['total_positive']==int(arrays['accepted'].sum())==meta['old_count']+meta['new_count']
        patterns=[reg.tobytes().hex() for reg in regs[arrays['accepted']]];assert patterns==row['accepted_patterns']
        lookup={reg.tobytes().hex():i for i,reg in enumerate(regs)}
        if meta['old_count']:assert next(e for e in meta['events'] if e['name']=='old_after')['seconds']<=budget
        assert all(pr['completed_seconds']<=budget for pr in meta['proofs']) and all(pr['completed_seconds']>budget for pr in meta['late_proofs'])
        assert meta['late_old_count']==len(meta['late_old_proofs']) and meta['late_new_count']==len(meta['late_proofs'])
        if meta['late_old_count']:assert meta['old_count']==meta['new_count']==0 and meta['stop_event']=='old_after'
        assert {pr['pattern'] for pr in meta['old_proofs']}=={regs[i].tobytes().hex() for i in np.flatnonzero(arrays['old_positive'])}
        for pr in meta['old_proofs']:
            index=lookup[pr['pattern']];key=(seed,method,index,pr['direction']);checks['old_certificate_instances']+=1
            if key not in old_cache:
                exact=cert.exact_optimum(x,v,regs[index],bank[pr['direction']]);assert exact['positive'];old_cache[key]=True;checks['unique_old_certificates']+=1
        newkeys=[]
        for pr in meta['proofs']:
            q=pr['index'];index=int(indices[q]);reg=regs[index];credit=arrays['credit'][q];weights=arrays['weights'][q]
            assert pr['pattern']==reg.tobytes().hex() and pr['step']==int(arrays['first_step'][q])
            assert np.all(weights>=0) and abs(weights.sum()-1)<1e-12
            key=(seed,method,index,hashlib.sha256(credit.tobytes()+weights.tobytes()).hexdigest());checks['new_certificate_instances']+=1
            if key not in new_cache:
                exact=cert.exact_optimum(x,v,reg,credit);assert exact==pr['exact'] and exact['positive'];checks['unique_new_certificates']+=1
                ex=exact_mixture(bank,weights);value=rational_optimum(x,v,reg,ex);assert value>0;checks['exact_convex_unique']+=1
                error=sum(abs(ex[j][i]-F(float(credit[j,i]))) for j in range(reg.shape[0]) for i in range(reg.shape[1]))
                lower=F(int(exact['numerator']),int(exact['denominator']))-error;assert 0<lower<=value;checks['exact_rounding_unique']+=1
                new_cache[key]=exact;proofrows.append(dict(seed=seed,method=method,index=index,credit_weights_sha256=key[-1],
                    numerator=str(value.numerator),denominator=str(value.denominator),rounding_l1=float(error),lower=float(lower)))
            else:assert new_cache[key]==pr['exact']
            newkeys.append(list(key))
        for index in np.flatnonzero(arrays['accepted']):
            key=(seed,int(index))
            if key not in lp_cache:
                _,_,mat,rhs=neighbor.pattern_matrix(x,v,regs[index])
                lp=linprog(np.zeros(regs.shape[1]),A_ub=mat,b_ub=rhs,bounds=[(-.12,.12)]*regs.shape[1],options={'primal_feasibility_tolerance':1e-9})
                assert lp.status==2;lp_cache[key]=True;checks['unique_parameter_infeasible']+=1
        checks['late_old']+=meta['late_old_count'];checks['late_new']+=meta['late_new_count'];checks['clock_events']+=len(meta['events'])
        checks['cooperative_stops']+=meta['stop_reason']=='deadline';checks['step_caps']+=meta['stop_reason']=='step_cap';checks['runs']+=1
        runrows.append(dict(run_index=ri,seed=seed,method=method,budget_seconds=budget,repeat=row['repeat'],new_proof_keys=newkeys,
            oracle_pairs=meta['oracle_pairs'],response_batches=meta['response_batches'],updates=meta['update_batches'],clock_events=len(meta['events'])))
        if (ri+1)%9==0:print(json.dumps(dict(runs=ri+1,unique_new=checks['unique_new_certificates'],new_instances=checks['new_certificate_instances'])),flush=True)
    lookup={(r['seed'],r['method'],r['budget_seconds'],r['repeat']):r for r in rows};assert len(lookup)==1152
    seeds=p['seeds'];rng=np.random.default_rng(5900001);boot=rng.integers(0,len(seeds),(20000,len(seeds)))
    for block in s['summaries']:
        budget=block['budget_seconds'];means={m:np.array([np.mean([lookup[seed,m,budget,rep]['total_positive'] for rep in range(2)]) for seed in seeds]) for m in METHODS}
        for row in block['summaries']:
            method=row['method'];rr=[r for r in rows if r['method']==method and r['budget_seconds']==budget]
            for k,source in [('count_mean_over_repeats','total_positive'),('old_count_mean_over_repeats','old_count'),('new_count_mean_over_repeats','new_count')]:assert row[k]==sum(r[source] for r in rr)/2
            for k,source in [('mean_actual_seconds','total_seconds'),('mean_external_seconds','external_guard_seconds'),('overrun_mean','overrun_seconds'),('mean_response_batches','response_batches')]:assert row[k]==float(np.mean([r[source] for r in rr]))
            ov=np.array([r['overrun_seconds'] for r in rr]);assert row['overrun_p95']==float(np.quantile(ov,.95)) and row['overrun_max']==float(ov.max())
            ext=np.maximum(np.array([r['external_guard_seconds'] for r in rr])-budget,0);assert row['external_overrun_mean']==float(ext.mean()) and row['external_overrun_max']==float(ext.max())
            assert row['stop_reasons']==dict(Counter(r['stop_reason'] for r in rr)) and row['stop_events']==dict(Counter(str(r['stop_event']) for r in rr))
            assert row['late_old']==sum(r['late_old_count'] for r in rr) and row['late_new']==sum(r['late_new_count'] for r in rr)
        for row in block['comparisons']:
            method=row['comparator'];diff=means['alm_native']-means[method];assert row['native_minus_comparator_mean_per_task']==float(diff.mean())
            assert np.array_equal(row['paired_bootstrap95'],np.quantile(diff[boot].mean(1),[.025,.975]))
            common=aonly=bonly=0
            for seed in seeds:
                for rep in range(2):
                    aa=set(lookup[seed,'alm_native',budget,rep]['accepted_patterns']);bb=set(lookup[seed,method,budget,rep]['accepted_patterns'])
                    common+=len(aa&bb);aonly+=len(aa-bb);bonly+=len(bb-aa)
            assert row['common_mean_over_repeats']==common/2 and row['only_native_mean_over_repeats']==aonly/2 and row['only_comparator_mean_over_repeats']==bonly/2
    out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'unique_proofs.json').write_text(json.dumps(proofrows,indent=2),encoding='utf-8');(out/'run_checks.json').write_text(json.dumps(runrows,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=checks,summary_blocks_recomputed=4,source_sha256=sha(Path(__file__)),
        input_sha256={name:sha(inp/name) for name in ['protocol.json','summary.json','rows.json']},unique_proofs_sha256=sha(out/'unique_proofs.json'),run_checks_sha256=sha(out/'run_checks.json'),
        scope='Deterministic original-work replay, exact certificate safety and statistics checked; original wall timestamps not independently re-measured')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
