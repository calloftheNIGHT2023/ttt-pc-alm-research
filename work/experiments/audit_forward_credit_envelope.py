"""Rebuild all envelopes; independent reachability LPs and exact certificates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import forward_credit_envelope as model
from verify_forward_credit_envelope import crosscheck_intervals
from verify_exact_credit_hull import validate
from audit_baseline_history import direct_positive
from run_forward_credit_envelope import summarize
from run_credit_wall_budget import load_inputs
import neighbor_mode_memory as neighbor

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/forward_credit_envelope';inp=base/'development';src=Path(__file__).parent
    p=read(inp/'protocol.json');s=read(inp/'summary.json');assert s['execution_complete'];records=read(inp/'banks.json');inputs,hashes=load_inputs(root)
    for name,value in p['source_sha256'].items():assert sha(src/name)==value,name
    assert json.loads(json.dumps(hashes))==p['input_hashes'];counts=Counter();crosschecks=[];checks=[];parameters=set();lookup={}
    for seed,(x,v,regs,_) in inputs.items():
        banks,meta,details=model.build(x,v);cross=crosscheck_intervals(x);crosschecks.append(dict(seed=seed,**cross));counts.update(cross['checks'])
        for method in model.METHODS:
            rec=next(r for r in records if r['seed']==seed and r['method']==method);bank=banks[method]
            assert sha(inp/rec['bank_file'])==rec['bank_file_sha256'] and sha(inp/rec['intervals_file'])==rec['intervals_sha256'] and sha(inp/rec['rows_file'])==rec['rows_sha256']
            with np.load(inp/rec['bank_file']) as z:assert z[method].shape==bank.shape and z[method].dtype==bank.dtype and z[method].tobytes()==bank.tobytes();counts['bank_arrays']+=1
            assert hashlib.sha256(bank.tobytes()).hexdigest()==rec['bank_sha256'] and read(inp/rec['intervals_file'])==details
            for key in ['directions','bank_bytes','reachable_single_observation_patterns','signed_patterns','eta','scope']:assert meta[method][key]==rec[key]
            rows=read(inp/rec['rows_file']);assert len(rows)==len(regs);raw=Counter();effective=Counter();statuses={}
            for i,(row,reg) in enumerate(zip(rows,regs)):
                key=reg.tobytes().hex();assert row['index']==i and row['pattern']==key;r=row['result'];verified=validate(x,v,reg,bank,r);raw[r['status']]+=1;status=r['status']
                counts['raw_'+status]+=1
                if status=='positive':direct_positive(x,v,reg,bank,r['weights'],r['credit'],r['dual']['proof']);counts['exact_rounding']+=1
                if status=='nonpositive':counts['fraction_primal_directions']+=len(bank)
                if row['inherited'] is not None:
                    assert status=='unknown';q=row['inherited'];direct_positive(x,v,reg,bank,q['weights'],q['credit'],q['proof']);status='positive';counts['inherited_positive']+=1
                assert status==row['effective_status'];effective[status]+=1;statuses[key]=status;lookup[seed,method,key]=status;counts['regions']+=1
                if status=='positive' and (seed,key) not in parameters:
                    _,_,matrix,rhs=neighbor.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=matrix,b_ub=rhs,bounds=[(-.12,.12)]*4,options=dict(primal_feasibility_tolerance=1e-9));assert lp.status==2;parameters.add((seed,key))
                checks.append(dict(seed=seed,method=method,index=i,pattern=key,status=status,verified=verified))
            assert dict(raw)==rec['raw_counts'] and dict(effective)==rec['counts'] and statuses==rec['status_by_pattern'];counts['banks']+=1
        print(json.dumps(dict(seed=seed,regions=counts['regions'],positive=counts['raw_positive'],nonpositive=counts['raw_nonpositive'],unknown=counts['raw_unknown'])),flush=True)
    for seed,(x,v,regs,_) in inputs.items():
        for reg in regs:
            key=reg.tobytes().hex()
            for a,b in zip(model.METHODS,model.METHODS[1:]):
                if lookup[seed,a,key]=='positive':assert lookup[seed,b,key]=='positive'
    prior=root/'results/exact_credit_hull/development/banks.json';assert sha(prior)==p['prior_banks_sha256']
    ss,paired=summarize(records,read(prior));assert ss==s['summaries'] and paired==read(inp/'paired.json')
    assert [{k:v for k,v in r.items() if k!='strict_separations'} for r in paired]==s['comparisons'];counts['unique_parameter_infeasible']=len(parameters)
    oldpath=root/'results/baseline_history/analysis/retained_geometry.json';assert sha(oldpath)==p['prior_strict_sha256'];oldcases=[]
    for old in read(oldpath):oldcases.append(dict(seed=old['seed'],index=old['index'],pattern=old['pattern'],statuses={m:lookup[old['seed'],m,old['pattern']] for m in model.METHODS}))
    out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    for name,data in [('checks.json',checks),('intervals.json',crosschecks),('old_strict_cases.json',oldcases)]: (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=dict(counts),max_interval_lp_gap=max(r['max_endpoint_gap'] for r in crosschecks),old_strict_cases=oldcases,comparisons=s['comparisons'],source_sha256=sha(Path(__file__)),
        input_sha256={name:sha(inp/name) for name in ['protocol.json','summary.json','banks.json','paired.json']},output_sha256={name:sha(out/name) for name in ['checks.json','intervals.json','old_strict_cases.json']},scope=p['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
