"""Replay all shared-parameter probes; independent Fraction validation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import joint_forward_realization as model
from verify_joint_forward_realization import validate_generation
from verify_exact_credit_hull import validate
from audit_baseline_history import direct_positive
from run_joint_forward_realization import summarize
from run_credit_wall_budget import load_inputs
import neighbor_mode_memory as neighbor

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/joint_forward_realization';inp=base/'development';src=Path(__file__).parent
    p=read(inp/'protocol.json');s=read(inp/'summary.json');assert s['execution_complete'];records=read(inp/'banks.json');generations=read(inp/'generations.json');inputs,hashes=load_inputs(root)
    for name,value in p['source_sha256'].items():assert sha(src/name)==value,name
    assert json.loads(json.dumps(hashes))==p['input_hashes'];hist=root/'results/baseline_history/development'
    for name,value in p['full_history_input_sha256'].items():assert sha(hist/name)==value
    counts=Counter();checks=[];genchecks=[];parameters=set();lookup={}
    for gen in generations:
        seed=gen['seed'];x,v,regs,_=inputs[seed];arrays,generation,meta=model.generate(x,v)
        assert sha(inp/gen['arrays_file'])==gen['arrays_sha256'] and sha(inp/gen['generation_file'])==gen['generation_sha256']
        with np.load(inp/gen['arrays_file']) as z:
            assert set(z.files)==set(arrays)
            for key in z.files:assert z[key].dtype==arrays[key].dtype and z[key].shape==arrays[key].shape and z[key].tobytes()==arrays[key].tobytes();counts['replayed_arrays']+=1
        assert generation==read(inp/gen['generation_file'])
        for key,value in meta.items():
            if not key.endswith('_seconds'):assert value==gen['metadata'][key],key
        q=validate_generation(x,v,arrays,generation);counts.update(q);genchecks.append(dict(seed=seed,checks=q));counts['generations']+=1
        cap=gen['old_bank_source'];oldrec=gen['old_rows'];assert sha(hist/cap['arrays_file'])==cap['arrays_sha256'] and sha(hist/oldrec['rows_file'])==oldrec['rows_sha256']
        with np.load(hist/cap['arrays_file']) as z:oldbank=z['bank']
        oldrows=read(hist/oldrec['rows_file']);banks=dict(constructed_joint=arrays['bank'],old_plus_constructed_joint=np.concatenate([oldbank,arrays['bank']]));previous=None
        for method,bank in banks.items():
            rec=next(r for r in records if r['seed']==seed and r['method']==method);assert sha(inp/rec['rows_file'])==rec['rows_sha256'] and hashlib.sha256(bank.tobytes()).hexdigest()==rec['bank_sha256']
            rows=read(inp/rec['rows_file']);assert len(rows)==len(regs);raw=Counter();effective=Counter();statuses={}
            for i,(row,reg) in enumerate(zip(rows,regs)):
                key=reg.tobytes().hex();assert row['index']==i and row['pattern']==key;r=row['result'];verified=validate(x,v,reg,bank,r);counts['raw_'+r['status']]+=1;raw[r['status']]+=1;status=r['status']
                if status=='positive':direct_positive(x,v,reg,bank,r['weights'],r['credit'],r['dual']['proof']);counts['exact_rounding']+=1
                if status=='nonpositive':counts['fraction_primal_directions']+=len(bank)
                if row['inherited'] is not None:
                    assert status=='unknown';q=row['inherited'];direct_positive(x,v,reg,bank,q['weights'],q['credit'],q['proof']);status='positive';counts['inherited_positive']+=1
                assert status==row['effective_status'];effective[status]+=1;statuses[key]=status;lookup[seed,method,key]=status;counts['regions']+=1
                if method=='old_plus_constructed_joint' and (previous[i]['effective_status']=='positive' or oldrows[i]['effective_status']=='positive'):assert status=='positive'
                if status=='positive' and (seed,key) not in parameters:
                    _,_,matrix,rhs=neighbor.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=matrix,b_ub=rhs,bounds=[(-.12,.12)]*4,options=dict(primal_feasibility_tolerance=1e-9));assert lp.status==2;parameters.add((seed,key))
                checks.append(dict(seed=seed,method=method,index=i,pattern=key,status=status,verified=verified))
            assert dict(raw)==rec['raw_counts'] and dict(effective)==rec['counts'] and statuses==rec['status_by_pattern'];counts['banks']+=1;previous=rows
            print(json.dumps(dict(seed=seed,method=method,regions=counts['regions'],positive=counts['raw_positive'],nonpositive=counts['raw_nonpositive'],unknown=counts['raw_unknown'])),flush=True)
    prior=root/'results/exact_credit_hull/development/banks.json';assert sha(prior)==p['alm_banks_sha256'];ss,paired=summarize(records,read(prior));assert ss==s['summaries'] and paired==read(inp/'paired.json')
    assert [{k:v for k,v in r.items() if k!='strict_separations'} for r in paired]==s['comparisons'];counts['unique_parameter_infeasible']=len(parameters)
    oldpath=root/'results/baseline_history/analysis/retained_geometry.json';assert sha(oldpath)==p['prior_strict_sha256']
    oldcases=[dict(seed=r['seed'],index=r['index'],pattern=r['pattern'],statuses={m:lookup[r['seed'],m,r['pattern']] for m in model.METHODS}) for r in read(oldpath)]
    out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    for name,data in [('checks.json',checks),('generations.json',genchecks),('old_strict_cases.json',oldcases)]: (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=dict(counts),old_strict_cases=oldcases,comparisons=s['comparisons'],source_sha256=sha(Path(__file__)),
        input_sha256={name:sha(inp/name) for name in ['protocol.json','summary.json','generations.json','banks.json','paired.json']},output_sha256={name:sha(out/name) for name in ['checks.json','generations.json','old_strict_cases.json']},scope=p['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
