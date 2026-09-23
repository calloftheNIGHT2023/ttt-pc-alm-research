"""Independent primal/dual rational checks and symmetric separation audit."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import neighbor_mode_memory as neighbor
from run_exact_credit_hull import prepare,summarize,METHODS
from run_credit_wall_budget import load_inputs
from verify_exact_credit_hull import validate
from audit_joint_credit_minimax import exact_mixture


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/exact_credit_hull';inp=base/'development';p=read(inp/'protocol.json');s=read(inp/'summary.json');assert s['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    inputs,data_hashes=load_inputs(root);prepare(inputs);assert json.loads(json.dumps(data_hashes))==p['input_hashes']
    records=read(inp/'banks.json');assert len(records)==160;counts=Counter();verified=[];original=set();lookup={};maxerror=0.
    for rec in records:
        seed=rec['seed'];method=rec['method'];x,v,regs,banks=inputs[seed];bank=banks[method];assert len(bank)==rec['directions']
        assert hashlib.sha256(bank.tobytes()).hexdigest()==rec['bank_sha256'] and bank.nbytes==rec['bank_bytes']
        assert sha(inp/rec['rows_file'])==rec['rows_sha256'];rows=read(inp/rec['rows_file']);assert len(rows)==len(regs);status=Counter();by_pattern={}
        for i,(row,reg) in enumerate(zip(rows,regs)):
            key=reg.tobytes().hex();assert row['index']==i and row['pattern']==key;r=row['result'];proof=validate(x,v,reg,bank,r)
            counts['regions']+=1;counts[r['status']]+=1;status[r['status']]+=1;by_pattern[key]=r['status'];lookup[seed,method,key]=r['status']
            if r['status']=='positive':
                weights=np.array(r['weights']);a=exact_mixture(bank,weights);credit=np.array(r['credit']);err=sum(abs(a[j][k]-F(float(credit[j,k]))) for j in range(reg.shape[0]) for k in range(reg.shape[1]))
                target=r['dual']['proof']['mixture_rounding_l1'];assert err==F(int(target['numerator']),int(target['denominator']));counts['exact_rounding']+=1
                pair=(seed,key)
                if pair not in original:
                    _,_,mat,rhs=neighbor.pattern_matrix(x,v,reg)
                    lp=linprog(np.zeros(reg.shape[0]),A_ub=mat,b_ub=rhs,bounds=[(-.12,.12)]*reg.shape[0],options=dict(primal_feasibility_tolerance=1e-9))
                    assert lp.status==2;original.add(pair)
            elif r['status']=='nonpositive':counts['exact_primal_directions']+=proof['directions']
            verified.append(dict(seed=seed,method=method,index=i,pattern=key,proof=proof))
        assert dict(status)==rec['counts'] and by_pattern==rec['status_by_pattern'];counts['banks']+=1
        print(json.dumps(dict(banks=counts['banks'],regions=counts['regions'],positive=counts['positive'],nonpositive=counts['nonpositive'],unknown=counts['unknown'])),flush=True)
    summaries,paired=summarize(records);assert summaries==s['summaries'] and paired==read(inp/'paired.json')
    assert [{k:v for k,v in row.items() if k!='strict_separations'} for row in paired]==s['comparisons']
    counts['unique_parameter_infeasible']=len(original)
    # Every prior accepted timed proof must be compatible with the exact upper side.
    timed=read(root/'results/credit_wall_budget/development/rows.json');pending=set();timed_positive=set()
    for r in timed:
        for key in r['accepted_patterns']:
            q=(r['seed'],r['method'],key);assert lookup[q]!='nonpositive';timed_positive.add(q)
            if lookup[q]=='unknown':pending.add(q)
    counts['timed_positive_compatible']=len(timed_positive);counts['timed_positive_still_unknown']=len(pending)
    # An upper witness for the strong union forbids positive constituents.
    union_unknown=0
    for seed,(x,v,regs,banks) in inputs.items():
        for reg in regs:
            key=reg.tobytes().hex();u=lookup[seed,'strong_union',key]
            if u=='nonpositive':assert all(lookup[seed,m,key]!='positive' for m in p['union_components'])
            if any(lookup[seed,m,key]=='positive' for m in p['union_components']) and u=='unknown':union_unknown+=1
    counts['union_positive_constituent_still_unknown']=union_unknown
    out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'checks.json').write_text(json.dumps(verified,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=dict(counts),comparisons=s['comparisons'],source_sha256=sha(Path(__file__)),
        input_sha256={name:sha(inp/name) for name in ['protocol.json','summary.json','banks.json','paired.json']},
        checks_sha256=sha(out/'checks.json'),scope='Exact separation or nonseparation for fixed libraries only; unknown never counted as inability')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
