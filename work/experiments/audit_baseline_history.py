"""Full-history audit: unchanged captures, exact duals and every primal dot."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from math import lcm
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import baseline_history_capture as history
from run_baseline_history import OLD,summarize
from run_credit_wall_budget import load_inputs
from audit_joint_credit_minimax import exact_mixture,rational_optimum
from verify_joint_credit_minimax import reduced_lp
from verify_exact_credit_hull import independent_witness
import local_region_screen as screen
import neighbor_mode_memory as neighbor


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


class RowDyadicDots:
    """Independent row-specific denominators, unlike the solver's whole-bank scale."""
    def __init__(self,bank):
        self.rows=[]
        for row in bank.reshape(len(bank),-1):
            pairs=[float(t).as_integer_ratio() for t in row];power=max(d.bit_length()-1 for n,d in pairs)
            self.rows.append((power,[(j,n<<(power-(d.bit_length()-1))) for j,(n,d) in enumerate(pairs) if n]))

    def maximum(self,residual):
        den=lcm(*(r.denominator for r in residual));nums=[r.numerator*(den//r.denominator) for r in residual];best=None;bestpower=0
        for power,row in self.rows:
            value=sum(c*nums[j] for j,c in row)
            assert value<=0,'Exact primal upper witness has a positive direction'
            if best is None or (value<<bestpower)>(best<<power):best=value;bestpower=power
        return F(best,den*(1<<bestpower))


def direct_witness(x,v,reg,dots,witness):
    y=[F(int(n),int(d)) for n,d in witness['y']];depth,n=reg.shape;bias=y[:depth];h=np.array(y[depth:],dtype=object).reshape(depth,n)
    zl,zh,hl,hh=[a[0] for a in screen.boxes(v,reg[None])];assert all(-F(.12)<=b<=F(.12) for b in bias);residual=[]
    for j in range(depth):
        for i in range(n):
            z=bias[j]+(F(float(x[i])) if j==0 else h[j-1,i])
            assert F(float(zl[j,i]))<=z<=F(float(zh[j,i])) and F(float(hl[j,i]))<=h[j,i]<=F(float(hh[j,i]))
            residual.append(h[j,i]-int(screen.base.SLOPES[reg[j,i]])*z-int(screen.base.INTERCEPTS[reg[j,i]]))
    mx=dots.maximum(residual);assert [str(mx.numerator),str(mx.denominator)]==witness['proof']['max_credit']
    assert [[str(r.numerator),str(r.denominator)] for r in residual]==witness['proof']['residual']
    return float(mx)


def direct_positive(x,v,reg,bank,weights,credit,proof):
    weights=np.array(weights);credit=np.array(credit);assert np.all(weights>=0) and weights.sum()>0
    a=exact_mixture(bank,weights);value=rational_optimum(x,v,reg,a);lower=proof['exact_convex_lower'];lower=F(int(lower['numerator']),int(lower['denominator']))
    assert value>=lower>0
    err=sum(abs(a[j][i]-F(float(credit[j,i]))) for j in range(reg.shape[0]) for i in range(reg.shape[1]));target=proof['mixture_rounding_l1']
    assert err==F(int(target['numerator']),int(target['denominator']))
    return float(value)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/baseline_history';inp=base/'development'
    p=read(inp/'protocol.json');s=read(inp/'summary.json');assert s['execution_complete'];captures=read(inp/'captures.json');records=read(inp/'banks.json');inputs,data_hashes=load_inputs(root)
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert json.loads(json.dumps(data_hashes))==p['input_hashes'];counts=Counter();capture_checks=[];banks={}
    for cap in captures:
        seed=cap['seed'];learner=cap['learner'];x,v,regs,_=inputs[seed];assert sha(inp/cap['arrays_file'])==cap['arrays_sha256']
        actual=history.capture(x,v,learner,True);old,arrays,meta=actual
        with np.load(inp/cap['arrays_file']) as z:
            for key in z.files:assert z[key].dtype==arrays[key].dtype and z[key].shape==arrays[key].shape and z[key].tobytes()==arrays[key].tobytes();counts['capture_arrays']+=1
        for k in ['trajectory_sha256','trace_events','evaluation_calls','raw_rows','directions','raw_bytes','bank_bytes','diagnostic_array_bytes','old_directions','old_subset_bytewise']:assert meta[k]==cap['actual'][k]
        assert meta['trajectory_sha256']==cap['reference']['trajectory_sha256'];counts['trajectory_events']+=meta['trace_events'];counts['evaluation_calls']+=meta['evaluation_calls']
        assert sha(root/cap['old_source']['file'])==cap['old_source']['sha256']
        with np.load(root/cap['old_source']['file']) as z:
            for key,index in [('regs',2),('bank',3),('labels',4),('steps',5)]:assert z[key].tobytes()==old[index].tobytes();counts['old_arrays']+=1
        counts['old_direction_inclusions']+=len(arrays['old_mapping']);banks[seed,cap['method']]=arrays['bank'];counts['captures']+=1
        capture_checks.append(dict(seed=seed,method=cap['method'],trajectory_sha256=meta['trajectory_sha256'],directions=meta['directions']))
    for seed in inputs:banks[seed,'strong_history_union']=np.concatenate([banks[seed,m+'_history_full'] for m in history.LEARNERS])
    old_records=read(root/'results/exact_credit_hull/development/banks.json');checks=[];parameters=set();max_lp_gap=0.;lookup={}
    for rec in records:
        seed=rec['seed'];method=rec['method'];x,v,regs,_=inputs[seed];bank=banks[seed,method];dots=RowDyadicDots(bank);slow_cases=0
        assert hashlib.sha256(bank.tobytes()).hexdigest()==rec['bank_sha256'] and bank.nbytes==rec['bank_bytes'];assert sha(inp/rec['rows_file'])==rec['rows_sha256']
        rows=read(inp/rec['rows_file']);assert len(rows)==len(regs);raw=Counter();effective=Counter();status_by_pattern={}
        for i,(row,reg) in enumerate(zip(rows,regs)):
            key=reg.tobytes().hex();assert row['index']==i and row['pattern']==key;r=row['result'];raw[r['status']]+=1
            lp=reduced_lp(x,v,reg,bank);assert lp.success==r['dual']['success'];counts['reduced_lp_checks']+=1
            if lp.success:gap=abs(lp.fun-r['dual']['numerical_value']);assert gap<1e-8;max_lp_gap=max(max_lp_gap,gap)
            value=None
            if r['status']=='positive':value=direct_positive(x,v,reg,bank,r['weights'],r['credit'],r['dual']['proof']);counts['new_exact_positive']+=1
            elif r['status']=='nonpositive':
                value=direct_witness(x,v,reg,dots,r['witness']);counts['exact_nonpositive']+=1;counts['exact_primal_directions']+=len(bank)
                # Full independent Fraction loop on the first two upper witnesses
                # in EVERY task/library, not only favorable cases.
                if slow_cases<2:independent_witness(x,v,reg,bank,r['witness']);slow_cases+=1;counts['slow_fraction_cases']+=1;counts['slow_fraction_directions']+=len(bank)
            else:assert r['status']=='unknown';counts['raw_unknown']+=1
            inherited=row['inherited'];status=r['status']
            if inherited is not None:
                assert status=='unknown' and row['prior_status']=='positive';value=direct_positive(x,v,reg,bank,inherited['weights'],inherited['credit'],inherited['proof']);status='positive';counts['inherited_positive']+=1
            assert status==row['effective_status'];effective[status]+=1;status_by_pattern[key]=status;lookup[seed,method,key]=status
            if row['prior_status']=='positive':assert status=='positive'
            if status=='positive' and (seed,key) not in parameters:
                _,_,mat,rhs=neighbor.pattern_matrix(x,v,reg);original=linprog(np.zeros(4),A_ub=mat,b_ub=rhs,bounds=[(-.12,.12)]*4,options=dict(primal_feasibility_tolerance=1e-9));assert original.status==2;parameters.add((seed,key))
            checks.append(dict(seed=seed,method=method,index=i,pattern=key,status=status,exact_value=value));counts['regions']+=1
        assert dict(raw)==rec['raw_counts'] and dict(effective)==rec['effective_counts'] and status_by_pattern==rec['status_by_pattern'];counts['banks']+=1
        print(json.dumps(dict(banks=counts['banks'],regions=counts['regions'],positive=counts['new_exact_positive'],nonpositive=counts['exact_nonpositive'],unknown=counts['raw_unknown'])),flush=True)
    summaries,paired=summarize(records,old_records);assert summaries==s['summaries'] and paired==read(inp/'paired.json')
    assert [{k:v for k,v in r.items() if k!='strict_separations'} for r in paired]==s['comparisons'];counts['unique_parameter_infeasible']=len(parameters)
    cases=[]
    for block in read(root/'results/exact_credit_hull/development/paired.json'):
        if block['comparator'] not in ['adam60_native','pc_native','nodual_native']:continue
        method=next(m for m,o in OLD.items() if o==block['comparator'])
        for case in block['strict_separations']:cases.append(dict(**case,full_method=method,full_status=lookup[case['seed'],method,case['pattern']]))
    out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');(out/'captures.json').write_text(json.dumps(capture_checks,indent=2),encoding='utf-8');(out/'old_strict_cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=dict(counts),old_strict_case_transitions=dict(Counter(r['full_status'] for r in cases)),max_reduced_lp_gap=max_lp_gap,comparisons=s['comparisons'],
        source_sha256=sha(Path(__file__)),input_sha256={name:sha(inp/name) for name in ['protocol.json','summary.json','captures.json','banks.json','paired.json']},
        output_sha256={name:sha(out/name) for name in ['checks.json','captures.json','old_strict_cases.json']},scope='Full recorded trajectory capacity verified exactly; diagnostic costs only, no independent online risk result')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
