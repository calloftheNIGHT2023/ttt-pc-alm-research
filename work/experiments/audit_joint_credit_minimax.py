"""Direct rational convex-credit optimum, reduced LP and full-region checks."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import optimized_branch_dual as original
import neighbor_mode_memory as neighbor
from verify_joint_credit_minimax import reduced_lp


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def exact_mixture(bank,weights):
    ids=np.flatnonzero(weights>0);ww=[F(float(weights[i])) for i in ids];total=sum(ww,F(0))
    return [[sum((w*F(float(bank[k,j,i])) for k,w in zip(ids,ww)),F(0))/total for i in range(bank.shape[2])] for j in range(bank.shape[1])]


def rational_optimum(x,v,reg,a):
    d,n=reg.shape;zl,zh,hl,hh=[[[F(float(t)) for t in row] for row in z[0]] for z in original.screen.boxes(v,reg[None])]
    total=sum(min(a[-1][i]*hl[-1][i],a[-1][i]*hh[-1][i]) for i in range(n))
    total-=sum(a[j][i]*int(original.base.INTERCEPTS[reg[j,i]]) for j in range(d) for i in range(n))
    for j in range(d):
        pl=[F(float(t)) for t in x] if j==0 else hl[j-1];ph=pl if j==0 else hh[j-1]
        low=max([-F(.12)]+[zl[j][i]-ph[i] for i in range(n)]);high=min([F(.12)]+[zh[j][i]-pl[i] for i in range(n)])
        assert low<=high
        sa=[int(original.base.SLOPES[reg[j,i]])*a[j][i] for i in range(n)]
        c=[(F(0) if j==0 else a[j-1][i])-sa[i] for i in range(n)]
        points=[low,high]+[max(low,min(high,zl[j][i]-pl[i] if c[i]>=0 else zh[j][i]-ph[i])) for i in range(n)]
        def value(b):
            return sum(min(c[i]*max(pl[i],zl[j][i]-b),c[i]*min(ph[i],zh[j][i]-b))-sa[i]*b for i in range(n))
        total+=min(value(b) for b in points)
    return total


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/joint_credit_minimax/development';p=json.loads((inp/'protocol.json').read_text());main=json.loads((inp/'summary.json').read_text())
    assert main['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    banks=json.loads((inp/'banks.json').read_text());source=root/'results/light_h2_credit/full_bank_ceiling'
    prior=root/'results/bounded_error_primal_gate/component';ia=json.loads((prior/'integer_audit.json').read_text());assert ia['passed']
    assert ia['source_sha256']==sha(Path(__file__).with_name('audit_primal_gate_integer.py'))
    assert ia['audits_sha256']==sha(prior/'audits.json')
    oldgate={(b['seed'],b['method']):b for b in json.loads((prior/'audits.json').read_text())}
    counts=dict(banks=0,regions=0,reduced_lp_replays=0,exact_convex_positive=0,strict_synergy_old_gates=0,full_region_lp_infeasible=0,
        exact_rounding_bounds=0,old_proof_retained=0)
    records=[];banksets={};maxgap=0.;maxerr=0.
    for rec in banks:
        assert sha(source/rec['source_data_file'])==rec['source_data_sha256']
        assert sha(inp/rec['rows_file'])==rec['rows_sha256'] and sha(inp/rec['arrays_file'])==rec['arrays_sha256']
        with np.load(source/rec['source_data_file']) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
        with np.load(inp/rec['arrays_file']) as z:credits=z['credit'];weights=z['weights'];old_positive=z['old_positive'];old_skip=z['old_gate_skipped']
        gr=oldgate[rec['seed'],rec['method']];assert sha(prior/gr['gate_file'])==gr['gate_sha256']
        with np.load(prior/gr['gate_file']) as z:assert np.array_equal(old_skip,z['skip'])
        rows=json.loads((inp/rec['rows_file']).read_text());accepted=set();proposals=[]
        for i,(row,reg) in enumerate(zip(rows,regs)):
            counts['regions']+=1;assert row['index']==i and row['pattern']==reg.tobytes().hex()
            assert row['old_positive']==bool(old_positive[i]) and row['old_gate_skipped']==bool(old_skip[i])
            if row['old_positive']:accepted.add(row['pattern']);counts['old_proof_retained']+=1;continue
            m=row['joint'];lp=reduced_lp(x,v,reg,bank);assert lp.success==m['success'];counts['reduced_lp_replays']+=1
            if not lp.success:assert lp.status==m['status'];continue
            err=abs(lp.fun-m['numerical_value']);assert err<1e-8;maxerr=max(maxerr,err)
            maxgap=max(maxgap,abs(m['duality_gap']))
            if not m['positive']:continue
            a=exact_mixture(bank,weights[i]);target=rational_optimum(x,v,reg,a);assert target>0
            proof=m['proof'];lower=F(int(proof['exact_convex_lower']['numerator']),int(proof['exact_convex_lower']['denominator']))
            assert 0<lower<=target
            err=sum(abs(a[j][k]-F(float(credits[i,j,k]))) for j in range(reg.shape[0]) for k in range(reg.shape[1]))
            actual=F(int(proof['mixture_rounding_l1']['numerator']),int(proof['mixture_rounding_l1']['denominator']));assert err==actual
            counts['exact_convex_positive']+=1;counts['exact_rounding_bounds']+=1
            _,_,mat,rhs=neighbor.pattern_matrix(x,v,reg)
            full=linprog(np.zeros(reg.shape[0]),A_ub=mat,b_ub=rhs,bounds=[(-.12,.12)]*reg.shape[0],options={'primal_feasibility_tolerance':1e-9})
            assert full.status==2;counts['full_region_lp_infeasible']+=1
            counts['strict_synergy_old_gates']+=int(old_skip[i]);accepted.add(row['pattern'])
            proposals.append(dict(index=i,pattern=row['pattern'],exact_value=float(target),exact_numerator=str(target.numerator),
                exact_denominator=str(target.denominator),strict_endpoint_synergy=bool(old_skip[i]),weight_support=int(np.sum(weights[i]>0))))
        banksets[rec['seed'],rec['method']]=accepted
        records.append(dict(seed=rec['seed'],method=rec['method'],proofs=proposals));counts['banks']+=1
        print(json.dumps(dict(banks=counts['banks'],exact_convex_positive=counts['exact_convex_positive'])),flush=True)
    matched=[]
    for seed in p['seeds']:
        native=banksets[seed,'alm_native'];residual=banksets[seed,'alm_residual']
        matched.append(dict(seed=seed,native_total=len(native),residual_total=len(residual),native_only=sorted(native-residual),residual_only=sorted(residual-native)))
    assert counts['exact_convex_positive']==main['counts']['new_exact_proofs'] and counts['strict_synergy_old_gates']==main['counts']['new_from_old_gate_skip']
    result=dict(passed=True,checks=counts,max_reduced_lp_error=maxerr,max_numerical_duality_gap=maxgap,alm_same_trajectory_comparison=matched,
        source_sha256=sha(Path(__file__)),input_sha256={name:sha(inp/name) for name in ['protocol.json','banks.json','summary.json']},
        old_gate_integer_audit_sha256=sha(prior/'integer_audit.json'),scope='All new proofs directly recomputed as exact convex mixtures; numerical nonpositive cases are not certified hull exhaustion')
    out=root/'results/joint_credit_minimax/audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'proofs.json').write_text(json.dumps(records,indent=2),encoding='utf-8');result['proofs_sha256']=sha(out/'proofs.json')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
