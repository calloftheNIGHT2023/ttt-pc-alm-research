"""Independent exact mixtures and post-run ceiling comparison; never an input."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import optimized_branch_dual as original
import neighbor_mode_memory as neighbor
from audit_joint_credit_minimax import exact_mixture, rational_optimum
from verify_finite_credit_game import guarded_solve


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',type=Path,required=True); root=ap.parse_args().project.resolve()
    inp=root/'results/finite_credit_game/development'; p=json.loads((inp/'protocol.json').read_text()); s=json.loads((inp/'summary.json').read_text())
    assert s['execution_complete'] and s['banks_sha256']==sha(inp/'banks.json') and s['protocol_sha256']==sha(inp/'protocol.json')
    for name,value in p['source_sha256'].items(): assert sha(Path(__file__).with_name(name))==value,name
    source=root/'results/light_h2_credit/full_bank_ceiling'; banks=json.loads((inp/'banks.json').read_text())
    ceiling=root/'results/joint_credit_minimax/development'; cp=json.loads((ceiling/'summary.json').read_text()); assert cp['execution_complete']
    joint={(r['seed'],r['method']):r for r in json.loads((ceiling/'banks.json').read_text())}
    ca=root/'results/joint_credit_minimax/audit/summary.json'; assert json.loads(ca.read_text())['passed']
    checks=dict(banks=0,regions=0,replayed_regions=0,replayed_arrays=0,old_proofs=0,new_exact=0,
                exact_convex_mixtures=0,exact_rounding_bounds=0,full_region_infeasible=0,strict_synergy=0)
    rows=[]; maxdiff=0.; sets={}
    for rec in banks:
        assert sha(source/rec['source_data_file'])==rec['source_data_sha256']
        assert sha(inp/rec['arrays_file'])==rec['arrays_sha256'] and sha(inp/rec['meta_file'])==rec['meta_sha256']
        with np.load(source/rec['source_data_file']) as z: x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
        with np.load(inp/rec['arrays_file']) as z: arrays={key:z[key] for key in z.files}
        meta=json.loads((inp/rec['meta_file']).read_text()); indices=arrays['indices']; old=arrays['old_positive']
        assert np.array_equal(indices,np.flatnonzero(~old))
        again,am=guarded_solve(x,v,regs[indices],bank)
        for key,value in again.items(): assert value.tobytes()==arrays[key].tobytes(); checks['replayed_arrays']+=1
        assert am['proofs']==meta['proofs']; checks['replayed_regions']+=len(indices)
        fixed=original.float_optimum(x,v,regs[indices],arrays['credit'])
        diff=float(np.max(abs(fixed-arrays['best_value']),initial=0)); assert diff<1e-10; maxdiff=max(maxdiff,diff)
        assert np.all(arrays['weights']>=0) and np.max(abs(arrays['weights'].sum(1)-1),initial=0)<1e-12
        assert np.max(abs(np.einsum('rk,kdn->rdn',arrays['weights'],bank)-arrays['credit']),initial=0)<1e-12
        for trace in meta['traces']:
            assert trace['positive']==int(np.sum((arrays['first_step']>0)&(arrays['first_step']<=trace['step'])))
        cr=joint[rec['seed'],rec['method']]
        assert sha(ceiling/cr['rows_file'])==cr['rows_sha256']
        jj=json.loads((ceiling/cr['rows_file']).read_text())
        accepted={regs[i].tobytes().hex() for i in np.flatnonzero(old)}; positive=[]
        for proof in meta['proofs']:
            q=proof['index']; index=int(indices[q]); reg=regs[index]; a=arrays['credit'][q]; w=arrays['weights'][q]
            assert proof['pattern']==reg.tobytes().hex() and proof['step']==int(arrays['first_step'][q])
            direct=original.exact_optimum(x,v,reg,a); assert direct==proof['exact'] and direct['positive']; checks['new_exact']+=1
            exacta=exact_mixture(bank,w); value=rational_optimum(x,v,reg,exacta); assert value>0; checks['exact_convex_mixtures']+=1
            error=sum(abs(exacta[j][i]-F(float(a[j,i]))) for j in range(reg.shape[0]) for i in range(reg.shape[1]))
            exactfloat=F(int(direct['numerator']),int(direct['denominator'])); lower=exactfloat-error
            assert 0<lower<=value; checks['exact_rounding_bounds']+=1
            _,_,mat,rhs=neighbor.pattern_matrix(x,v,reg)
            lp=linprog(np.zeros(reg.shape[0]),A_ub=mat,b_ub=rhs,bounds=[(-.12,.12)]*reg.shape[0],options={'primal_feasibility_tolerance':1e-9})
            assert lp.status==2; checks['full_region_infeasible']+=1
            assert jj[index]['joint']['positive']; synergy=jj[index]['old_gate_skipped']; checks['strict_synergy']+=int(synergy)
            positive.append(dict(index=index,step=proof['step'],strict_synergy=synergy,value=float(value),
                numerator=str(value.numerator),denominator=str(value.denominator),rounding_l1=float(error),convex_lower=float(lower)))
            accepted.add(proof['pattern'])
        eligible=np.array([jj[i]['joint']['positive'] for i in indices]); w=arrays['weights']
        entropy=-(np.where(w>0,w*np.log(np.maximum(w,np.finfo(float).tiny)),0)).sum(1)
        rows.append(dict(seed=rec['seed'],method=rec['method'],joint_new=cr['new_proofs'],finite_new=len(positive),
            proofs=positive,strict_synergy=sum(q['strict_synergy'] for q in positive),
            entropy_fraction_on_joint_positive=(entropy[eligible]/np.log(len(bank))).tolist(),
            last_best_value_on_joint_positive=arrays['best_value'][eligible].tolist()))
        sets[rec['seed'],rec['method']]=accepted
        checks['banks']+=1;checks['regions']+=len(regs);checks['old_proofs']+=int(old.sum())
        print(json.dumps(dict(banks=checks['banks'],positive=checks['new_exact'])),flush=True)
    paired=[]
    for seed in p['seeds']:
        a=sets[seed,'alm_native']; b=sets[seed,'alm_residual']
        paired.append(dict(seed=seed,native_total=len(a),residual_total=len(b),native_only=len(a-b),residual_only=len(b-a)))
    summaries=[]
    for method in p['methods']:
        rr=[r for r in rows if r['method']==method]; ent=[v for r in rr for v in r['entropy_fraction_on_joint_positive']]
        summaries.append(dict(method=method,joint_new=sum(r['joint_new'] for r in rr),finite_new=sum(r['finite_new'] for r in rr),
            strict_synergy=sum(r['strict_synergy'] for r in rr),entropy_fraction_quantiles=np.quantile(ent,[0,.5,1]).tolist() if ent else []))
    assert checks['new_exact']==s['counts']['new_proofs'] and checks['replayed_regions']==3266 and checks['old_proofs']==732
    out=root/'results/finite_credit_game/audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'proofs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=checks,summaries=summaries,alm_paired_sets=paired,max_saved_objective_error=maxdiff,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),banks_sha256=sha(inp/'banks.json'),
        main_summary_sha256=sha(inp/'summary.json'),proofs_sha256=sha(out/'proofs.json'),ceiling_audit_sha256=sha(ca),
        scope='Independent exact certificate arithmetic plus full replay; ceiling and entropy only examined after frozen run')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
