"""All-path moment bounds, exact convex certificates, and changed-set audit."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp
import credit_moment_step as model
import neighbor_mode_memory as neighbor
from audit_joint_credit_minimax import exact_mixture,rational_optimum
from verify_credit_moment_step import guarded_solve


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def checked_replay(x,v,regs,bank):
    update=model.update;checks=dict(rows=0,max_post_minus_target=-1.,min_mean_increase=1.,min_partition_slack=1.,min_potential_slack=1.,min_pythagorean_slack=1.)
    def wrapped(logs,gains,delta=model.DELTA):
        new,meta=update(logs,gains,delta);assert np.all(meta['status']==0)
        low=gains.min(1);span=np.ptp(gains,axis=1);gg=(gains-low[:,None])/span[:,None]
        p=model.probabilities(logs);q=model.probabilities(new);mu=(p*gg).sum(1);post=(q*gg).sum(1)
        var=(p*(gg-mu[:,None])**2).sum(1);target=(delta-low)/span;gap=target-mu;b=meta['beta']
        assert np.all(post<=target+1e-12) and np.all(post>=mu-1e-12)
        lz=logsumexp(logs+b[:,None]*gg,axis=1)-logsumexp(logs,axis=1)
        upper=mu*b+var*np.expm1(b)-var*b;partition=upper-lz;assert partition.min(initial=0)>-1e-10
        lower=(var+gap)*b-gap;decrease=b*gg.max(1)-lz;assert lower.min(initial=0)>-1e-12
        potential=decrease-lower;assert potential.min(initial=0)>-1e-10
        move=b*post-lz;pyth=decrease-move;assert pyth.min(initial=0)>-1e-10
        checks['rows']+=len(gains);checks['max_post_minus_target']=max(checks['max_post_minus_target'],float((post-target).max()))
        for key,val in [('min_mean_increase',post-mu),('min_partition_slack',partition),('min_potential_slack',potential),('min_pythagorean_slack',pyth)]:
            checks[key]=min(checks[key],float(val.min()))
        return new,meta
    with patch.object(model,'update',wrapped):arrays,meta=guarded_solve(x,v,regs,bank)
    return arrays,meta,checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/credit_moment_step/development';p=json.loads((inp/'protocol.json').read_text());s=json.loads((inp/'summary.json').read_text())
    assert s['execution_complete'] and s['banks_sha256']==sha(inp/'banks.json') and s['protocol_sha256']==sha(inp/'protocol.json')
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    source=root/'results/light_h2_credit/full_bank_ceiling';banks=json.loads((inp/'banks.json').read_text())
    ceiling=root/'results/joint_credit_minimax/development';joint={(r['seed'],r['method']):r for r in json.loads((ceiling/'banks.json').read_text())}
    ca=root/'results/joint_credit_minimax/audit/summary.json';assert json.loads(ca.read_text())['passed']
    baselines={label:root/f'results/{folder}/development' for label,folder in [('fixed','finite_credit_game'),('projection','credit_halfspace_projection')]}
    prior={label:{(r['seed'],r['method']):r for r in json.loads((path/'banks.json').read_text())} for label,path in baselines.items()}
    counts=dict(banks=0,regions=0,replayed_regions=0,replayed_arrays=0,old_proofs=0,new_exact=0,exact_convex_mixtures=0,
        exact_rounding_bounds=0,full_region_infeasible=0,strict_synergy=0)
    rows=[];pathchecks=[];sets={};maxdiff=0.
    for rec in banks:
        assert sha(source/rec['source_data_file'])==rec['source_data_sha256']
        assert sha(inp/rec['arrays_file'])==rec['arrays_sha256'] and sha(inp/rec['meta_file'])==rec['meta_sha256']
        with np.load(source/rec['source_data_file']) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
        with np.load(inp/rec['arrays_file']) as z:arrays={key:z[key] for key in z.files}
        meta=json.loads((inp/rec['meta_file']).read_text());indices=arrays['indices'];old=arrays['old_positive']
        assert np.array_equal(indices,np.flatnonzero(~old))
        again,am,checks=checked_replay(x,v,regs[indices],bank)
        for key,value in again.items():assert value.tobytes()==arrays[key].tobytes();counts['replayed_arrays']+=1
        assert am['proofs']==meta['proofs'];counts['replayed_regions']+=len(indices);pathchecks.append(dict(seed=rec['seed'],method=rec['method'],**checks))
        fixed=model.original.float_optimum(x,v,regs[indices],arrays['credit']);diff=float(np.max(abs(fixed-arrays['best_value']),initial=0))
        assert diff<1e-10;maxdiff=max(maxdiff,diff)
        assert np.max(abs(arrays['weights'].sum(1)-1),initial=0)<1e-12 and np.all(arrays['weights']>=0)
        assert np.max(abs(np.einsum('rk,kdn->rdn',arrays['weights'],bank)-arrays['credit']),initial=0)<1e-12
        assert np.all((arrays['first_step']>0)==(arrays['terminal']==1))
        assert meta['mean_evaluations']==2*meta['variance_evaluations']==2*checks['rows']
        for trace in meta['traces']:assert trace['positive']==int(np.sum((arrays['first_step']>0)&(arrays['first_step']<=trace['step'])))
        cr=joint[rec['seed'],rec['method']];assert sha(ceiling/cr['rows_file'])==cr['rows_sha256'];jj=json.loads((ceiling/cr['rows_file']).read_text())
        accepted={regs[i].tobytes().hex() for i in np.flatnonzero(old)};positive=[]
        for proof in meta['proofs']:
            q=proof['index'];index=int(indices[q]);reg=regs[index];a=arrays['credit'][q];w=arrays['weights'][q]
            assert proof['pattern']==reg.tobytes().hex() and proof['step']==int(arrays['first_step'][q])
            direct=model.original.exact_optimum(x,v,reg,a);assert direct==proof['exact'] and direct['positive'];counts['new_exact']+=1
            exacta=exact_mixture(bank,w);value=rational_optimum(x,v,reg,exacta);assert value>0;counts['exact_convex_mixtures']+=1
            error=sum(abs(exacta[j][i]-F(float(a[j,i]))) for j in range(reg.shape[0]) for i in range(reg.shape[1]))
            lower=F(int(direct['numerator']),int(direct['denominator']))-error;assert 0<lower<=value;counts['exact_rounding_bounds']+=1
            _,_,mat,rhs=neighbor.pattern_matrix(x,v,reg)
            lp=linprog(np.zeros(reg.shape[0]),A_ub=mat,b_ub=rhs,bounds=[(-.12,.12)]*reg.shape[0],options={'primal_feasibility_tolerance':1e-9})
            assert lp.status==2;counts['full_region_infeasible']+=1
            assert jj[index]['joint']['positive'];synergy=jj[index]['old_gate_skipped'];counts['strict_synergy']+=int(synergy)
            positive.append(dict(index=index,step=proof['step'],strict_synergy=synergy,value=float(value),numerator=str(value.numerator),
                denominator=str(value.denominator),rounding_l1=float(error),convex_lower=float(lower)))
            accepted.add(proof['pattern'])
        compare={}
        for label,path in baselines.items():
            br=prior[label][rec['seed'],rec['method']];assert sha(path/br['arrays_file'])==br['arrays_sha256']
            with np.load(path/br['arrays_file']) as z:assert np.array_equal(z['indices'],indices);before=z['first_step']>0
            after=arrays['first_step']>0
            compare[label]=dict(retained=int(np.sum(before&after)),lost=int(np.sum(before&~after)),new=int(np.sum(~before&after)))
        rows.append(dict(seed=rec['seed'],method=rec['method'],joint_new=cr['new_proofs'],moment_new=len(positive),proofs=positive,
            strict_synergy=sum(q['strict_synergy'] for q in positive),comparison=compare))
        sets[rec['seed'],rec['method']]=accepted;counts['banks']+=1;counts['regions']+=len(regs);counts['old_proofs']+=int(old.sum())
        print(json.dumps(dict(banks=counts['banks'],proofs=counts['new_exact'],moment_rows=sum(q['rows'] for q in pathchecks))),flush=True)
    paired=[]
    for seed in p['seeds']:
        a=sets[seed,'alm_native'];b=sets[seed,'alm_residual'];paired.append(dict(seed=seed,native_total=len(a),residual_total=len(b),native_only=len(a-b),residual_only=len(b-a)))
    summaries=[]
    for method in p['methods']:
        rr=[r for r in rows if r['method']==method]
        summaries.append(dict(method=method,**{key:sum(r[key] for r in rr) for key in ['joint_new','moment_new','strict_synergy']},
            comparison={label:{key:sum(r['comparison'][label][key] for r in rr) for key in ['retained','lost','new']} for label in baselines}))
    assert counts['new_exact']==s['counts']['new_proofs'] and counts['replayed_regions']==3266 and counts['old_proofs']==732
    out=root/'results/credit_moment_step/audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'proofs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'moment_checks.json').write_text(json.dumps(pathchecks,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=counts,summaries=summaries,alm_paired_sets=paired,max_saved_objective_error=maxdiff,
        moment_checks=dict(rows=sum(q['rows'] for q in pathchecks),max_post_minus_target=max(q['max_post_minus_target'] for q in pathchecks),
            **{key:min(q[key] for q in pathchecks) for key in ['min_mean_increase','min_partition_slack','min_potential_slack','min_pythagorean_slack']}),
        source_sha256=sha(Path(__file__)),input_sha256={name:sha(inp/name) for name in ['protocol.json','banks.json','summary.json']},
        proofs_sha256=sha(out/'proofs.json'),moment_checks_sha256=sha(out/'moment_checks.json'),ceiling_audit_sha256=sha(ca),
        scope='All actual moment steps numerically checked; all accepted convex-credit exclusions independently exact')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
