"""All-path projection checks, exact certificates and post-run comparisons."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog
import credit_halfspace_projection as model
import neighbor_mode_memory as neighbor
from audit_joint_credit_minimax import exact_mixture,rational_optimum
from verify_credit_halfspace_projection import guarded_solve


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def checked_replay(x,v,regs,bank):
    project=model.project
    checked=dict(projected_rows=0,limited_rows=0,max_mean_error=0.,max_kkt_range=0.,min_pythagorean_slack=1.,min_progress_slack=1.)
    def wrapped(logs,gains,delta=model.DELTA):
        new,meta=project(logs,gains,delta)
        assert not np.any(meta['status']>=2)
        limited=meta['status']==1;checked['limited_rows']+=int(limited.sum())
        if limited.any():assert np.all(gains[limited].max(1)<=delta)
        ids=np.flatnonzero(meta['status']==0)
        if len(ids):
            l=logs[ids];nl=new[ids];g=gains[ids];span=np.ptp(g,axis=1);gg=(g-g.min(1,keepdims=True))/span[:,None]
            p=model.probabilities(l);q=model.probabilities(nl)
            lp=l-l.max(1,keepdims=True);lp-=np.log(np.exp(lp).sum(1,keepdims=True))
            lq=nl-nl.max(1,keepdims=True);lq-=np.log(np.exp(lq).sum(1,keepdims=True))
            mean=(q*g).sum(1);error=abs(mean-delta);assert np.max(error)<1e-10
            kkt=np.ptp(lq-lp-meta['beta'][ids,None]*gg,axis=1);assert np.max(kkt)<1e-9
            move=(q*(lq-lp)).sum(1);best=g.argmax(1)
            # Reference e_argmax belongs to this individual halfspace. It is
            # not asserted to be a common separator for the full trajectory.
            decrease=lq[np.arange(len(ids)),best]-lp[np.arange(len(ids)),best]
            slack=decrease-move;assert slack.min(initial=0)>-1e-9
            gap=delta-(p*g).sum(1);lower=2*gap**2/span**2
            assert (move-lower).min(initial=0)>-1e-9
            checked['projected_rows']+=len(ids)
            checked['max_mean_error']=max(checked['max_mean_error'],float(error.max()))
            checked['max_kkt_range']=max(checked['max_kkt_range'],float(kkt.max()))
            checked['min_pythagorean_slack']=min(checked['min_pythagorean_slack'],float(slack.min()))
            checked['min_progress_slack']=min(checked['min_progress_slack'],float((move-lower).min()))
        return new,meta
    with patch.object(model,'project',wrapped):arrays,meta=guarded_solve(x,v,regs,bank)
    return arrays,meta,checked


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/credit_halfspace_projection/development';p=json.loads((inp/'protocol.json').read_text());s=json.loads((inp/'summary.json').read_text())
    assert s['execution_complete'] and s['banks_sha256']==sha(inp/'banks.json') and s['protocol_sha256']==sha(inp/'protocol.json')
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    source=root/'results/light_h2_credit/full_bank_ceiling';banks=json.loads((inp/'banks.json').read_text())
    ceiling=root/'results/joint_credit_minimax/development';joint={(r['seed'],r['method']):r for r in json.loads((ceiling/'banks.json').read_text())}
    ca=root/'results/joint_credit_minimax/audit/summary.json';assert json.loads(ca.read_text())['passed']
    baseline=root/'results/finite_credit_game/development';prior={(r['seed'],r['method']):r for r in json.loads((baseline/'banks.json').read_text())}
    counts=dict(banks=0,regions=0,replayed_regions=0,replayed_arrays=0,old_proofs=0,new_exact=0,exact_convex_mixtures=0,
        exact_rounding_bounds=0,full_region_infeasible=0,strict_synergy=0,previous_retained=0,previous_lost=0)
    rows=[];sets={};maxdiff=0.;projection=[]
    for rec in banks:
        assert sha(source/rec['source_data_file'])==rec['source_data_sha256']
        assert sha(inp/rec['arrays_file'])==rec['arrays_sha256'] and sha(inp/rec['meta_file'])==rec['meta_sha256']
        with np.load(source/rec['source_data_file']) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
        with np.load(inp/rec['arrays_file']) as z:arrays={key:z[key] for key in z.files}
        meta=json.loads((inp/rec['meta_file']).read_text());indices=arrays['indices'];old=arrays['old_positive']
        assert np.array_equal(indices,np.flatnonzero(~old))
        again,am,checks=checked_replay(x,v,regs[indices],bank)
        for key,value in again.items():assert value.tobytes()==arrays[key].tobytes();counts['replayed_arrays']+=1
        assert am['proofs']==meta['proofs'];counts['replayed_regions']+=len(indices);projection.append(dict(seed=rec['seed'],method=rec['method'],**checks))
        fixed=model.original.float_optimum(x,v,regs[indices],arrays['credit'])
        diff=float(np.max(abs(fixed-arrays['best_value']),initial=0));assert diff<1e-10;maxdiff=max(maxdiff,diff)
        assert np.max(abs(arrays['weights'].sum(1)-1),initial=0)<1e-12 and np.all(arrays['weights']>=0)
        assert np.max(abs(np.einsum('rk,kdn->rdn',arrays['weights'],bank)-arrays['credit']),initial=0)<1e-12
        assert np.all((arrays['first_step']>0)==(arrays['terminal']==1))
        assert int(arrays['oracle_count'].sum())==meta['oracle_pairs'] and int(arrays['mean_count'].sum())==meta['mean_evaluations']
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
        br=prior[rec['seed'],rec['method']];assert sha(baseline/br['arrays_file'])==br['arrays_sha256']
        with np.load(baseline/br['arrays_file']) as z:assert np.array_equal(z['indices'],indices);oldnew=z['first_step']>0
        retained=int(np.sum(oldnew&(arrays['first_step']>0)));lost=int(np.sum(oldnew&(arrays['first_step']==0)))
        counts['previous_retained']+=retained;counts['previous_lost']+=lost
        eligible=np.array([jj[i]['joint']['positive'] for i in indices]);w=arrays['weights'];entropy=-(w*np.log(np.maximum(w,np.finfo(float).tiny))).sum(1)
        rows.append(dict(seed=rec['seed'],method=rec['method'],joint_new=cr['new_proofs'],projection_new=len(positive),proofs=positive,
            strict_synergy=sum(q['strict_synergy'] for q in positive),previous_retained=retained,previous_lost=lost,
            entropy_fraction_on_joint_positive=(entropy[eligible]/np.log(len(bank))).tolist()))
        sets[rec['seed'],rec['method']]=accepted;counts['banks']+=1;counts['regions']+=len(regs);counts['old_proofs']+=int(old.sum())
        print(json.dumps(dict(banks=counts['banks'],proofs=counts['new_exact'],projection_rows=sum(q['projected_rows'] for q in projection))),flush=True)
    paired=[]
    for seed in p['seeds']:
        a=sets[seed,'alm_native'];b=sets[seed,'alm_residual'];paired.append(dict(seed=seed,native_total=len(a),residual_total=len(b),native_only=len(a-b),residual_only=len(b-a)))
    summaries=[]
    for method in p['methods']:
        rr=[r for r in rows if r['method']==method];ent=[v for r in rr for v in r['entropy_fraction_on_joint_positive']]
        summaries.append(dict(method=method,**{key:sum(r[key] for r in rr) for key in ['joint_new','projection_new','strict_synergy','previous_retained','previous_lost']},
            entropy_fraction_quantiles=np.quantile(ent,[0,.5,1]).tolist() if ent else []))
    assert counts['new_exact']==s['counts']['new_proofs'] and counts['replayed_regions']==3266 and counts['old_proofs']==732
    out=root/'results/credit_halfspace_projection/audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'proofs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'projection_checks.json').write_text(json.dumps(projection,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=counts,summaries=summaries,alm_paired_sets=paired,max_saved_objective_error=maxdiff,
        projection_checks=dict(rows=sum(q['projected_rows'] for q in projection),max_mean_error=max(q['max_mean_error'] for q in projection),
            max_kkt_range=max(q['max_kkt_range'] for q in projection),min_pythagorean_slack=min(q['min_pythagorean_slack'] for q in projection),
            min_progress_slack=min(q['min_progress_slack'] for q in projection)),source_sha256=sha(Path(__file__)),
        input_sha256={name:sha(inp/name) for name in ['protocol.json','banks.json','summary.json']},
        proofs_sha256=sha(out/'proofs.json'),projection_checks_sha256=sha(out/'projection_checks.json'),ceiling_audit_sha256=sha(ca),
        scope='All root paths checked with local feasible references, not a proof of a common global margin; all exclusions independently exact')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
