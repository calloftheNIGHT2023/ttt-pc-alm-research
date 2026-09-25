"""Exact certificate replay, independent LP, and joint-direction witnesses.

Cross-bank tests are post-run explanatory diagnostics, not changed deployment
decisions or a claim that one learner was given another learner's trajectory.
"""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import light_h2_credit as memory


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def bound_value(x,v,reg,a):
    result=memory.certificate.exact_optimum(x,v,reg,a)
    return F(int(result['numerator']),int(result['denominator'])),result


def relaxed_lp(x,v,reg,a):
    d,n=reg.shape;screen=memory.certificate.screen;zl,zh,hl,hh=screen.boxes(v,reg[None])
    eq=np.zeros((d*n,d+2*d*n));rhs=np.zeros(d*n)
    for j in range(d):
        for i in range(n):
            row=j*n+i;eq[row,j]=-1;eq[row,d+row]=1
            if j:eq[row,d+d*n+(j-1)*n+i]=-1
            else:rhs[row]=x[i]
    objective=np.r_[np.zeros(d),(-memory.base.SLOPES[reg]*a).ravel(),a.ravel()]
    bounds=[(-screen.B,screen.B)]*d+list(zip(zl.ravel(),zh.ravel()))+list(zip(hl.ravel(),hh.ravel()))
    lp=linprog(objective,A_eq=eq,b_eq=rhs,bounds=bounds,options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    value=None if not lp.success else float(lp.fun-(a*memory.base.INTERCEPTS[reg]).sum())
    return lp,value


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/light_h2_credit/development';out=root/'results/light_h2_credit/certificate_audit'
    p=json.loads((inp/'protocol.json').read_text());assert json.loads((inp/'run_audit.json').read_text())['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    checks=dict(details=0,exact_certificates=0,full_region_lp_infeasible=0,independent_relaxed_lp=0,
        cross_bank_exact_directions=0);summaries=[];witnesses=[];maximum_lp_error=0.
    for seed in p['seeds']:
        details={}
        for cfg in [c for c in p['configs'] if c['family']!='regression']:
            path=inp/f'detail_{seed}_{cfg["name"]}.json';detail=json.loads(path.read_text());details[cfg['name']]=detail;checks['details']+=1
            with np.load(inp/f'state_{seed}_{cfg["name"]}_0_4.npz') as z:x=z['x'];v=z['v']
            bank=np.array(detail['credit_bank']).reshape(-1,4,len(x))
            assert len(bank)<=32
            for proof in detail['credit_proofs']:
                reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,len(x));a=np.array(proof['a'])
                assert np.array_equal(a,bank[proof['direction']])
                value,exact=bound_value(x,v,reg,a);assert value>0 and exact==proof['exact'];checks['exact_certificates']+=1
                _,_,g,rhs=memory.neighbor.pattern_matrix(x,v,reg)
                lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4,options={'primal_feasibility_tolerance':1e-9})
                assert lp.status==2,(seed,cfg['name'],lp.status);checks['full_region_lp_infeasible']+=1
                lp,optimum=relaxed_lp(x,v,reg,a)
                if 'empty_layer' in exact:assert lp.status==2
                else:
                    assert lp.success;error=abs(float(value)-optimum);assert error<1e-8
                    maximum_lp_error=max(maximum_lp_error,error)
                checks['independent_relaxed_lp']+=1
            summaries.append(dict(seed=seed,method=cfg['name'],proofs=len(detail['credit_proofs']),
                exact_checks=sum(c['credit']['exact_checks'] for c in detail['screen_calls']),detail_sha256=sha(path)))
        # Is there a concrete mixed local credit direction unavailable to the
        # fixed tested alternatives? Test every local certificate, not only a
        # favorable hand-picked example. No query data enters these bounds.
        local=details['alm_native']
        for proof in local['credit_proofs']:
            reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,4);alternatives={}
            for name in ['alm_axes','alm_random','alm_residual','adam60_native']:
                values=[]
                for a in np.array(details[name]['credit_bank']).reshape(-1,4,4):
                    value,meta=bound_value(x,v,reg,a);values.append(value);checks['cross_bank_exact_directions']+=1
                maximum=max(values) if values else None
                alternatives[name]=dict(directions=len(values),has_positive=bool(maximum is not None and maximum>0),
                    maximum_exact=None if maximum is None else str(maximum))
            witnesses.append(dict(seed=seed,pattern=proof['pattern'],a=proof['a'],local_exact=proof['exact'],alternatives=alternatives,
                beats_all_coordinate_tests=not alternatives['alm_axes']['has_positive'],
                independent_of_all_tested_banks=not any(r['has_positive'] for r in alternatives.values())))
        print(json.dumps(dict(seed=seed,**checks)),flush=True)
    result=dict(passed=True,checks=checks,maximum_relaxed_lp_error=maximum_lp_error,
        local_proofs=len(witnesses),joint_vs_axes_witnesses=sum(w['beats_all_coordinate_tests'] for w in witnesses),
        local_exclusive_of_all_tested_banks=sum(w['independent_of_all_tested_banks'] for w in witnesses),
        per_method=[dict(method=c['name'],proofs=sum(r['proofs'] for r in summaries if r['method']==c['name'])) for c in p['configs'] if c['family']!='regression'],
        relaxed_lp_primal_and_dual_tolerance=1e-9,source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),
        scope='Exact rational sufficiency plus numeric LP corroboration; alternatives are finite tested banks, not all possible BP/geometry directions; cross-bank tests are explanatory and untimed')
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('summary.json',result),('records.json',summaries),('joint_witnesses.json',witnesses)]:
        (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
