"""318 independent full sequential step replay and raw-energy witnesses."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
from affine_root_validation_v1 import state,domain,exact_step
from audit_affine_root_proposal_v1 import g
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def independent_blocks(s):
    d,n=len(s['b']),len(s['x']);out=[];knots=[F(0),F(1,2),F(1)]
    for j in range(d-1,-1,-1):
        for i in range(n):
            lo,hi=(max(F(0),s['v'][i]-s['eps']),min(F(1),s['v'][i]+s['eps'])) if j==d-1 else (F(0),F(1))
            cuts={lo,hi}
            if j+1<d:cuts|={k-s['b'][j+1] for k in knots if lo<k-s['b'][j+1]<hi}
            out.append(('h',j,i,s['h'][j][i],sorted(cuts)))
    for j in range(d):
        prev=s['x'] if j==0 else s['h'][j-1];B=s['bound']
        out.append(('b',j,None,s['b'][j],sorted({-B,B}|{k-p for k in knots for p in prev if -B<k-p<B})))
    return out


def raw_energy(s,delta,block,z,t):
    kind,j,i,old,_=block;d,n=len(s['b']),len(s['x']);start=d+d*n
    uu=[[s['direction'][l][k]+t*delta[start+l*n+k] for k in range(n)] for l in range(d)]
    if kind=='h':
        prev=s['x'][i] if j==0 else s['h'][j-1][i]
        errors=[z-g(prev+s['b'][j])+uu[j][i]]
        if j+1<d:errors.append(s['h'][j+1][i]-g(z+s['b'][j+1])+uu[j+1][i])
        return sum((e*e for e in errors),F(0))+s['trust']*(z-old)**2
    prev=s['x'] if j==0 else s['h'][j-1]
    errors=[h-g(p+z)+u for h,p,u in zip(s['h'][j],prev,uu[j])]
    return sum((e*e for e in errors),F(0))/n+s['trust']*(z-old)**2


def run(root,out):
    start=time.perf_counter();source=root/'results/primal_stasis_escape/development_v1'
    parent=root/'results/invariant_dual_drift/development_v1'
    summary=read(source/'summary.json');assert summary['passed'];manifest=read(source/'before_geometry_manifest.json')
    assert sha(source/'before_geometry_manifest.json')==summary['manifest_sha256']
    for n,digest in manifest['files_sha256'].items():assert sha(source/n)==digest
    for n,digest in read(source/'protocol.json')['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    groups={f:dict(counts=Counter(),exit_phase_histogram=Counter(),seconds=Counter(),max_float_gap=0.,exit_tasks=set())
            for f in ['alm_keep','alm_reset','nodual']};checks=Counter();exits=[]
    for entry in read(source/'rows.json'):
        r=read(source/entry['file']);result=r['result'];old=read(parent/r['source_file']);p=old['proposal']
        group=groups[r['family']];group['counts']['cases']+=1;group['counts'][result['status']]+=1
        group['seconds']['incremental_exact_escape']+=r['incremental_exact_seconds']
        group['seconds']['prior_line_generation_core']+=sum(old['seconds'].values())
        if not p['consistent']:
            assert result['status']=='no_restricted_invariant_line';continue
        q,delta=list(map(F,p['q'])),list(map(F,p['d']));x,v=list(map(F,old['x'])),list(map(F,old['v']))
        n=len(x);depth=len(q)//(1+2*n);dim=depth+depth*n;s=state(q,depth,x,v)
        if result['status']=='q_outside_domain':assert domain(s);continue
        if result['status']=='drift_not_actual_dual_increment':
            rr=[]
            for j,b in enumerate(s['b']):
                prev=x if j==0 else s['h'][j-1]
                rr.extend((F(1,2) if old['method']=='alm' else 0)*(h-g(z+b)) for h,z in zip(s['h'][j],prev))
            assert rr!=delta[dim:];continue
        if result['status']=='phase_zero_not_primal_stationary':
            actual,_=exact_step(s,old['method']);assert actual[:dim]!=q[:dim]
            checks['rejected_phase_zero_full_steps']+=1;continue
        assert result['applicable'] and not domain(s)
        bs=independent_blocks(s);expected={(i,z) for i,b in enumerate(bs) for z in b[4]}
        got={(w['block'],F(w['candidate'])) for w in result['endpoint_witnesses']};assert got==expected
        for w in result['endpoint_witnesses']:
            b=bs[w['block']];z=F(w['candidate']);oldz=b[3]
            e0=raw_energy(s,delta,b,z,F(0))-raw_energy(s,delta,b,oldz,F(0))
            e1=raw_energy(s,delta,b,z,F(1))-raw_energy(s,delta,b,oldz,F(1))
            slope=e1-e0;assert e0==F(w['delta'])>=0 and slope==F(w['beta'])
            if slope<0:
                upper=w['strict_integer_exit_upper_bound'];assert e0+upper*slope<0
                assert e0+(upper-1)*slope>=0
            checks['independent_endpoint_witnesses']+=1
        first=result.get('first_exit_input_phase')
        horizon=first if first is not None else 64
        actual_last=None
        for t in range(horizon+1):
            current=[z+t*e for z,e in zip(q,delta)];actual,_=exact_step(state(current,depth,x,v),old['method'])
            expected=[z+(t+1)*e for z,e in zip(q,delta)]
            assert (actual==expected)==(first is None or t<first),(r['family'],r['seed'],r['location'],t,result['status'])
            checks['independent_full_sequential_steps']+=1;actual_last=actual
        if first is not None:
            assert first>=1 and result['first_changed_update_number']==first+1
            assert actual_last==list(map(F,result['exit_output']))
            group['exit_phase_histogram'][first]+=1;group['exit_tasks'].add(r['seed'])
            group['max_float_gap']=max(group['max_float_gap'],r['floating_vs_exact_max_gap'])
            group['counts']['exact_float_forward_mode_differences']+=not r['exact_float_forward_mode_match']
            group['counts']['exit_exact_support_feasible']+=result['exit_metrics']['support_band_feasible']
            group['counts']['exit_float_support_feasible']+=r['floating_exit_metrics']['support_band_feasible']
            group['seconds']['floating_exit_validation']+=r['floating_exit_seconds']
            exits.append(dict(seed=r['seed'],family=r['family'],location=r['location'],phase=first,
                              file=entry['file'],exact_mode=result['exit_metrics']['mode'],
                              float_mode=r['floating_exit_metrics']['mode']))
        elif result['status']=='primal_stasis_for_all_nonnegative_phases':
            assert all(F(w['beta'])>=0 for w in result['endpoint_witnesses'])
        else:assert result['status']=='finite_exit_beyond_cap' and result['finite_exit_witness']['strict_integer_exit_upper_bound']>64
        group['counts']['applicable']+=1
    for group in groups.values():group['exit_tasks']=sorted(group['exit_tasks'])
    save(out/'aggregate.json',groups);save(out/'exits.json',exits)
    final=dict(passed=True,checks=dict(checks),seconds=time.perf_counter()-start,
               input_summary_sha256=sha(source/'summary.json'),aggregate_sha256=sha(out/'aggregate.json'),
               exits_sha256=sha(out/'exits.json'),source_sha256=sha(Path(__file__)),
               query_targets_accessed=False,geometry_evaluated=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',final);print(final,flush=True)
    print(groups,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
