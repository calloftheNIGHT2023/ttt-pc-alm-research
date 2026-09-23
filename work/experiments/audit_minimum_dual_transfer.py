"""258C-D: exact bias optimization and conditional credit/bound audit."""
import argparse
from collections import Counter
from fractions import Fraction as F
import json
from pathlib import Path
import time
import numpy as np
import local_dual_jump_transfer as transfer
from audit_local_dual_jump import g,branch
from audit_local_dual_jump_transfer import optimum,key
from run_local_dual_jump_v2 import dump
from run_multiplier_fixed_point_screen import sha

def pack(z):return [str(z.numerator),str(z.denominator)]
def rational(a):return [F(float(z)) for z in a]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    parent=root/'results/minimum_sufficient_dual/transfer';out=root/'results/minimum_sufficient_dual/transfer_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    p=json.loads((parent/'protocol.json').read_text());summary=json.loads((parent/'summary.json').read_text());hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    for n in ['protocol','support_rows','geometry_files']:assert sha(parent/f'{n}.json')==summary[f'{n}_sha256']
    dump(out/'protocol.json',dict(source_sha256=hashes,parent_summary_sha256=sha(parent/'summary.json'),query_targets_accessed=False,
        oracle='independent exact 3-point bias minima; exact stationary credit and Cauchy bounds; actual binary solver error separated'))
    rows=json.loads((parent/'support_rows.json').read_text());index={(r['seed'],r['restart'],r['method']):r for r in rows};geometry={}
    for f in json.loads((parent/'geometry_files.json').read_text()):assert sha(parent/f['file'])==f['sha256'];geometry[int(f['file'].split('_')[0])]=json.loads((parent/f['file']).read_text())
    counts=Counter();exact_rows=[];credit=[];seen=set();maxgap=F(0);maxshift=F(0);begin=time.perf_counter()
    with transfer.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for row in rows:
            assert sha(parent/row['file'])==row['sha256']
            with np.load(parent/row['file']) as a:
                x,v,b,h,old=[a[k] for k in ['x','v','initial_b','initial_h','incumbent']];actual,_=transfer.run(b,h,old,x,v,row['event'],row['backend'])
                for k,value in actual.items():assert value.tobytes()==a[k].tobytes();counts['replayed_arrays']+=1
                bank=np.r_[old[None],b[None],a['trial_b']];errors=[]
                for bb in bank:
                    yy=x.copy()
                    for bias in bb:yy=np.maximum(0.,1.-np.abs(2.*(yy+bias)-1.))
                    errors.append(float(np.max(abs(yy-v))))
                order=sorted(range(len(bank)),key=lambda k:(errors[k]>.001001,.5*float(bank[k]@bank[k]) if errors[k]<=.001001 else errors[k],k));assert bank[order[0]].tobytes()==a['best'].tobytes();counts['archive_checks']+=1
                assert row['best_error']==errors[order[0]]
                kk=key(x,a['b']);pair=row['seed'],kk
                if pair not in seen:
                    result=transfer.base.branch_feasibility(x,v,a['b']);assert result['current_branch_feasible']==geometry[row['seed']][kk]['current_branch_feasible'];seen.add(pair);counts['lp_checks']+=1
                for j in range(4):
                    prev=x if j==0 else a['h'][j-1];target=a['h'][j]+a['u'][j];fun,minval,intervals=optimum(prev,target,b[j],True);gap=fun(F(float(a['b'][j])))-minval;assert F(0)<=gap<=F(2e-12);maxgap=max(maxgap,gap);counts['exact_bias_blocks']+=1;counts['exact_bias_intervals']+=intervals
                    exact_rows.append(dict(seed=row['seed'],restart=row['restart'],method=row['method'],layer=j,objective_gap=pack(gap)))
                if row['method']=='minimum_dual':
                    other=index[row['seed'],row['restart'],'minimum_activity']
                    with np.load(parent/other['file']) as ao:
                        assert a['h'].tobytes()==ao['h'].tobytes();assert not np.any(ao['u']);counts['same_activity_pairs']+=1
                        for j in range(4):
                            prev=rational(x if j==0 else a['h'][j-1]);bc,ba=F(float(a['b'][j])),F(float(ao['b'][j]));pc=[branch(z+bc) for z in prev];pa=[branch(z+ba) for z in prev];same=pc==pa
                            rec=dict(seed=row['seed'],restart=row['restart'],layer=j,same_bias_piece=same,dual_pattern=pc,activity_pattern=pa,actual_delta=pack(bc-ba));counts['credit_layers']+=1;counts['same_piece']+=same
                            interior=same and max(abs(bc),abs(ba))<F(.12) and all(z+bb not in [F(0),F(1,2),F(1)] for z in prev for bb in [bc,ba]);rec['both_strictly_interior']=interior
                            if same:
                                slopes=[[0,2,-2,0][k] for k in pc];yc=rational(a['h'][j]+a['u'][j]);ya=rational(ao['h'][j]);uu=rational(a['u'][j]);deltas=[p-q for p,q in zip(yc,ya)];rounding=[dd-u for dd,u in zip(deltas,uu)];s2=sum(s*s for s in slopes);den=s2+len(x)*F(.01)
                                shift=sum((s*dd for s,dd in zip(slopes,deltas)),F(0))/den;ideal_shift=sum((s*u for s,u in zip(slopes,uu)),F(0))/den;bound2=s2*sum(dd*dd for dd in deltas)/den**2;ideal_bound2=s2*sum(u*u for u in uu)/den**2;round_bound2=s2*sum(e*e for e in rounding)/den**2
                                assert shift**2<=bound2 and ideal_shift**2<=ideal_bound2 and (shift-ideal_shift)**2<=round_bound2;counts['exact_cauchy_bounds']+=3
                                rec.update(actual_target_shift=pack(shift),ideal_u_shift=pack(ideal_shift),target_rounding_shift=pack(shift-ideal_shift),target_bound2=pack(bound2),ideal_u_bound2=pack(ideal_bound2),target_rounding_bound2=pack(round_bound2))
                                if interior:
                                    offsets=[g(z+bc)-s*bc for z,s in zip(prev,slopes)];anchor=len(x)*F(.01)*F(float(b[j]));cs=(sum(s*(yy-cc) for s,yy,cc in zip(slopes,yc,offsets))+anchor)/den;aa=(sum(s*(yy-cc) for s,yy,cc in zip(slopes,ya,offsets))+anchor)/den
                                    assert cs-aa==shift;actual_error=bc-ba-shift;assert abs(actual_error)<F(1e-12);maxshift=max(maxshift,abs(actual_error));residual_bound=abs(bc-cs)+abs(ba-aa);assert max(F(0),abs(bc-ba)-residual_bound)**2<=bound2
                                    rec.update(exact_stationary_delta=pack(cs-aa),actual_solver_shift_error=pack(actual_error),solver_rounding_bound=pack(residual_bound));counts['interior_credit_identities']+=1;counts['actual_write_bounds']+=1
                            credit.append(rec)
                counts['method_states']+=1
            if counts['method_states']%34==0:print(json.dumps(dict(seed=row['seed'],counts=counts)),flush=True)
    assert counts['method_states']==544 and counts['exact_bias_blocks']==2176 and counts['same_activity_pairs']==272
    dump(out/'exact_bias_gaps.json',exact_rows);dump(out/'credit.json',credit);ans=dict(passed=True,counts=counts,max_exact_bias_objective_gap=float(maxgap),max_interior_actual_shift_error=float(maxshift),seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),exact_bias_gaps_sha256=sha(out/'exact_bias_gaps.json'),credit_sha256=sha(out/'credit.json'),query_targets_accessed=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()
