"""316 independent rank/witness and three-point quadratic block-energy audit."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
from multiplier_fixed_point_exact import rref_solve
from effective_affine_map_v1 import restore, evaluate
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def g(z):return max(F(0),F(1)-abs(2*z-1))
def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))


def minimize_by_energy(breaks, energy):
    """Reconstruct each scalar quadratic from values, not the solver formulas."""
    breaks=sorted(set(breaks));candidates=[]
    if len(breaks)==1:return breaks[0],1
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        middle=(lo+hi)/2;el,em,eh=energy(lo),energy(middle),energy(hi)
        aa=2*(el+eh-2*em)/(hi-lo)**2
        bb=(eh-el)/(hi-lo)-aa*(hi+lo)
        assert aa>0
        vertex=min(hi,max(lo,-bb/(2*aa)))
        candidates.append((energy(vertex),vertex))
    return min(candidates,key=lambda t:t[0])[1],len(candidates)


def independent_check(point,d,x,v,method):
    n=len(x);b=point[:d];h=[point[d+j*n:d+(j+1)*n] for j in range(d)]
    u=[point[d+d*n+j*n:d+d*n+(j+1)*n] for j in range(d)]
    B,T,E=F(.12),F(.01),F(.001);knots=[F(0),F(1,2),F(1)]
    domain=all(-B<=t<=B for t in b)
    domain=domain and all((max(F(0),v[i]-E) if j==d-1 else 0)<=h[j][i]<=
                         (min(F(1),v[i]+E) if j==d-1 else 1) for j in range(d) for i in range(n))
    before=x[:];modes=[];residual=[]
    for j in range(d):
        modes.extend(sum(z+b[j]>=k for k in knots) for z in before)
        before=[g(z+b[j]) for z in before]
        prev=x if j==0 else h[j-1]
        residual.extend(h[j][i]-g(prev[i]+b[j]) for i in range(n))
    errors=[p-y for p,y in zip(before,v)]
    metrics=dict(mode=''.join(map(str,modes)),forward=before,residual_max=max(map(abs,residual)),
                 support_mse=dot(errors,errors)/n,support_max_error=max(map(abs,errors)),
                 support_band_feasible=all(abs(e)<=E for e in errors))
    if not domain:return domain,False,metrics,0,0
    blocks,intervals=0,0;all_same=True
    for j in reversed(range(d)):
        for i in range(n):
            prev=x[i] if j==0 else h[j-1][i]
            a=g(prev+b[j])-u[j][i];old=h[j][i]
            if j==d-1:
                breaks=[max(F(0),v[i]-E),min(F(1),v[i]+E)]
                energy=lambda z:(z-a)**2+T*(z-old)**2
            else:
                nb=b[j+1];target=h[j+1][i]+u[j+1][i]
                breaks=[F(0),F(1)]+[k-nb for k in knots if 0<k-nb<1]
                energy=lambda z:(z-a)**2+(g(z+nb)-target)**2+T*(z-old)**2
            best,k=minimize_by_energy(breaks,energy)
            all_same=all_same and best==old;blocks+=1;intervals+=k
    for j in range(d):
        prev=x if j==0 else h[j-1];target=[h[j][i]+u[j][i] for i in range(n)]
        breaks=[-B,B]+[k-p for p in prev for k in knots if -B<k-p<B]
        energy=lambda z:sum(((g(p+z)-t)**2 for p,t in zip(prev,target)),F(0))/n+T*(z-b[j])**2
        best,k=minimize_by_energy(breaks,energy)
        all_same=all_same and best==b[j];blocks+=1;intervals+=k
    fixed=all_same and (method=='nodual' or not any(residual))
    return domain,fixed,metrics,blocks,intervals


def run(root,out):
    start=time.perf_counter();source=root/'results/affine_root_proposal/development_v1'
    summary=read(source/'summary.json');assert summary['passed']
    manifest=read(source/'before_aggregation_manifest.json')
    assert sha(source/'before_aggregation_manifest.json')==summary['manifest_sha256']
    for name,digest in manifest['files_sha256'].items():assert sha(source/name)==digest
    for name,digest in read(source/'protocol.json')['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest
    # Independent energy interpolator has known dyadic fixed and nonfixed cases.
    assert minimize_by_energy([F(-1),F(1)],lambda z:(z-F(1,3))**2)[0]==F(1,3)
    assert independent_check([F(1,16),F(5,8),F(0)],1,[F(1,4)],[F(5,8)],'alm')[1]
    assert not independent_check([F(0),F(5,8),F(0)],1,[F(1,4)],[F(5,8)],'alm')[1]
    counts=Counter();aggregate={family:dict(counts=Counter(),rank=Counter(),seconds=Counter(),
                                          max_rational_bits=0,float_root_gaps=[],fixed_examples=[])
                               for family in ['alm_keep','alm_reset','nodual']}
    for entry in read(source/'rows.json'):
        r=read(source/entry['file']);rows=restore(r['formula']);point=list(map(F,r['point']));sol=r['solution']
        x,v=list(map(F,r['x'])),list(map(F,r['v']));n=len(x);d=len(point)//(1+2*n);dim=d+d*n
        active=list(range(len(point))) if r['method']=='alm' else list(range(dim))
        assert active==sol['active']
        a=[[F(i==j)-rows[i].terms.get(j,F(0)) for j in active] for i in active]
        rhs=[rows[i].terms.get(-1,F(0)) for i in active]
        solved,meta=rref_solve(a,rhs,[point[j] for j in active])
        assert meta['rank']==sol['rank'] and (solved is not None)==sol['consistent']
        group=aggregate[r['family']];group['counts']['cases']+=1;group['rank'][sol['rank']]+=1
        group['max_rational_bits']=max(group['max_rational_bits'],sol['max_rational_bits'])
        group['seconds'].update(r['seconds'])
        if solved is None:
            w=list(map(F,sol['certificate']))
            assert all(dot(w,[row[j] for row in a])==0 for j in range(len(active)))
            assert dot(w,rhs)==1
            for k in range(3):
                probe=[F(k-j,19) for j in range(len(point))]
                if r['method']=='nodual':probe[dim:]=[F(0)]*(d*n)
                predicted=evaluate(rows,probe)
                assert dot(w,[predicted[j]-probe[j] for j in active])==1
                counts['exact_drift_probes']+=1
            status='no_fixed_point_in_selected_formula';counts['left_null_certificates']+=1
            group['counts']['w_has_multiplier_component']+=r['w_has_multiplier_component']
        else:
            assert solved==list(map(F,sol['root']))
            candidate=list(map(F,sol['full_root']));assert evaluate(rows,candidate)==candidate
            if r['method']=='nodual':assert not any(candidate[dim:])
            for vector in sol['nullspace']:
                vector=list(map(F,vector));assert all(dot(row,vector)==0 for row in a)
                counts['nullspace_vectors']+=1
            domain,fixed,metrics,blocks,intervals=independent_check(candidate,d,x,v,r['method'])
            counts['energy_blocks']+=blocks;counts['energy_intervals']+=intervals
            checked=r['validation'];assert domain==checked['domain_valid'] and fixed==checked['actual_fixed_point']
            assert metrics['mode']==checked['metrics']['mode']
            for key in ['residual_max','support_mse','support_max_error']:
                assert metrics[key]==F(checked['metrics'][key])
            assert metrics['forward']==list(map(F,checked['metrics']['forward']))
            assert metrics['support_band_feasible']==checked['metrics']['support_band_feasible']
            status='actual_fixed_point' if fixed else 'root_wrong_actual_update' if domain else 'root_outside_domain'
            if 'float_candidate_fixed_gap' in r:group['float_root_gaps'].append(r['float_candidate_fixed_gap'])
            group['counts']['consistent']+=1;group['counts']['domain_valid']+=domain
            if fixed:
                before=r['original_metrics'];group['counts']['fixed_support_feasible']+=metrics['support_band_feasible']
                group['counts']['fixed_zero_residual']+=metrics['residual_max']==0
                group['counts']['fixed_bias_changed']+=r['bias_changed']
                group['counts']['fixed_support_mse_improved']+=metrics['support_mse']<F(before['support_mse'])
                group['counts']['newly_support_feasible']+=metrics['support_band_feasible'] and not before['support_band_feasible']
                group['fixed_examples'].append(dict(seed=r['seed'],location=r['location'],
                                                   support_mse_before=before['support_mse'],support_mse_after=metrics['support_mse'],
                                                   support_feasible=metrics['support_band_feasible'],mode=metrics['mode']))
            counts['root_proposals']+=1
        assert status==r['status']==entry['status'];group['counts'][status]+=1
        counts['cases']+=1
    assert counts['cases']==393
    for g in aggregate.values():
        g['mean_total_core_seconds']=sum(g['seconds'].values())/g['counts']['cases']
        g['maximum_float_root_gap']=max(g['float_root_gaps'],default=None)
        del g['float_root_gaps']
    save(out/'aggregate.json',aggregate)
    final=dict(passed=True,counts=dict(counts),seconds=time.perf_counter()-start,
               input_summary_sha256=sha(source/'summary.json'),aggregate_sha256=sha(out/'aggregate.json'),
               source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'multiplier_fixed_point_exact.py']},
               query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',final)
    print(final,flush=True)
    for k,g in aggregate.items():print(k,{key:value for key,value in g.items() if key!='fixed_examples'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
