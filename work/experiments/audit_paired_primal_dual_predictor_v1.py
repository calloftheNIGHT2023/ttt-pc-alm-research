"""312 independent predictor construction, batched replay, geometry/set audit."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path
import time
import traceback

import numpy as np
import scipy.optimize as opt
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha
import complete_credit_mode_geometry_v1 as geometry

NAMES=['normal_one','normal_two','paired_bu','bias_only','dual_only','dual_full','full_bhu',
       'paired_b_zero_u','paired_b_last_du','paired_b_random_du','random_db_paired_u','paired_b_current_residual_u']


def alpha_of(b,d):
    limits=[F(1)]
    for bi,di in zip(b,d):
        bi,di=F(float(bi)),F(float(di))
        if di:limits.append(((F(.12) if di>0 else -F(.12))-bi)/di)
    exact=min(limits);alpha=float(exact)
    if F(alpha)>exact:alpha=float(np.nextafter(alpha,0.))
    return alpha,exact


def tent(x):return np.maximum(0,1-np.abs(2*x-1))


def run(root,out):
    begin=time.perf_counter();source=root/'results/paired_primal_dual_predictor/development_v1'
    evaluated=root/'results/paired_primal_dual_predictor/geometry_v1'
    counts=Counter(); sources={Path(__file__).name}
    for folder in [source,evaluated]:
        summary=read(folder/'summary.json');assert summary['passed']
        for n,digest in summary['outputs_sha256'].items():assert sha(folder/n)==digest
        for n,digest in read(folder/'protocol.json')['source_sha256'].items():
            assert sha(root/'work/experiments'/n)==digest;sources.add(n)
    geometric_states={(s['seed'],*s['location']):s for s in read(evaluated/'states.json')}
    all_sets=defaultdict(lambda:defaultdict(set));diags=[];guarded=[]
    def prohibited(*args,**kwargs):raise AssertionError('Global optimizer/BP during candidate audit')
    try:
        for obj,n in [(cold.bp,'evaluate'),(cold.bp,'refine'),(opt,'linprog'),(opt,'minimize')]:
            guarded.append((obj,n,getattr(obj,n)));setattr(obj,n,prohibited)
        with discovery_box(.12):
            for row in read(source/'rows.json'):
                p=read(source/row['file']);seed=p['seed'];key=[seed,*p['location']]
                b,h,u=[np.array(p['current'][k]) for k in ['b','h','u']]
                db,dh,du=[np.array(p['current'][k])-np.array(p['previous'][k]) for k in ['b','h','u']]
                x,v=np.array(p['x_observed']),np.array(p['v_observed'])
                alpha,exact=alpha_of(b,db)
                assert alpha==p['metadata']['alpha'] and exact==F(p['metadata']['exact_alpha'])
                su=np.random.default_rng(np.random.SeedSequence([312901,*key,0])).choice([-1.,1.],u.shape)
                sb=np.random.default_rng(np.random.SeedSequence([312901,*key,1])).choice([-1.,1.],b.shape)
                ra,rexact=alpha_of(b,db*sb)
                assert ra==p['metadata']['random_alpha'] and rexact==F(p['metadata']['random_exact_alpha'])
                pb=np.clip(b+alpha*db,-.12,.12);ph=np.clip(h+alpha*dh,0,1)
                ph[-1]=np.clip(ph[-1],np.maximum(0,v-.001),np.minimum(1,v+.001))
                residual=np.array([h[j]-tent((x if j==0 else h[j-1])+b[j]) for j in range(4)])
                starts=[(b,h,u),(b,h,u),(pb,h,u+alpha*du),(pb,h,u),(b,h,u+alpha*du),
                    (b,h,u+du),(pb,ph,u+alpha*du),(pb,h,np.zeros_like(u)),(pb,h,alpha*du),
                    (pb,h,u+alpha*du*su),(np.clip(b+ra*db*sb,-.12,.12),h,u+ra*du),
                    (pb,h,u+alpha*.5*residual)]
                assert [m['method'] for m in p['methods']]==NAMES
                for m,initial in zip(p['methods'],starts):
                    for k,a in zip(['b','h','u'],initial):
                        assert np.array_equal(m['states'][0][k],a),(key,m['method'],k)
                        counts['predictor_array_checks']+=1
                bb=np.array([s[0] for s in starts]);hh=np.stack([s[1] for s in starts],axis=1)
                uu=np.stack([s[2] for s in starts],axis=1)
                batch=cold.Local(bb,x,v,'alm');batch.h=hh.copy();batch.u=uu.copy();batch.step()
                for i,m in enumerate(p['methods']):
                    for k,a in [('b',batch.b[i]),('h',batch.h[:,i]),('u',batch.u[:,i])]:
                        assert np.array_equal(m['states'][1][k],a),(key,m['method'],k,'batch')
                        counts['batched_output_array_checks']+=1
                b2=cold.Local(batch.b[1:2],x,v,'alm');b2.h=batch.h[:,1:2].copy();b2.u=batch.u[:,1:2].copy();b2.step()
                for k,a in [('b',b2.b[0]),('h',b2.h[:,0]),('u',b2.u[:,0])]:
                    assert np.array_equal(p['methods'][1]['states'][2][k],a)
                    counts['normal_second_step_array_checks']+=1
                gm=geometric_states[tuple(key)];actual_errors={}
                for m in p['methods']:
                    errors=[]
                    for j,(s,mode) in enumerate(zip(m['states'],m['point_modes'])):
                        value=x.copy();codes=[]
                        for bias in s['b']:
                            z=value+bias;codes.extend(int(t>=0)+int(t>=.5)+int(t>=1) for t in z);value=tent(z)
                        assert bytes(codes).hex()==mode
                        counts['independent_float_forward_replays']+=1
                        errors.append(float(np.max(abs(value-v))))
                        # The actual point, not an LP witness from its branch.
                        _,aa,rr,_,_=geometry.matrices(x,v,mode)
                        slacks=[rhs-sum((a*F(bias) for a,bias in zip(ar,s['b'])),F(0)) for ar,rhs in zip(aa,rr)]
                        actual=gm['actual_points'][m['method']][j]
                        assert actual['closed_feasible']==all(s>=0 for s in slacks)
                        assert F(actual['minimum_slack'])==min(slacks)
                        counts['independent_point_feasibility_checks']+=1
                        all_sets[seed][m['method']].add(mode)
                    actual_errors[m['method']]=errors
                byname={m['method']:m for m in p['methods']}
                paired=np.array(byname['paired_bu']['states'][-1]['b'])
                counts['zero_db_states']+=bool(np.all(db==0))
                counts['db_at_most_1e_minus_12_states']+=bool(np.max(abs(db))<=1e-12)
                counts['paired_final_b_equal_bias_only']+=np.array_equal(paired,byname['bias_only']['states'][-1]['b'])
                counts['paired_final_b_equal_dual_only']+=np.array_equal(paired,byname['dual_only']['states'][-1]['b'])
                counts['paired_final_b_equal_normal_one']+=np.array_equal(paired,byname['normal_one']['states'][-1]['b'])
                diags.append(dict(seed=seed,location=p['location'],db_max=float(np.max(abs(db))),
                    dh_max=float(np.max(abs(dh))),du_max=float(np.max(abs(du))),
                    residual_max=float(np.max(abs(residual))),
                    paired_vs_normal_b_gap=float(np.max(abs(paired-np.array(byname['normal_one']['states'][-1]['b'])))),
                    actual_support_max_errors=actual_errors))
    finally:
        for obj,n,val in guarded:setattr(obj,n,val)
    recomputed={n:Counter() for n in NAMES};tasks=read(evaluated/'tasks.json')
    assert len(tasks)==64 and len(diags)==131
    for task in tasks:
        data=read(evaluated/task['file']);seed=task['seed'];x=np.array(data['x_observed']);v=np.array(data['v_observed'])
        positive=set()
        for mode,result in data['geometry'].items():
            _,aa,rr,_,_=geometry.matrices(x,v,mode)
            counts['exact_certificate_rechecks']+=geometry.verify_certificates(aa,rr,result)
            if result['positive_volume_certified']:positive.add(mode)
        old=set(data['original_positive_modes']);prior=set(data['all_308_309_positive_modes'])
        for n,m in data['methods'].items():
            proposed=all_sets[seed][n];pp=proposed&positive;new=pp-old
            assert proposed==set(m['proposal_modes']) and pp==set(m['positive_modes'])
            assert new==set(m['new_positive_modes']) and old|pp==set(m['retained_pool'])
            assert not m['unknown_modes']
            counts['method_set_checks']+=1
            r=recomputed[n];r['tasks_with_new_positive']+=bool(new);r['task_positive_pairs']+=len(new)
            r['new_pairs_vs_all_308_309']+=len(pp-prior);r['unknown_pairs']+=0
            r['numeric_new_volume']+=sum(data['geometry'][mm]['volume'] for mm in sorted(new) if data['geometry'][mm]['volume'] is not None)
    for n,a in read(evaluated/'aggregate.json').items():
        for field,value in a.items():assert value==recomputed[n][field];counts['aggregate_number_checks']+=1
    norms={k:dict(minimum=min(d[k] for d in diags),median=float(np.median([d[k] for d in diags])),
        maximum=max(d[k] for d in diags)) for k in ['db_max','dh_max','du_max','residual_max','paired_vs_normal_b_gap']}
    save(out/'state_diagnostics.json',diags)
    final=dict(passed=True,counts=dict(counts),state_norms=norms,
        source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(sources)},
        input_summaries_sha256={str(f.relative_to(root)):sha(f/'summary.json') for f in [source,evaluated]},
        outputs_sha256={'state_diagnostics.json':sha(out/'state_diagnostics.json')},
        query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False,
        seconds=time.perf_counter()-begin)
    save(out/'summary.json',final);print({k:v for k,v in final.items() if k not in ['source_sha256','input_summaries_sha256','outputs_sha256']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
