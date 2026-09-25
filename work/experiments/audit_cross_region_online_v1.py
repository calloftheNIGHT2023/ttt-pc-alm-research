"""358 independent scalar risk, exact certificate and ordered-prefix audit."""
from collections import Counter
from fractions import Fraction as F
import math
from pathlib import Path
import time
import traceback
import numpy as np
from audit_search_radius_development_v1 import read,sha,save,load,same,close,complete
from audit_stasis_escape_geometry_v1 import inequalities
from audit_post_escape_geometry_v1 import certificates
import branch_image_chain_v1 as exact

BASE='results/cross_region_online'


def run(root,out):
    begin=time.perf_counter();pred=root/BASE/'development_predictions_v1';scored=root/BASE/'development_evaluation_v1'
    complete(pred);complete(scored);p=read(pred/'protocol.json');ep=read(scored/'protocol.json');pre=root/BASE/'preflight_predictions_v1';complete(pre)
    assert p['source_sha256']==ep['source_sha256']==read(pre/'protocol.json')['source_sha256']
    for n,h in p['source_sha256'].items():assert sha(root/n)==h
    seal=read(pred/'before_query_manifest.json');assert not seal['query_targets_accessed']
    assert seal['rows_sha256']==sha(pred/'rows.json') and seal['protocol_sha256']==sha(pred/'protocol.json')
    assert ep['prediction_summary_sha256']==sha(pred/'summary.json') and ep['before_query_manifest_sha256']==sha(pred/'before_query_manifest.json')
    seeds,names=p['seeds'],p['methods'];assert seeds==list(range(328000000,328000032)) and len(names)==151
    rows=read(pred/'rows.json');index={(r['seed'],r['method']):r for r in rows};assert len(rows)==len(index)==32*151
    z=load(scored/'task_metrics.npz');metrics=['mse257','mse129','point_mse257','point_mse129']
    assert list(z['seeds'])==seeds and list(z['methods'])==names and list(z['metric_names'])==metrics
    configs={c['name']:c for c in p['configs']};assert len(configs)==14
    counts=Counter();risk=np.empty_like(z['risk']);groups=[];mechanisms=[]
    for si,seed in enumerate(seeds):
        b=np.random.default_rng(seed).uniform(-.12,.12,4);truth=[]
        for query in np.linspace(0,1,257):
            y=float(query)
            for bias in b:y=max(0.,1.-abs(2.*(y+float(bias))-1.))
            truth.append(y)
        expected=[p['configs'][int(j)]['name'] for j in np.random.default_rng(np.random.SeedSequence([358929,seed])).permutation(14)]
        rr=sorted([r for r in rows if r['seed']==seed and r['origin']=='new_358'],key=lambda r:r['order'])
        assert [r['method'] for r in rr]==expected and read(pred/str(seed)/'commit.json')['rows']==rr
        ref=p['observed_inputs'][str(seed)];assert sha(root/ref['file'])==ref['sha256'];inputs=load(root/ref['file'])
        taskmeta={};taskarrays={}
        for mi,name in enumerate(names):
            r=index[seed,name];assert sha(root/r['file'])==r['sha256'] and sha(root/r['metadata_file'])==r['metadata_sha256']
            a=load(root/r['file']);m=read(root/r['metadata_file'])['metadata'];assert not m['query_targets_accessed'] and m['execution_failed']==r['execution_failed']
            counts['input_arrays']+=same(a,inputs,['x_observed','v_observed','q_observed'])
            if name==p['primary']:groups.append('execution_failed' if m['execution_failed'] else 'support_fit' if m['selected_state']['support_fit'] else 'fallback')
            if name in configs:
                cfg=configs[name];taskmeta[name]=m;taskarrays[name]=a
                assert r['origin']=='new_358' and r['seconds']==m['charged_complete_seconds']>0
                if seed==seeds[0]:counts['preflight_arrays']+=same(a,load(pre/str(seed)/(name+'.npz')),list(a))
                if not m['execution_failed']:
                    oldr=index[seed,cfg['reference_name']];oa=load(root/oldr['file']);om=read(root/oldr['metadata_file'])['metadata']
                    assert m['visited_modes']==om['visited_modes'] and m['original_positive_modes']==om['original_positive_modes']
                    assert m['trajectory_state_sha256']==om['trajectory_state_sha256'];counts['trajectory_states']+=len(m['trajectory_state_sha256'])
                    assert m['restarts']==m['atomic_trial_points']==629 and m['prior_starts']==17
                    assert m['no_global_bp_guard_enabled']==(m['solver']!='adam' and m['channel']!='bp')
                    if 'channel' in cfg:
                        counts['invariant_arrays']+=same(a,oa,['best_bank','selected_b','point_prediction','trigger_b','trigger_h','trigger_u','trigger_credit','trigger_location'])
                        assert m['selected_state']==om['selected_state'] and set(m['original_positive_modes'])<=set(m['positive_modes'])
                        assert m['no_archived_solver_state_input'] and m['visited_source']=='Own current live trajectory, not old files or another solver'
                        regs=a['search_regions'];first=a['search_first_step'];credit=a['search_proof_credit']
                        component=root/'results/cross_region_credit/development_v1'/str(seed)
                        with np.load(component/'pool.npz',allow_pickle=False) as zz:assert regs.tobytes()==zz['regions'].tobytes()
                        assert not {r.tobytes().hex() for r in regs}&set(m['visited_modes'])
                        ids=np.flatnonzero(first==0);cap=cfg['settings'].get('geometry_budget',8)
                        if cap is not None:ids=ids[:cap]
                        selected=[regs[i].tobytes().hex() for i in ids]
                        assert selected==[q['mode'] for q in m['proposal']['proposals']]
                        assert set(selected)==set(m['new_mode_classifications_detail'])
                        by={q['index']:q for q in m['proposal']['screening']['proofs']};assert set(by)==set(np.flatnonzero(first>0))
                        for i,proof in by.items():
                            value=exact.fixed_value(a['x_observed'],a['v_observed'],credit[i],regs[i])
                            assert (proof['structural_empty'] and value is None) or value==F(proof['lower'])>0
                            assert proof['mode']==regs[i].tobytes().hex() and proof['step']==first[i]
                            if proof['kind']=='transfer':
                                source=by[proof['source_index']];assert source['kind']=='local' and source['step']<=proof['step']
                                assert credit[i].tobytes()==credit[source['index']].tobytes();counts['causal_transfers']+=1
                            counts['credit_certificates']+=1
                        mechanisms.append(dict(seed=seed,method=name,candidates_before_screen=len(regs),rejected=len(by),geometry_calls=len(selected),
                            new_positive_modes=m['new_positive_modes'],positive_modes=m['positive_modes'],pool_seconds=m['proposal']['pool']['total_seconds'],
                            screening_seconds=m['proposal']['screening']['total_seconds']))
                    else:counts['native_arrays']+=same(a,oa,list(a))
                    if m['positive_modes']:
                        yy=np.broadcast_to(a['x_observed'],(len(a['points']),4))
                        for layer in range(4):yy=np.maximum(0.,1.-abs(2*(yy+a['points'][:,layer,None])-1.))
                        assert np.max(abs(yy-a['v_observed']))<=.001+1e-7;counts['support_particles']+=len(a['points'])
                    for key,note in m['new_mode_classifications_detail'].items():
                        aa,rhs=inequalities(a['x_observed'],a['v_observed'],key);counts['geometry_certificates']+=certificates(aa,rhs,note)
            else:assert r['origin']=='frozen_pre358' and r['seconds'] is None
            values=[]
            for field in ['prediction','point_prediction']:
                assert a[field].shape==(257,) and np.isfinite(a[field]).all() and np.all((a[field]>=0)&(a[field]<=1))
                for stride in [1,2]:values.append(math.fsum((float(v)-t)**2 for v,t in zip(a[field][::stride],truth[::stride]))/len(truth[::stride]))
            risk[si,mi]=values;counts['risk_fields']+=4
        for c in ['dual','zero']:
            on,off='cross_'+c+'_reuse_g8','cross_'+c+'_independent_g8'
            if not taskmeta[on]['execution_failed'] and not taskmeta[off]['execution_failed']:
                f=taskarrays[off]['search_first_step'];g=taskarrays[on]['search_first_step'];mask=f>0
                assert np.all((g[mask]>0)&(g[mask]<=f[mask]))
                assert set(taskmeta[off]['positive_modes'])<=set(taskmeta[on]['positive_modes']);counts['positive_prefix_inclusions']+=1
        if not taskmeta['cross_c20_all']['execution_failed']:
            full=set(taskmeta['cross_c20_all']['positive_modes'])
            for name,cfg in configs.items():
                if 'channel' in cfg and not taskmeta[name]['execution_failed']:assert set(taskmeta[name]['positive_modes'])<=full;counts['full_pool_inclusions']+=1
    gap=float(np.max(abs(risk-z['risk'])));assert gap<2e-12 and groups==list(z['groups'])
    methods=read(scored/'methods.json');assert len(methods)==len(names)
    for m in methods:
        name=m['method'];j=names.index(name);rr=[index[s,name] for s in seeds]
        assert m['failures']==sum(r['execution_failed'] for r in rr)
        assert m['current_development_time_available']==(name in configs)
        if name in configs:close(m['mean_current_seconds'],math.fsum(r['seconds'] for r in rr)/len(seeds))
        else:assert m['mean_current_seconds'] is None
        for k,metric in enumerate(metrics):
            close(m['metrics'][metric],math.fsum(map(float,risk[:,j,k]))/len(seeds))
            for g in set(groups):close(m['group_metrics'][g][metric],math.fsum(float(risk[i,j,k]) for i,gg in enumerate(groups) if gg==g)/groups.count(g))
    comp=read(scored/'comparisons.json');expected={(a,b,k) for a in configs for b in names if a!=b for k in metrics}
    assert len(comp)==len(expected)==8400 and {(r['candidate'],r['control'],r['metric']) for r in comp}==expected
    for r in comp:
        delta=risk[:,names.index(r['candidate']),metrics.index(r['metric'])]-risk[:,names.index(r['control']),metrics.index(r['metric'])]
        total=math.fsum(map(float,delta));largest=int(np.argmax(abs(delta)));absolute=math.fsum(map(float,abs(delta)));n=len(seeds)
        close(r['mean_difference'],total/n);assert r['improved']==sum(delta<0) and r['equal']==sum(delta==0) and r['worse']==sum(delta>0)
        assert r['largest_absolute_task_seed']==seeds[largest];close(r['largest_absolute_task_delta'],float(delta[largest]))
        close(r['largest_absolute_share'],float(abs(delta[largest])/absolute) if absolute else 0.)
        close(r['leave_largest_absolute_out_mean'],(total-float(delta[largest]))/(n-1))
        close(r['worst_leave_one_out_mean'],max((total-float(v))/(n-1) for v in delta));counts['comparison_rows']+=1
    save(out/'mechanisms.json',mechanisms)
    result=dict(passed=True,counts=dict(counts),maximum_scalar_risk_gap=gap,seconds=time.perf_counter()-begin,
        prediction_summary_sha256=sha(pred/'summary.json'),evaluation_summary_sha256=sha(scored/'summary.json'),
        source_sha256=sha(Path(__file__)),core_research_goal_complete=False,outputs_sha256={'mechanisms.json':sha(out/'mechanisms.json')})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/BASE/'development_audit_v1';out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except Exception:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
