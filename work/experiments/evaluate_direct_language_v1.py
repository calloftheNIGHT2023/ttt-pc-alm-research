"""365 sealed scoring plus independent scalar-risk/geometry/mode-pool audit."""
from collections import Counter
import math
from pathlib import Path
import time
import traceback
import numpy as np
import run_direct_language_v1 as suite
from analyze_recovered_online_comparison import forward
from evaluate_search_radius_development_v1 import contrast,METRICS
from audit_stasis_escape_geometry_v1 import inequalities
from audit_post_escape_geometry_v1 import certificates


def run(root,out):
    begin=time.perf_counter();pred=root/suite.BASE/'development_predictions_v1';summary=suite.complete(pred)
    p=suite.read(pred/'protocol.json');assert p['source_sha256']==suite.gate(root)
    assert p['seeds']==suite.SEEDS and p['methods']==suite.NAMES
    assert suite.verify(root,pred,p)==summary['checks']
    seal=suite.read(pred/'before_query_manifest.json')
    assert not seal['query_targets_accessed'] and seal['rows_sha256']==suite.sha(pred/'rows.json') and seal['protocol_sha256']==suite.sha(pred/'protocol.json')
    suite.save(out/'protocol.json',dict(prediction_summary_sha256=suite.sha(pred/'summary.json'),
        prediction_seal_sha256=suite.sha(pred/'before_query_manifest.json'),source_sha256=p['source_sha256'],
        primary=p['primary'],query_targets_accessed=True,new_blind_tasks=False,descriptive_only=True))
    rows=suite.read(pred/'rows.json');index={(r['seed'],r['method']):r for r in rows};risk=np.empty((32,4,4));checks=Counter();tasknotes=[]
    maxgap=0.;maxreadgap=0.
    for i,seed in enumerate(suite.SEEDS):
        teacher=np.random.default_rng(seed).uniform(-.12,.12,4);q=np.linspace(0,1,257);truth=forward(q,teacher[None])[0]
        scalar=[]
        for t in q:
            y=float(t)
            for b in teacher:y=max(0.,1.-abs(2.*(y+float(b))-1.))
            scalar.append(y)
        met={};arrays={}
        for j,name in enumerate(suite.NAMES):
            r=index[seed,name];a=suite.load(root/r['file']);m=suite.read(root/r['metadata_file'])['metadata'];met[name]=m;arrays[name]=a
            for k,(field,stride) in enumerate((f,s) for f in ['prediction','point_prediction'] for s in [1,2]):
                risk[i,j,k]=np.mean((a[field][::stride]-truth[::stride])**2)
                independent=math.fsum((float(v)-t)**2 for v,t in zip(a[field][::stride],scalar[::stride]))/len(scalar[::stride])
                maxgap=max(maxgap,abs(independent-risk[i,j,k]));checks['scalar_risks']+=1
            if m['positive_modes']:
                yy=np.broadcast_to(a['x_observed'],(len(a['points']),4))
                for layer in range(4):yy=np.maximum(0.,1.-abs(2.*(yy+a['points'][:,layer,None])-1.))
                assert np.max(abs(yy-a['v_observed']))<=.001+1e-7;checks['support_particles']+=len(a['points'])
            for key,note in m['new_mode_classifications_detail'].items():
                aa,rhs=inequalities(a['x_observed'],a['v_observed'],key);checks['geometry_certificates']+=certificates(aa,rhs,note)
        dm=met['direct_language_all'];da=arrays['direct_language_all'];full=met['frontier_c20_all']
        assert not dm['mother_trajectory_called'] and not dm['global_bp_called'] and not dm['archived_candidate_state_accessed']
        assert dm['pool']['visited']==0 and dm['pool']['enumerated']==dm['pool']['language_counts_product']
        regs={r.tobytes().hex() for r in da['search_regions']};assert regs==set(dm['new_mode_classifications_detail'])
        positive={k for k,n in dm['new_mode_classifications_detail'].items() if n['numerical_volume_available']}
        assert positive==set(dm['positive_modes']) and len(positive)==len(da['positive_volumes'])
        # Recompute mean readout without the production make_predict helper.
        values=np.broadcast_to(q,(len(da['points']),len(q)))
        for layer in range(4):values=np.maximum(0.,1.-abs(2.*(values+da['points'][:,layer,None])-1.))
        gap=float(np.max(abs(values.mean(axis=0)-da['prediction'])));maxreadgap=max(maxreadgap,gap)
        checks['direct_no_mother_and_readouts']+=1
        tasknotes.append(dict(seed=seed,enumerated=dm['pool']['enumerated'],remaining=len(regs),direct_positive=len(positive),
            mother_all_positive=len(full['positive_modes']),direct_only=sorted(positive-set(full['positive_modes'])),
            mother_only=sorted(set(full['positive_modes'])-positive),
            classifications=dict(Counter(n['classification'] for n in dm['new_mode_classifications_detail'].values())),
            bitwise_same_prediction=da['prediction'].tobytes()==arrays['frontier_c20_all']['prediction'].tobytes(),
            max_prediction_gap=float(np.max(abs(da['prediction']-arrays['frontier_c20_all']['prediction']))),
            direct_pool_seconds=dm['pool']['total_seconds'],direct_geometry_seconds=dm['geometry_seconds'],
            direct_returned_array_bytes=dm['returned_array_bytes'],direct_geometry_array_bytes=dm['geometry_numeric_bytes_subtotal']))
    assert maxgap<2e-12 and maxreadgap<2e-12 and np.isfinite(risk).all()
    with (out/'task_metrics.npz').open('xb') as f:np.savez_compressed(f,seeds=suite.SEEDS,methods=suite.NAMES,metric_names=METRICS,risk=risk)
    methods=[dict(method=name,metrics={metric:math.fsum(map(float,risk[:,j,k]))/32 for k,metric in enumerate(METRICS)},
        mean_current_seconds=math.fsum(index[s,name]['seconds'] for s in suite.SEEDS)/32,failures=0) for j,name in enumerate(suite.NAMES)]
    comparisons=[dict(candidate=a,control=b,metric=metric,**contrast(risk[:,i,k]-risk[:,j,k],suite.SEEDS))
        for i,a in enumerate(suite.NAMES) for j,b in enumerate(suite.NAMES) if i!=j for k,metric in enumerate(METRICS)]
    for c in comparisons:
        d=risk[:,suite.NAMES.index(c['candidate']),METRICS.index(c['metric'])]-risk[:,suite.NAMES.index(c['control']),METRICS.index(c['metric'])]
        assert abs(c['mean_difference']-math.fsum(map(float,d))/32)<2e-12
        assert c['improved']==int((d<0).sum()) and c['equal']==int((d==0).sum()) and c['worse']==int((d>0).sum());checks['paired_comparisons']+=1
    for name,value in [('methods.json',methods),('comparisons.json',comparisons),('mechanisms.json',tasknotes)]:suite.save(out/name,value)
    assert suite.gate(root)==p['source_sha256']
    result=dict(passed=True,checks=dict(checks),maximum_scalar_risk_gap=maxgap,maximum_direct_readout_gap=maxreadgap,
        seconds=time.perf_counter()-begin,core_research_goal_complete=False,
        outputs_sha256={n:suite.sha(out/n) for n in ['protocol.json','task_metrics.npz','methods.json','comparisons.json','mechanisms.json']})
    suite.save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/suite.BASE/'development_evaluation_v1';out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except Exception:suite.save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
