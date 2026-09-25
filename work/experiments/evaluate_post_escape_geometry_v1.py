"""319 post-seal common geometry, all fifteen pools and deduplicated points."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
import complete_credit_mode_geometry_v1 as geometry
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha,certify_label,describe

FAMILIES=['alm_keep','alm_reset','nodual']
VARIANTS=['original64','corrected64','escaped64','corrected_virtual','escaped_matched']
NAMES=[f+'_'+v for f in FAMILIES for v in VARIANTS]


def exact_forward(b,x):
    values=list(map(F,x));codes=[]
    for bias in map(F,b):
        zz=[z+bias for z in values];codes.extend(sum(z>=k for k in [F(0),F(1,2),F(1)]) for z in zz)
        values=[max(F(0),min(2*z,2-2*z)) for z in zz]
    return bytes(codes).hex(),values


def compare(methods,original,prior):
    result={}
    for f in FAMILIES:
        p={v:set(methods[f+'_'+v]['positive_modes']) for v in VARIANTS}
        result[f]=dict(escaped_new_vs_original=sorted(p['escaped64']-original),
            escaped_new_vs_prior=sorted(p['escaped64']-prior),
            escaped_new_vs_prior_and_corrected64=sorted(p['escaped64']-prior-p['corrected64']),
            escaped_new_vs_prior_and_all_controls=sorted(p['escaped64']-prior-set().union(*(p[v] for v in VARIANTS if v!='escaped64'))),
            corrected64_new_vs_prior=sorted(p['corrected64']-prior),
            escaped_only_vs_corrected_virtual=sorted(p['escaped64']-p['corrected_virtual']),
            corrected_virtual_only_vs_escaped=sorted(p['corrected_virtual']-p['escaped64']),
            escaped_matched_only_vs_corrected64=sorted(p['escaped_matched']-p['corrected64']),
            corrected64_only_vs_escaped_matched=sorted(p['corrected64']-p['escaped_matched']))
    return result


def run(root,out):
    start=time.perf_counter();source=root/'results/post_escape_continuation/development_v1'
    audited=root/'results/post_escape_continuation/audit_v2/summary.json';assert read(audited)['passed']
    summary=read(source/'summary.json');assert summary['passed']
    manifest=read(source/'before_geometry_manifest.json');assert sha(source/'before_geometry_manifest.json')==summary['manifest_sha256']
    for n,digest in manifest['files_sha256'].items():assert sha(source/n)==digest
    priordir=root/'results/paired_primal_dual_predictor/geometry_v1'
    prior_summary=read(priordir/'summary.json');assert prior_summary['passed']
    for n,digest in prior_summary['outputs_sha256'].items():assert sha(priordir/n)==digest
    additional=root/'results/primal_stasis_escape/geometry_v1';extra_summary=read(additional/'summary.json');assert extra_summary['passed']
    for n,digest in extra_summary['outputs_sha256'].items():assert sha(additional/n)==digest
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in
                                                [Path(__file__).name,'complete_credit_mode_geometry_v1.py','evaluate_complete_credit_mode_geometry_v1.py']},
                                 continuation_sha256=sha(source/'summary.json'),audit_sha256=sha(audited),
                                 prior_geometry_sha256=sha(priordir/'summary.json'),entry_geometry_sha256=sha(additional/'summary.json'),
                                 methods=NAMES,tasks=64,query_targets_accessed=False,posterior_moments_accessed=False,
                                 resources_matched=False,global_lp_scope='Post-seal evaluator, not online credit.',
                                 point_scope='All distinct binary64 parameter vectors, exact support fit and native floating support fit separately.'))
    grouped=defaultdict(list)
    for row in read(source/'rows.json'):grouped[row['seed']].append(row)
    descriptions={(t['file'],t['index']):t for t in read(source/'trajectories.json')}
    oldtasks=read(priordir/'tasks.json');extra={r['seed']:r for r in read(additional/'tasks.json')}
    counts=Counter();tasks=[];files={}
    for oldrow in oldtasks:
        seed=oldrow['seed'];old=read(priordir/oldrow['file']);entry=read(additional/extra[seed]['file'])
        x,v=np.array(old['x_observed']),np.array(old['v_observed']);original=set(old['original_positive_modes'])
        prior=original|set(old['all_308_309_positive_modes'])|set().union(*(set(m['positive_modes']) for m in old['methods'].values()))
        cache=dict(old['geometry']);cache.update(entry['geometry']);props={n:set() for n in NAMES};points={};loaded={}
        occurrence=Counter()
        for row in grouped[seed]:
            with np.load(source/row['file'],allow_pickle=False) as z:loaded[row['family'],row['variant']]={k:z[k] for k in z.files}
        def add_point(b,code,value,name):
            key=b.tobytes();mode=code.tobytes().hex();occurrence[name]+=1
            if key not in points:points[key]=dict(b=b.tolist(),mode=mode,native_prediction=value.tolist(),methods=set(),occurrences=Counter())
            else:assert points[key]['mode']==mode and points[key]['native_prediction']==value.tolist()
            points[key]['methods'].add(name);points[key]['occurrences'][name]+=1
        for row in grouped[seed]:
            arrays=loaded[row['family'],row['variant']];name=row['family']+'_'+row['variant']
            assert np.array_equal(x,arrays['x_observed']) and np.array_equal(v,arrays['v_observed'])
            for i,horizon in enumerate(arrays['horizons']):
                described=descriptions[row['file'],i];props[name].update(described['proposal_modes'])
                for t in range(int(horizon)+1):add_point(arrays['b'][t,i],arrays['forward_codes'][t,i],arrays['forward_values'][t,i],name)
                if described['known_q_prefix_mode'] is not None:
                    q=loaded[row['family'],'corrected64'];assert q['forward_codes'][0,i].tobytes().hex()==described['known_q_prefix_mode']
                    add_point(q['b'][0,i],q['forward_codes'][0,i],q['forward_values'][0,i],name)
        classified={}
        for mode in sorted(set().union(*props.values())):
            if mode in cache:result=cache[mode];counts['reused_classifications']+=1
            else:
                result=geometry.classify_mode(x,v,mode);counts['new_classifications']+=1;counts['new_lp_calls']+=result['lp_calls']
            _,a,rhs,_,_=geometry.matrices(x,v,mode);counts['certificate_rechecks']+=certify_label(a,rhs,result)
            classified[mode]=result;counts[result['classification']]+=1
        pointstats={n:dict(unique_parameter_points=0,exact_support_feasible=0,native_support_feasible=0,
                          exact_native_mode_disagreements=0,point_occurrences=occurrence[n]) for n in NAMES}
        pointrecords=[]
        for p in points.values():
            exact_mode,prediction=exact_forward(p['b'],x)
            ef=all(abs(z-F(y))<=F(.001) for z,y in zip(prediction,v)) and all(abs(F(b))<=F(.12) for b in p['b'])
            nf=bool(np.all(abs(np.array(p['native_prediction'])-v)<=.001) and np.all(abs(np.array(p['b']))<=.12))
            mismatch=exact_mode!=p['mode'];p.update(exact_mode=exact_mode,exact_prediction=prediction,
                exact_support_feasible=ef,native_support_feasible=nf,exact_native_mode_disagreement=mismatch)
            p['methods']=sorted(p['methods']);p['occurrences']=dict(p['occurrences'])
            for n in p['methods']:
                pointstats[n]['unique_parameter_points']+=1;pointstats[n]['exact_support_feasible']+=ef
                pointstats[n]['native_support_feasible']+=nf;pointstats[n]['exact_native_mode_disagreements']+=mismatch
            pointrecords.append(p);counts['unique_parameter_points']+=1;counts['exact_support_feasible_points']+=ef
            counts['native_support_feasible_points']+=nf;counts['exact_native_mode_disagreements']+=mismatch
        counts['proposal_point_occurrences']+=sum(occurrence.values())
        methods={n:describe(props[n],classified,original) for n in NAMES};comparison=compare(methods,original,prior)
        record=dict(seed=seed,methods=methods,comparisons=comparison,point_stats=pointstats,points=pointrecords,
                    original_positive_modes=sorted(original),prior_positive_modes=sorted(prior),geometry=classified,
                    x_observed=x.tolist(),v_observed=v.tolist())
        filename=f'{seed}_geometry.json';save(out/filename,record);files[filename]=sha(out/filename)
        tasks.append(dict(seed=seed,file=filename,sha256=files[filename],methods=methods,comparisons=comparison,point_stats=pointstats))
        if grouped[seed]:print(dict(seed=seed,unique_points=len(points),modes=len(classified),seconds=time.perf_counter()-start),flush=True)
    assert len(tasks)==64 and counts['proposal_point_occurrences']==127725+2*86
    aggregate={}
    for n in NAMES:
        aggregate[n]=dict(tasks_with_new_positive=sum(bool(t['methods'][n]['new_positive_modes']) for t in tasks),
                          new_positive_pairs=sum(len(t['methods'][n]['new_positive_modes']) for t in tasks),
                          positive_pairs=sum(len(t['methods'][n]['positive_modes']) for t in tasks),
                          **{k:sum(t['point_stats'][n][k] for t in tasks) for k in tasks[0]['point_stats'][n]})
    comparisons={f:{k:sum(len(t['comparisons'][f][k]) for t in tasks) for k in tasks[0]['comparisons'][f]} for f in FAMILIES}
    for n,value in [('tasks.json',tasks),('aggregate.json',aggregate),('comparisons.json',comparisons)]:save(out/n,value);files[n]=sha(out/n)
    files['protocol.json']=sha(out/'protocol.json')
    final=dict(passed=True,counts=dict(counts),aggregate=aggregate,comparisons=comparisons,seconds=time.perf_counter()-start,
               query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False,outputs_sha256=files)
    save(out/'summary.json',final);print({k:v for k,v in final.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
