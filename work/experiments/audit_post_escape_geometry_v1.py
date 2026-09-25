"""319 independent geometry certificates, raw point coverage and all controls."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
from audit_stasis_escape_geometry_v1 import inequalities
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

FAMILIES=['alm_keep','alm_reset','nodual']
VARIANTS=['original64','corrected64','escaped64','corrected_virtual','escaped_matched']


def certificates(a,rhs,result):
    positive=False;impossible=False;zero=False;count=0
    for c in result['certificates']:
        if c['type']=='negative_constant_row':
            index=c['row_index'];assert not any(a[index]) and rhs[index]==F(c['rhs'])<0;impossible=True
        elif c['type'] in ['exact_interior_cube','exact_point_check']:
            p=list(map(F,c['point']));slacks=[r-sum((v*b for v,b in zip(row,p)),F(0)) for row,r in zip(a,rhs)]
            feasible=all(s>=0 for s in slacks);strict=feasible and all(s>0 for s,row in zip(slacks,a) if any(row))
            assert c['closed_region_feasible']==feasible and c['strict_interior']==strict
            if strict:
                radius=F(c['radius']);assert radius>0
                assert all(s-radius*sum(map(abs,row),F(0))>0 if any(row) else s>=0 for s,row in zip(slacks,a));positive=True
        else:
            assert c['type']=='exact_box_separation';w=[F(0)]*len(a)
            for i,v in c['weights']:assert not w[i];w[i]=F(v);assert w[i]>=0
            coeff=[sum((v*row[j] for v,row in zip(w,a)),F(0)) for j in range(4)]
            constant=sum((v*r for v,r in zip(w,rhs)),F(0));lower=-constant-F(.12)*sum(map(abs,coeff),F(0))
            assert coeff==list(map(F,c['weighted_coefficients'])) and constant==F(c['weighted_rhs']) and lower==F(c['lower'])
            if c['conclusion']=='infeasible':assert lower>0;impossible=True
            if c['conclusion']=='zero_volume_or_empty':
                assert lower==0 and (any(coeff) or any(v>0 and any(row) for v,row in zip(w,a)));zero=True
        count+=1
    assert not (positive and (impossible or zero));assert positive==result['positive_volume_certified']
    if result['classification']=='positive_volume':assert positive
    elif result['classification']=='infeasible':assert impossible
    elif result['classification']=='zero_volume_or_empty':assert zero and not impossible
    else:assert result['classification']=='unresolved' and not (positive or impossible or zero)
    return count


def run(root,out):
    start=time.perf_counter();folder=root/'results/post_escape_continuation/geometry_v1';source=root/'results/post_escape_continuation/development_v1'
    summary=read(folder/'summary.json');assert summary['passed']
    for n,digest in summary['outputs_sha256'].items():assert sha(folder/n)==digest
    source_rows=defaultdict(list)
    for row in read(source/'rows.json'):source_rows[row['seed']].append(row)
    descriptions={(t['file'],t['index']):t for t in read(source/'trajectories.json')}
    checks=Counter();tasks=[]
    for taskrow in read(folder/'tasks.json'):
        task=read(folder/taskrow['file']);seed=task['seed'];x=np.array(task['x_observed']);v=np.array(task['v_observed'])
        for mode,result in task['geometry'].items():
            a,rhs=inequalities(x,v,mode);checks['independent_certificates']+=certificates(a,rhs,result)
        expected=defaultdict(Counter);loaded={}
        for row in source_rows[seed]:
            with np.load(source/row['file'],allow_pickle=False) as z:loaded[row['family'],row['variant']]={k:z[k] for k in z.files}
        for row in source_rows[seed]:
            ar=loaded[row['family'],row['variant']];name=row['family']+'_'+row['variant']
            for i,length in enumerate(ar['horizons']):
                for b in ar['b'][:int(length)+1,i]:expected[b.tobytes()][name]+=1
                if descriptions[row['file'],i]['known_q_prefix_mode'] is not None:
                    q=loaded[row['family'],'corrected64']['b'][0,i];expected[q.tobytes()][name]+=1
        assert len(expected)==len(task['points']);props={n:set() for n in task['methods']}
        stats={n:dict(unique_parameter_points=0,exact_support_feasible=0,native_support_feasible=0,
                      exact_native_mode_disagreements=0,point_occurrences=0) for n in task['methods']};seen=set()
        for p in task['points']:
            b=np.array(p['b'],dtype=float);key=b.tobytes();assert key in expected and key not in seen;seen.add(key)
            assert dict(expected[key])==p['occurrences'] and sorted(expected[key])==p['methods']
            exact=list(map(F,x));native=x.copy();ec=[];nc=[]
            for bias in b:
                zz=[q+F(bias) for q in exact];ec.extend(sum(z>=k for k in [F(0),F(1,2),F(1)]) for z in zz)
                exact=[max(F(0),1-abs(2*z-1)) for z in zz]
                zz=native+bias;nc.extend(int(z>=0)+int(z>=.5)+int(z>=1) for z in zz);native=np.maximum(0,1-np.abs(2*zz-1))
            assert bytes(ec).hex()==p['exact_mode'] and bytes(nc).hex()==p['mode']
            assert exact==list(map(F,p['exact_prediction'])) and np.array_equal(native,p['native_prediction'])
            exactfit=all(abs(q-F(y))<=F(.001) for q,y in zip(exact,v)) and all(abs(F(z))<=F(.12) for z in b)
            nativefit=bool(np.all(abs(native-v)<=.001) and np.all(abs(b)<=.12));mismatch=ec!=nc
            assert p['exact_support_feasible']==exactfit and p['native_support_feasible']==nativefit and p['exact_native_mode_disagreement']==mismatch
            for name in p['methods']:
                stats[name]['unique_parameter_points']+=1;stats[name]['exact_support_feasible']+=exactfit
                stats[name]['native_support_feasible']+=nativefit;stats[name]['exact_native_mode_disagreements']+=mismatch
                stats[name]['point_occurrences']+=expected[key][name];props[name].add(p['mode'])
            checks['independent_point_predictions']+=1;checks['point_occurrences']+=sum(expected[key].values())
        assert stats==task['point_stats']
        for name,modes in props.items():
            assert sorted(modes)==task['methods'][name]['proposal_modes']
            positive={m for m in modes if task['geometry'][m]['positive_volume_certified']}
            assert sorted(positive)==task['methods'][name]['positive_modes']
            assert sorted(positive-set(task['original_positive_modes']))==task['methods'][name]['new_positive_modes']
            checks['method_pool_checks']+=1
        prior=set(task['prior_positive_modes']);original=set(task['original_positive_modes'])
        for f in FAMILIES:
            p={variant:set(task['methods'][f+'_'+variant]['positive_modes']) for variant in VARIANTS};stored=task['comparisons'][f]
            expected_comparisons=dict(escaped_new_vs_original=p['escaped64']-original,
                escaped_new_vs_prior=p['escaped64']-prior,
                escaped_new_vs_prior_and_corrected64=p['escaped64']-prior-p['corrected64'],
                escaped_new_vs_prior_and_all_controls=p['escaped64']-prior-set().union(*(p[z] for z in VARIANTS if z!='escaped64')),
                corrected64_new_vs_prior=p['corrected64']-prior,
                escaped_only_vs_corrected_virtual=p['escaped64']-p['corrected_virtual'],
                corrected_virtual_only_vs_escaped=p['corrected_virtual']-p['escaped64'],
                escaped_matched_only_vs_corrected64=p['escaped_matched']-p['corrected64'],
                corrected64_only_vs_escaped_matched=p['corrected64']-p['escaped_matched'])
            assert {k:sorted(value) for k,value in expected_comparisons.items()}==stored;checks['comparison_sets']+=len(stored)
        tasks.append(task)
    for name,values in summary['aggregate'].items():
        expected=dict(tasks_with_new_positive=sum(bool(t['methods'][name]['new_positive_modes']) for t in tasks),
                      new_positive_pairs=sum(len(t['methods'][name]['new_positive_modes']) for t in tasks),
                      positive_pairs=sum(len(t['methods'][name]['positive_modes']) for t in tasks),
                      **{k:sum(t['point_stats'][name][k] for t in tasks) for k in tasks[0]['point_stats'][name]})
        assert expected==values;checks['aggregate_fields']+=len(values)
    for f,values in summary['comparisons'].items():
        assert values=={k:sum(len(t['comparisons'][f][k]) for t in tasks) for k in values};checks['aggregate_fields']+=len(values)
    assert checks['point_occurrences']==127897
    result=dict(passed=True,checks=dict(checks),seconds=time.perf_counter()-start,input_sha256=sha(folder/'summary.json'),
                source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'audit_stasis_escape_geometry_v1.py']},
                query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
