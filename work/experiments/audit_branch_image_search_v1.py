"""321 independent real-data suffix search, K-best proposals and domination."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
from independent_branch_image_search_v1 import IndependentSearch,layer_table,EPS
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


def run(root,out):
    start=time.perf_counter();source=root/'results/branch_image_chain/development_v1'
    summary=read(source/'summary.json');assert summary['passed']
    manifest=read(source/'before_geometry_manifest.json');assert sha(source/'before_geometry_manifest.json')==summary['manifest_sha256']
    for n,digest in manifest['files_sha256'].items():assert sha(source/n)==digest
    tests=root/'results/branch_image_chain/independent_tests_v1/summary.json';tested=read(tests);assert tested['passed']
    for n,digest in tested['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    oldroot=root/'results/factorized_dual_branch_search/development_v1'
    protocol=read(source/'protocol.json');assert sha(oldroot/'summary.json')==protocol['input_summary_sha256']
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in
                                               [Path(__file__).name,'independent_branch_image_search_v1.py']},
         input_summary_sha256=sha(source/'summary.json'),independent_tests_sha256=sha(tests),query_targets_accessed=False))
    counts=Counter();rows=[];aggregate={n:Counter() for n in CHANNELS}
    for index,row in enumerate(read(source/'rows.json')):
        rec=read(source/row['file']);old=read(oldroot/rec['source_file']);assert sha(oldroot/rec['source_file'])==rec['source_sha256']
        for field in ['seed','location','x_observed','v_observed','b','h','u','original_mode','credits']:
            assert rec[field]==old[field];counts['source_fields']+=1
        x=np.array(rec['x_observed']);v=np.array(rec['v_observed']);pattern=np.array(list(bytes.fromhex(rec['original_mode']))).reshape(4,4)
        oldgeometry=read(root/'results/post_escape_continuation/geometry_v1'/f"{rec['seed']}_geometry.json")
        details={}
        for name in CHANNELS:
            credit=np.array(rec['credits'][name]);checker=IndependentSearch(x,v,credit,pattern);result=rec['results'][name]
            best={m:checker.best(0,-1,m) for m in range(17)}
            assert {str(m):str(p[0]) for m,p in best.items() if p is not None}==result['shell_minima']
            counts['global_shell_minima']+=len(result['shell_minima'])
            current=checker.fixed(pattern);zero=best[0]
            assert (current is None)==(zero is None)
            if current is not None:assert zero[0]==current and zero[1]==tuple(pattern.ravel())
            assert result['current_lower']==(None if current is None else str(current))
            assert result['current_structurally_infeasible']==(current is None)
            positive=current is None or current>0;assert positive==result['current_certified_infeasible']
            oldcurrent=F(old['results'][name]['current_lower'])
            assert current is None or current>=oldcurrent;counts['current_domination']+=1
            if positive:
                assert oldgeometry['geometry'][rec['original_mode']]['classification']=='infeasible';counts['independent_current_infeasibility']+=1
            allowed=[m for m,pair in best.items() if m>=1 and pair is not None and pair[0]<=0]
            minimum=min(allowed) if allowed else None
            assert result['minimum_nonexcluded_hamming']==minimum
            assert result['necessary_hamming_lower_bound']==(minimum if positive else None)
            expected=[]
            if minimum is not None:
                for m in range(minimum,min(minimum+2,16)+1):
                    pool=checker.kbest(m,k=8);counts['independent_kbest_shells']+=1
                    for rank,(value,path) in enumerate(pool):
                        if value<=0:expected.append(dict(mode=bytes(path).hex(),hamming=m,rank=rank,lower=str(value)))
            assert expected==result['proposals'];counts['independent_kbest_proposals']+=len(expected)
            aa=[tuple(map(F,credit[j])) for j in range(4)];oldtables=[]
            for j in range(4):
                lo=tuple(map(F,x)) if j==0 else (F(0),)*4;hi=lo if j==0 else (F(1),)*4
                prev=(F(0),)*4 if j==0 else aa[j-1]
                oldtables.append({r:value for r,_,value in layer_table(lo,hi,prev,aa[j])})
            terminal=sum((min(a*max(F(0),F(y)-EPS),a*min(F(1),F(y)+EPS)) for a,y in zip(aa[-1],v)),F(0))
            for p in expected:
                path=tuple(bytes.fromhex(p['mode']));pp=[path[j*4:(j+1)*4] for j in range(4)]
                assert F(p['lower'])==checker.fixed(pp)
                bound=terminal+sum((oldtables[j][pp[j]] for j in range(4)),F(0))
                assert F(p['lower'])>=bound;counts['proposal_domination']+=1
            details[name]=dict(shells=len(result['shell_minima']),proposals=len(expected),best_first_expansions=checker.expansions,
                               current_structural=current is None,current_positive=positive)
            for key,value in [('cases',1),('proposals',len(expected)),('current_certified_infeasible',int(positive)),
                              ('current_structurally_infeasible',int(current is None))]:aggregate[name][key]+=value
            counts['best_first_expansions']+=checker.expansions;checker.clear();checker.best=None
        rows.append(dict(seed=rec['seed'],location=rec['location'],checks=details))
        if (index+1)%5==0:print(dict(audited_states=index+1,seconds=time.perf_counter()-start),flush=True)
    assert len(rows)==131
    for name,fields in aggregate.items():
        for field,value in fields.items():assert summary['aggregate'][name][field]==value;counts['aggregate_fields']+=1
    save(out/'rows.json',rows)
    result=dict(passed=True,counts=dict(counts),seconds=time.perf_counter()-start,
                input_sha256=sha(source/'summary.json'),protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),
                scope='Independent scalar reverse minima and exact best-first K-order on all proposed real-data shells; all fixed lower bounds and domination.',
                query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
