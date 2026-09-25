"""368 frozen no-query component resource test and exact positive-mode retention."""
import argparse
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import time
import traceback
import os
import numpy as np
import prefix_language_join_v1 as model
from budget_reinvestment_suite_v1 import read,sha,save,complete
from test_region_conditioned_credit_v1 import guarded
from run_cross_region_credit_v1 import pool
from posterior_confirmation_pipeline import discovery_box
from run_support_language_online_v1 import same

BASE='results/prefix_language_join'
DESIGN='outputs/ttt-pc-alm-research/368_prefix_language_scaling_protocol_v1.md'


def hashes(root):
    p=read(root/'results/direct_language/development_predictions_v1/protocol.json')
    h=dict(p['source_sha256'])
    for n,v in h.items():assert sha(root/n)==v
    for n in [DESIGN]+['work/experiments/'+f for f in ['prefix_language_join_v1.py','run_prefix_language_join_v1.py']]:h[n]=sha(root/n)
    return h


def singleton(x,v,d):
    accepted=[];b=F(.12);eps=F(.001);xx=F(float(x));vv=F(float(v))
    lowz=[-b,F(0),F(1,2),F(1)];highz=[F(0),F(1,2),F(1),1+b]
    for path in product(range(4),repeat=d):
        low=high=xx
        for s in path:
            left=max(low-b,lowz[s]);right=min(high+b,highz[s])
            if left>right:break
            slope=[0,2,-2,0][s];inter=[0,0,2,0][s]
            low,high=sorted([slope*left+inter,slope*right+inter])
        else:
            if max(low,vv-eps,F(0))<=min(high,vv+eps,F(1)):accepted.append(path)
    return accepted


def audit_arrays(x,v,a,m,positive):
    counts=dict(fraction_single_languages=0,prefix_rows=0,positive_prefix_inclusions=0)
    d=4;previous={b''};expanded=0
    languages=[singleton(xx,vv,d) for xx,vv in zip(x,v)]
    assert list(map(len,languages))==m['single_counts'];counts['fraction_single_languages']+=len(x)
    for i,paths in enumerate(languages):assert paths==[tuple(map(int,p)) for p in a['language_'+str(i)]]
    for step in m['steps']:
        k=step['k'];obs=step['observation'];expanded+=step['processed_expansions']
        assert step['parent_count']==len(previous) and step['possible_expansions']==len(previous)*len(languages[obs])
        assert step['retained']+step['rejected_c5']+step['rejected_c20']==step['processed_expansions']
        if not step['completed']:break
        assert step['processed_expansions']==step['possible_expansions']
        arr=a['prefix_'+str(k)];now={r.tobytes() for r in arr};assert len(now)==len(arr)==step['retained']
        for reg in arr:
            assert reg[:,:-1].copy().tobytes() in previous and tuple(map(int,reg[:,-1])) in languages[obs]
            counts['prefix_rows']+=1
        for key in positive:
            pattern=np.frombuffer(bytes.fromhex(key),np.uint8).reshape(d,len(x))
            assert pattern[:,m['order'][:k]].copy().tobytes() in now
            counts['positive_prefix_inclusions']+=1
        previous=now
    assert expanded==m['expanded']<=m['max_expanded']
    assert m['complete_stages']==sum(s['completed'] for s in m['steps'])
    if m['completed']:
        ids=m['order'];last=a['prefix_'+str(len(x))]
        expected={r[:,np.argsort(ids)].copy().tobytes() for r in last}
        assert expected=={r.tobytes() for r in a['regions']} and {bytes.fromhex(k) for k in positive}<=expected
    else:assert len(a['regions'])==0 and m['stop_reason'] is not None
    return counts


def witness_checks():
    rng=np.random.default_rng(368111);counts={};records=[]
    for n in [4,8,16,24]:
        for rep in range(3):
            x=rng.uniform(0,1,n);b=rng.uniform(-.12,.12,4);y=x.copy();codes=[]
            exact=[F(float(t)) for t in x]
            for bias in b:
                z=[t+F(float(bias)) for t in exact]
                codes.append([int(t>=0)+int(t>=F(1,2))+int(t>=1) for t in z])
                exact=[max(F(0),1-abs(2*t-1)) for t in z]
                y=np.maximum(0.,1.-abs(2.*(y+bias)-1.))
            key=np.array(codes,np.uint8).tobytes().hex()
            assert all(abs(t-F(float(v)))<F(.001) for t,v in zip(exact,y))
            for order in ['observed','small_first']:
                with guarded():a,m=model.join(x,y,ordering=order)
                checks=audit_arrays(x,y,a,m,[key])
                for name,value in checks.items():counts[name]=counts.get(name,0)+value
                records.append(dict(n=n,rep=rep,ordering=order,completed=m['completed'],seconds=m['total_seconds']))
    return dict(passed=True,calls=len(records),checks=counts,records=records)


def load_cases(root,stage):
    current=root/'results/direct_language/development_predictions_v1';complete(current)
    rows=[r for r in read(current/'rows.json') if r['method']=='direct_language_all'];cases=[]
    for r in sorted(rows,key=lambda r:r['seed']):
        p=root/r['file'];assert sha(p)==r['sha256']
        with np.load(p,allow_pickle=False) as z:x,v=z['x_observed'].copy(),z['v_observed'].copy()
        cases.append(dict(group='n4_equivalence',seed=r['seed'],n=4,x=x,v=v,source_file=r['file'],source_sha256=r['sha256'],metadata=r['metadata_file']))
    if stage=='preflight':return cases[:1]
    source=root/'results/context_block_scaling/development';old=read(source/'rows.json')
    for seed in range(5920000,5920016):
        for n in [4,8,16,24]:
            r=next(r for r in old if r['seed']==seed and r['n']==n and r['method']=='linear_ls' and r['repetition']==0)
            p=source/r['state_file'];assert r['complete'] and sha(p)==r['state_sha256']
            with np.load(p,allow_pickle=False) as z:x,v=z['x'].copy(),z['v'].copy()
            cases.append(dict(group='context_scaling',seed=seed,n=n,x=x,v=v,source_file=str(p.relative_to(root)),source_sha256=sha(p)))
    return cases


def run(root,out,stage):
    begin=time.perf_counter();h=hashes(root)
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    pre=root/BASE/'preflight_v1'
    if stage=='development':complete(pre);assert read(pre/'protocol.json')['source_sha256']==h
    cases=load_cases(root,stage);save(out/'protocol.json',dict(source_sha256=h,stage=stage,query_targets_accessed=False,
        cases=[{k:v for k,v in c.items() if k not in ['x','v']} for c in cases],limits=dict(expanded=65536,states=20000,seconds=8)))
    if stage=='preflight':save(out/'witness_checks.json',witness_checks())
    outputs=[];totals={};byseed={}
    for c in cases:byseed.setdefault(c['seed'],[]).append(c)
    with discovery_box(.12):
        for seed,cc in sorted(byseed.items()):
            jobs=[(c,o) for c in cc for o in ['observed','small_first']]
            for index in np.random.default_rng(np.random.SeedSequence([368929,seed])).permutation(len(jobs)):
                c,ordering=jobs[int(index)];x,v=c['x'],c['v'];before=(x.tobytes(),v.tobytes())
                with guarded():a,m=model.join(x,v,ordering=ordering)
                assert before==(x.tobytes(),v.tobytes())
                m['no_global_solver_guard']=True
                directory=out/f"{seed}_{c['n']}_{ordering}";directory.mkdir()
                np.savez_compressed(directory/'arrays.npz',**a,x_observed=x,v_observed=v)
                save(directory/'metadata.json',m)
                # Only after live output is saved read old pool/positive-mode metadata.
                if c['group']=='n4_equivalence':
                    assert sha(root/c['source_file'])==c['source_sha256']
                    oldmeta=read(root/c['metadata'])['metadata'];positive=oldmeta['positive_modes']
                    with guarded():
                        tick=time.perf_counter();flat,fm=pool(x,v,np.zeros((4,len(x)),np.uint8),[]);flat_seconds=time.perf_counter()-tick
                    with np.load(root/c['source_file'],allow_pickle=False) as z:assert {r.tobytes() for r in flat}=={r.tobytes() for r in z['search_regions']}
                    flatset={r.tobytes() for r in flat};newset={r.tobytes() for r in a['regions']}
                    comparison=dict(flat_seconds=flat_seconds,flat_remaining=len(flat),same_full_pool=m['completed'] and flatset==newset,
                        joined_only=len(newset-flatset),flat_only=len(flatset-newset),flat_enumerated=fm['enumerated'])
                else:
                    old=read(root/'results/context_block_scaling/development/rows.json')
                    positive=sorted(set().union(*(set(r.get('positive_mode_keys',[])) for r in old if r['seed']==seed and r['n']==c['n'] and r['complete'])))
                    comparison=None
                checked=audit_arrays(x,v,a,m,positive)
                if stage=='preflight':
                    with guarded():again,mm=model.join(x,v,ordering=ordering)
                    checked['replay_arrays']=same(a,again,list(a));assert mm['completed']==m['completed']
                if stage=='development' and c is cases[0]:
                    pp=pre/f"{seed}_{c['n']}_{ordering}"/'arrays.npz'
                    with np.load(pp,allow_pickle=False) as z:checked['preflight_arrays']=same(a,{k:z[k] for k in a},list(a))
                for k,n in checked.items():totals[k]=totals.get(k,0)+n
                save(directory/'audit.json',dict(passed=True,checks=checked,positive_modes=positive,comparison=comparison))
                row=dict(seed=seed,n=c['n'],group=c['group'],ordering=ordering,completed=m['completed'],stop_reason=m['stop_reason'],
                    seconds=m['total_seconds'],expanded=m['expanded'],full_product=m['full_product'],remaining=len(a['regions']),
                    peak_states=m['peak_completed_prefix_states'],returned_array_bytes=m['returned_array_bytes'],
                    positive_reference_modes=len(positive),comparison=comparison,directory=str(directory.relative_to(root)),
                    files={name:sha(directory/name) for name in ['arrays.npz','metadata.json','audit.json']})
                outputs.append(row)
            print(dict(stage=stage,seeds=len({r['seed'] for r in outputs}),calls=len(outputs),complete=sum(r['completed'] for r in outputs),seconds=time.perf_counter()-begin),flush=True)
    assert hashes(root)==h;save(out/'rows.json',outputs)
    result=dict(passed=True,calls=len(outputs),completed=sum(r['completed'] for r in outputs),budget_exhausted=sum(not r['completed'] for r in outputs),
        checks=totals,seconds=time.perf_counter()-begin,query_targets_accessed=False,core_research_goal_complete=False,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','rows.json']})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','development'],required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];out=root/BASE/(args.stage+'_v1');out.mkdir(parents=True,exist_ok=False)
    try:run(root,out,args.stage)
    except Exception:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
