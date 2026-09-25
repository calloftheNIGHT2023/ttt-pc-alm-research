"""Independent scalar ordering signals, state matches, resource and pool audit."""
from collections import Counter,defaultdict
import math
from pathlib import Path
import time
import numpy as np
from budget_reinvestment_suite_v1 import read,sha,save,complete


def load(file):
    with np.load(file,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def run(root,out):
    begin=time.perf_counter();source=root/'results/multiplier_constraint_order/development_v1';complete(source)
    p=read(source/'protocol.json');rows=read(source/'rows.json');assert len(rows)==288
    for file,h in p['source_sha256'].items():assert sha(root/file)==h
    counts=Counter();maximum_gap=0.;groups=defaultdict(list);pools=[]
    for r in rows:
        directory=root/r['directory']
        for file,h in r['files'].items():assert sha(directory/file)==h
        a=load(directory/'arrays.npz');m=read(directory/'metadata.json');x=a['original_x'];v=a['original_v'];pi=a['support_order'];n=len(x)
        assert sorted(map(int,pi))==list(range(n)) and np.array_equal(a['x_observed'],x[pi]) and np.array_equal(a['v_observed'],v[pi])
        assert m['query_targets_accessed'] is False and m['geometry_accessed'] is False and m['generation_charged']
        assert r['seconds']==m['total_component_seconds']>=m['ordering_seconds']+m['total_seconds']>0
        assert r['expanded']==sum(s['processed_expansions'] for s in m['steps'])<=65536
        assert r['completed']==m['completed'] and (m['stop_reason'] is None)==m['completed']
        name=r['method'];scores=None
        if name=='observed':expected=list(range(n))
        elif name=='random':expected=list(np.random.default_rng(np.random.SeedSequence([370111,r['seed'],n])).permutation(n))
        elif name=='small_first':expected=sorted(range(n),key=lambda i:(int(a['language_counts'][i]),i))
        elif name=='farthest_x':
            expected=[min(range(n),key=lambda i:(x[i],i))]
            while len(expected)<n:
                expected.append(min(set(range(n))-set(expected),key=lambda i:(-min(abs(float(x[i])-float(x[j])) for j in expected),i)))
        elif name=='pair_compatibility':
            c=a['language_counts'];k=a['pair_counts'];assert np.array_equal(k,k.T) and np.all(k>=0) and np.all(k<=c[:,None]*c[None])
            assert r['pair_candidate_rows']==sum(int(c[i]*c[j]) for i in range(n) for j in range(i+1,n))
            expected=[min(range(n),key=lambda i:(int(c[i]),i))]
            while len(expected)<n:
                def value(i):return math.log(int(c[i]))+sum(math.log(max(1,int(k[i,j]))/int(c[i]*c[j])) for j in expected)
                expected.append(min(set(range(n))-set(expected),key=lambda i:(value(i),i)))
        else:
            b,h,u=a['order_b'],a['order_h'],a['order_u'];assert b.shape==(17,4) and h.shape==u.shape==(4,17,n)
            scores=[]
            for i in range(n):
                bank=[]
                for restart in range(17):
                    if name=='alm_dual':value=math.fsum(abs(float(t)) for t in u[:,restart,i])
                    elif name in ['alm_residual','pc_residual']:
                        value=0.
                        for layer in range(4):
                            previous=float(x[i]) if layer==0 else float(h[layer-1,restart,i])
                            computed=float(h[layer,restart,i])-max(0.,1.-abs(2.*(previous+float(b[restart,layer]))-1.))
                            maximum_gap=max(maximum_gap,abs(computed-float(a['order_residual'][layer,restart,i])))
                            value+=abs(computed)
                    else:
                        assert name=='alm_bp_score';y=float(x[i]);jac=[0.]*4
                        for layer in range(4):
                            z=y+float(b[restart,layer]);s=2. if 0.<z<.5 else -2. if .5<z<1. else 0.
                            jac=[t*s for t in jac];jac[layer]+=s;y=max(0.,1.-abs(2.*z-1.))
                        error=y-float(v[i]);residual=math.copysign(max(abs(error)-.001,0.),error)
                        value=math.fsum(abs(t*residual) for t in jac)
                    bank.append(value)
                scores.append(math.fsum(bank)/17)
            maximum_gap=max(maximum_gap,float(np.max(abs(np.array(scores)-a['order_scores']))))
            expected=sorted(range(n),key=lambda i:(-a['order_scores'][i],i))
            assert m['global_bp_score']==(name=='alm_bp_score')
            if name=='pc_residual':assert not u.any()
            counts['scalar_observation_signals']+=n
        assert list(pi)==expected;counts['orderings']+=1;counts['resource_rows']+=1
        groups[r['seed'],n].append((r,a,m))
    assert maximum_gap<2e-12
    for (seed,n),items in groups.items():
        assert len(items)==9;by={r['method']:(r,a,m) for r,a,m in items}
        for other in ['alm_residual','alm_bp_score']:
            for field in ['order_b','order_h','order_u','order_residual']:
                assert by['alm_dual'][1][field].tobytes()==by[other][1][field].tobytes();counts['same_alm_state_arrays']+=1
        complete_sets={}
        for r,a,m in items:
            if r['completed']:
                keys={reg[:,np.argsort(a['support_order'])].copy().tobytes().hex() for reg in a['regions']}
                assert len(keys)==len(a['regions']);complete_sets[r['method']]=keys
        base=next(iter(complete_sets.values()),set())
        pools.append(dict(seed=seed,n=n,complete_methods=list(complete_sets),all_complete_pools_equal=all(keys==base for keys in complete_sets.values()),
            complete_pool_counts={name:len(keys) for name,keys in complete_sets.items()}))
        counts['task_groups']+=1
    save(out/'pools.json',pools)
    result=dict(passed=True,checks=dict(counts),maximum_scalar_signal_gap=maximum_gap,seconds=time.perf_counter()-begin,
        query_targets_accessed=False,core_research_goal_complete=False,
        source_sha256=sha(Path(__file__)),development_summary_sha256=sha(source/'summary.json'),outputs_sha256={'pools.json':sha(out/'pools.json')})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/'results/multiplier_constraint_order/audit_v1';out.mkdir(parents=True,exist_ok=False)
    run(root,out)
