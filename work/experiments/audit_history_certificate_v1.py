"""374 independent source signal, all exact certificates, matched-state and resource audit."""
from collections import Counter,defaultdict
from pathlib import Path
from fractions import Fraction as F
import math,time
import numpy as np
import run_history_certificate_v1 as run


def scalar_credit(x,v,b,h,u,index,name,seed):
    n=len(x);raw=np.zeros((4,n))
    if '_dual' in name:return u[:,index].copy()
    if '_random' in name:return np.random.default_rng(np.random.SeedSequence([374137,seed,n])).choice(np.array([-1.,1.]),size=(4,n))
    if '_zero' in name:return raw
    for i in range(n):
        if '_residual' in name:
            previous=float(x[i])
            for j in range(4):
                z=previous+float(b[j]);raw[j,i]=float(h[j,i])-max(0.,1.-abs(2*z-1));previous=float(h[j,i])
        else:
            assert name=='alm_bp128';value=float(x[i]);derivatives=[]
            for j in range(4):
                z=value+float(b[j]);derivatives.append(2. if 0.<z<.5 else -2. if .5<z<1. else 0.);value=max(0.,1.-abs(2*z-1))
            err=value-float(v[i]);adj=-math.copysign(max(abs(err)-.001,0.),err);raw[3,i]=adj
            for j in range(2,-1,-1):adj*=derivatives[j+1];raw[j,i]=adj
    return raw


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def audit(root,out):
    tick=time.perf_counter();base=root/run.BASE;summary=run.complete(base/'development_v1');p=run.read(base/'development_v1/protocol.json');assert p['source_sha256']==run.hashes(root)
    rows=run.read(base/'development_v1/rows.json');assert summary['calls']==len(rows)==432 and summary['all_methods_sealed_before_external_labels']
    counts=Counter();by=defaultdict(dict);stats=defaultdict(lambda:Counter());maxgap=0.;costs=defaultdict(list);trace=[]
    cases={(c['seed'],c['method']):c for c in p['cases']}
    for row in rows:
        directory=root/row['directory']
        for file,h in row['files'].items():assert run.sha(directory/file)==h
        a=load(directory/'arrays.npz');m=run.read(directory/'metadata.json');c=cases[row['seed'],row['ordering']];x,v,regs=run.data(root,c);name=row['method']
        assert m['historical_generation_charged'] and not m['query_targets_accessed'] and not m['full_context_suffix_accessed'] and not m['lp_accessed']
        assert m['global_bp_used']==(name=='alm_bp128')
        assert m['total_seconds']>=m['generation_seconds']+m['solver_seconds']>0
        assert m['returned_array_bytes']==sum(t.nbytes for t in a.values()) and row['certified']==int((a['first_step']>=0).sum())
        if not name.startswith('cold_'):
            errors=[];moves=[]
            for b in a['mother_b']:
                predicted=x.copy()
                for bias in b:predicted=np.maximum(0.,1.-abs(2.*(predicted+bias)-1.))
                errors.append(float(np.max(abs(predicted-v))));moves.append(float(.5*(b*b).sum()))
            # Selection movement scale does not alter tie order; check its actual source too.
            idx=min(range(17),key=lambda i:(errors[i],float(a['mother_moves'][i]),i));assert idx==m['selected_restart']
            assert a['start_b'].tobytes()==a['mother_b'][idx].tobytes() and a['start_h'].tobytes()==a['mother_h'][:,idx].tobytes()
            raw=scalar_credit(x,v,a['start_b'],a['start_h'],a['mother_u'],idx,name,row['seed'])
            maxgap=max(maxgap,float(abs(raw-a['credit_raw']).max()));counts['scalar_credit_entries']+=raw.size
        else:assert not a['credit'].any() and not a['start_b'].any()
        norm=float(np.linalg.norm(a['credit_raw']));assert norm==m['credit_raw_norm']
        assert np.array_equal(a['credit'],a['credit_raw']/norm if norm>0 else a['credit_raw'])
        assert np.array_equal(a['initial_a'],np.broadcast_to(a['credit'],regs.shape))
        assert np.array_equal(a['initial_p'],np.array([0.,2.,-2.,0.])[regs]*a['initial_a'])
        counts['exact_wide_domain_proofs']+=run.verify_proofs(x,v,regs,a)
        diagnoses=run.read(root/'results/prefix_obstruction/development_v3'/f"{row['seed']}_{row['ordering']}"/'classification.json')
        for i,r in enumerate(diagnoses):
            if r['classification']=='positive_volume':assert a['first_step'][i]<0;counts['positive_retained']+=1
            if a['first_step'][i]>=0:assert r['classification']=='infeasible'
        previous=-1.
        for cp in m['checkpoints']:
            assert cp['rejected']==int(((a['first_step']>=0)&(a['first_step']<=cp['step'])).sum()) and cp['solver_seconds']>previous;previous=cp['solver_seconds'];counts['checkpoints']+=1
            if cp['step'] in [0,32,64,128,256,512,1024]:trace.append(dict(seed=row['seed'],ordering=row['ordering'],source_completed=row['source_completed'],method=name,step=cp['step'],certified=cp['rejected'],generation_plus_checkpoint_seconds=m['generation_seconds']+cp['solver_seconds']))
        key=(name,'complete' if row['source_completed'] else 'budget_exhausted');stats[key].update(calls=1,samples=len(regs),certified=row['certified'],initial_certified=int((a['first_step']==0).sum()))
        costs[key].append(row['total_seconds']);by[row['seed'],row['ordering']][name]=(row,a,m)
    assert maxgap<2e-12
    paired=[]
    for (seed,order),items in by.items():
        for name in run.model.METHODS:
            source='alm_dual128' if name.startswith('alm_') else 'pc_zero128' if name.startswith('pc_') else None
            if source:
                for field in ['mother_b','mother_h','mother_u','start_b','start_h','initial_b','initial_h','initial_z']:
                    assert items[name][1][field].tobytes()==items[source][1][field].tobytes();counts['same_primal_state_arrays']+=1
        cold=items['cold_zero128'][1]['first_step'];long=items['cold_zero1024'][1]['first_step']
        assert np.array_equal(cold>=0,(long>=0)&(long<=128));counts['cold_prefix_replays']+=1
        primary=items[run.model.PRIMARY][1]['first_step']>=0
        for name in run.model.METHODS:
            other=items[name][1]['first_step']>=0
            paired.append(dict(seed=seed,ordering=order,source_completed=items[name][0]['source_completed'],control=name,
                primary_only=int((primary&~other).sum()),control_only=int((other&~primary).sum()),both=int((other&primary).sum())))
    groups=[dict(method=name,source_status=status,**dict(value),mean_total_seconds=math.fsum(costs[name,status])/len(costs[name,status])) for (name,status),value in stats.items()]
    run.save(out/'groups.json',groups);run.save(out/'paired.json',paired);run.save(out/'checkpoints.json',trace)
    result=dict(passed=True,checks=dict(counts),maximum_scalar_credit_gap=maxgap,seconds=time.perf_counter()-tick,
        development_summary_sha256=run.sha(base/'development_v1/summary.json'),source_sha256=run.sha(Path(__file__)),query_targets_accessed=False,
        outputs_sha256={f:run.sha(out/f) for f in ['groups.json','paired.json','checkpoints.json']})
    run.save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/run.BASE/'audit_v1';out.mkdir(parents=True,exist_ok=False);audit(root,out)
