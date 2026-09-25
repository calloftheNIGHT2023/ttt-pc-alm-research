"""376 proof, state-isolation, checkpoint and complete-cost audit."""
from collections import Counter,defaultdict
from pathlib import Path
import math,time
import numpy as np
import run_regional_alm_v1 as run


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def audit(root,out):
    start=time.perf_counter();base=root/run.BASE;s=run.complete(base/'development_v1');p=run.read(base/'development_v1/protocol.json');assert p['source_sha256']==run.hashes(root)
    assert s['all_methods_sealed_before_external_labels'];rows=run.read(base/'development_v1/rows.json');assert len(rows)==432
    cases={(c['seed'],c['method']):c for c in p['cases']};groups=defaultdict(dict);counts=Counter();stats=defaultdict(lambda:Counter());times=defaultdict(list);curve=[]
    for r in rows:
        directory=root/r['directory']
        for f,h in r['files'].items():assert run.sha(directory/f)==h
        a=load(directory/'arrays.npz');m=run.read(directory/'metadata.json');c=cases[r['seed'],r['ordering']];x,v,regs=run.prior.data(root,c)
        assert not any(m[t] for t in ['query_targets_accessed','suffix_accessed','global_bp_used','lp_used'])
        assert m['total_seconds']>m['checkpoints'][-1]['seconds']>0 and m['returned_array_bytes']==sum(t.nbytes for t in a.values())
        assert r['certified']==int((a['first_step']>=0).sum()) and all(np.isfinite(t).all() for t in a.values())
        counts['exact_wide_domain_proofs']+=run.prior.verify_proofs(x,v,regs,a)
        diagnoses=run.read(root/'results/prefix_obstruction/development_v3'/f"{r['seed']}_{r['ordering']}"/'classification.json')
        for i,item in enumerate(diagnoses):
            if item['classification']=='positive_volume':assert a['first_step'][i]<0;counts['known_positive_retained']+=1
            if a['first_step'][i]>=0:assert item['classification']=='infeasible'
        zl,zh,hl,hh=run.model.screen.boxes(v,regs)
        assert np.all(abs(a['final_b'])<=.12) and np.all((a['final_z']>=zl)&(a['final_z']<=zh)) and np.all((a['final_h']>=hl)&(a['final_h']<=hh));counts['final_box_checks']+=1
        prev=-1.
        for cp in m['checkpoints']:
            assert cp['certified']==int(((a['first_step']>=0)&(a['first_step']<=cp['step'])).sum())
            assert cp['seconds']>prev;prev=cp['seconds'];counts['checkpoints']+=1
            if cp['step'] in [0,32,64,128,256,512,1024]:curve.append(dict(seed=r['seed'],ordering=r['ordering'],source_completed=r['source_completed'],method=r['method'],**cp))
        assert m['proposal_rows']==3*len(regs)*len(m['checkpoints']);counts['proposal_accounting']+=1
        key=(r['method'],'complete' if r['source_completed'] else 'budget_exhausted')
        stats[key].update(calls=1,samples=len(regs),certified=r['certified'],initial_certified=int((a['first_step']==0).sum()));times[key].append(r['total_seconds'])
        groups[r['seed'],r['ordering']][r['method']]=(r,a,m)
    paired=[]
    for (seed,order),items in groups.items():
        assert len(items)==9
        for name in ['regional_active128','regional_passive128','regional_instant128','regional_active1024','regional_passive1024','pdhg_box128','pdhg_box1024']:
            for field in ['initial_b','initial_z','initial_h']:
                assert items[name][1][field].tobytes()==items['regional_active128'][1][field].tobytes();counts['same_initial_arrays']+=1
        for field in ['final_b','final_z','final_h']:
            assert items['regional_passive128'][1][field].tobytes()==items['regional_instant128'][1][field].tobytes();counts['passive_instant_primal_arrays']+=1
        for prefix in ['regional_active','regional_passive','pdhg_cold','pdhg_box']:
            short=items[prefix+'128'][1]['first_step'];long=items[prefix+'1024'][1]['first_step']
            assert np.array_equal(short>=0,(long>=0)&(long<=128));counts['long_short_prefix_matches']+=1
        primary=items[run.model.PRIMARY][1]['first_step']>=0
        for name in run.model.METHODS:
            other=items[name][1]['first_step']>=0
            paired.append(dict(seed=seed,ordering=order,source_completed=items[name][0]['source_completed'],control=name,
                primary_only=int((primary&~other).sum()),control_only=int((other&~primary).sum()),both=int((other&primary).sum())))
    resultgroups=[dict(method=name,source_status=status,**dict(g),mean_total_seconds=math.fsum(times[name,status])/len(times[name,status])) for (name,status),g in stats.items()]
    run.save(out/'groups.json',resultgroups);run.save(out/'paired.json',paired);run.save(out/'checkpoints.json',curve)
    result=dict(passed=True,checks=dict(counts),seconds=time.perf_counter()-start,development_summary_sha256=run.sha(base/'development_v1/summary.json'),
        source_sha256=run.sha(Path(__file__)),query_targets_accessed=False,outputs_sha256={f:run.sha(out/f) for f in ['groups.json','paired.json','checkpoints.json']})
    run.save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/run.BASE/'audit_v1';out.mkdir(parents=True,exist_ok=False);audit(root,out)
