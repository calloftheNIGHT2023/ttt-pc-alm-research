"""372 preflight assertion repair; immutable selection -> all local screens -> offline exact diagnosis."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import os,time,traceback
import numpy as np
import prefix_obstruction_v1 as model
from budget_reinvestment_suite_v1 import read,sha,save,complete

BASE='results/prefix_obstruction';METHODS=['observed','farthest_x','alm_dual']
DESIGN='outputs/ttt-pc-alm-research/372_prefix_obstruction_protocol_v1.md'


def hashes(root):
    parent=read(root/'results/multiplier_constraint_order/development_v1/protocol.json')['source_sha256']
    for p,h in parent.items():assert sha(root/p)==h
    h={**parent}
    for p in [DESIGN,'outputs/ttt-pc-alm-research/372_preflight_assertion_repair_v2.md']+['work/experiments/'+n for n in ['prefix_obstruction_v1.py','run_prefix_obstruction_v1.py',
        'audit_prefix_obstruction_v1.py','run_prefix_obstruction_v2.py','audit_prefix_obstruction_v2.py','complete_credit_mode_geometry_v1.py','enumerate_support_modes.py']]:h[p]=sha(root/p)
    return h


def prepare(root,out):
    complete(root/'results/multiplier_constraint_order/development_v1');complete(root/'results/multiplier_constraint_order/audit_v1')
    rows=read(root/'results/multiplier_constraint_order/development_v1/rows.json');selected=[]
    for r in sorted(rows,key=lambda t:(t['seed'],t['method'])):
        if r['n']!=24 or r['method'] not in METHODS:continue
        source=root/r['directory']
        for f,h in r['files'].items():assert sha(source/f)==h
        meta=read(source/'metadata.json');k=meta['complete_stages'];assert k>0
        with np.load(source/'arrays.npz',allow_pickle=False) as z:
            regs=z['prefix_'+str(k)];x=z['x_observed'][:k];v=z['v_observed'][:k]
            count=len(regs);assert count>0
            indices=np.sort(np.random.default_rng(np.random.SeedSequence([372171,r['seed'],METHODS.index(r['method'])])).choice(count,min(64,count),replace=False))
            arrays=dict(x=x,v=v,regions=regs[indices],source_indices=indices,observation_ids=z['support_order'][:k])
        directory=out/f"{r['seed']}_{r['method']}";directory.mkdir();np.savez_compressed(directory/'selection.npz',**arrays)
        selected.append(dict(seed=r['seed'],method=r['method'],k=k,source_completed=r['completed'],source_count=count,
            samples=len(indices),source_directory=r['directory'],source_files=r['files'],directory=str(directory.relative_to(root)),
            selection_sha256=sha(directory/'selection.npz')))
    assert len(selected)==48
    old=read(root/BASE/'selection_v1/protocol.json');replayed=0
    for current,previous in zip(selected,old['cases']):
        assert all(current[k]==previous[k] for k in ['seed','method','k','source_completed','source_count','samples','source_directory','source_files'])
        with np.load(root/current['directory']/'selection.npz',allow_pickle=False) as a,np.load(root/previous['directory']/'selection.npz',allow_pickle=False) as b:
            assert a.files==b.files
            for key in a.files:assert np.array_equal(a[key],b[key]);replayed+=1
    save(out/'repair_replay.json',dict(passed=True,selection_arrays=replayed,unchanged_cases=48))
    save(out/'protocol.json',dict(source_sha256=hashes(root),cases=selected,old_development_tasks=True,query_targets_accessed=False))
    save(out/'summary.json',dict(passed=True,cases=len(selected),samples=sum(r['samples'] for r in selected),
        outputs_sha256={'protocol.json':sha(out/'protocol.json')}))
    print(dict(stage='prepare',cases=len(selected),samples=sum(r['samples'] for r in selected)),flush=True)


def preflight(root,out):
    checks=Counter();rng=np.random.default_rng(372071);certs=[]
    for n in [4,16,24]:
        for rep in range(2):
            x=rng.uniform(.05,.95,n);b=rng.uniform(-.1,.1,4);h=x.copy();regs=[]
            for bias in b:
                z=h+bias;regs.append(np.searchsorted([0.,.5,1.],z,side='right'));h=np.maximum(0.,1.-abs(2*z-1.))
            reg=np.array(regs,np.uint8);aa,rr=model.matrices(x,h,reg);answer=model.classify(aa,rr)
            assert answer['classification']=='positive_volume'
            checks['exact_certificates']+=model.proof.verify_certificates(aa,rr,answer)
            a,m=model.probes(x,h,reg[None]);assert not a['c100_reject'][0] and not a['pdhg128_reject'][0]
            checks['legal_witnesses']+=1;certs.append(answer)
    # Contradictory observed outputs for equal inputs, identical supplied paths.
    x=np.array([.2,.2]);v=np.array([.2,.8]);reg=np.ones((4,2),np.uint8)
    aa,rr=model.matrices(x,v,reg);answer=model.classify(aa,rr);assert answer['classification']=='infeasible'
    checks['exact_certificates']+=model.proof.verify_certificates(aa,rr,answer);certs.append(answer)
    arrays,meta=model.probes(x,v,reg[None]);checks['contradiction_c100_detected']=int(arrays['c100_reject'][0]);checks['contradiction_pdhg_detected']=int(arrays['pdhg128_reject'][0])
    for i,p in meta['pdhg']['certificates'].items():
        bound=model.exact_box_bound(x,v,reg,p['p'],p['a']);assert F(p['lower'])<=bound and bound>0;checks['pdhg_exact']+=1
    aa,rr=model.proof.box_rows(4);aa.extend([[F(1),F(0),F(0),F(0)],[-F(1),F(0),F(0),F(0)]]);rr.extend([F(0),F(0)])
    answer=model.classify(aa,rr);assert answer['classification']=='zero_volume_or_empty'
    checks['exact_certificates']+=model.proof.verify_certificates(aa,rr,answer);certs.append(answer)
    save(out/'cases.json',certs);save(out/'summary.json',dict(passed=True,checks=dict(checks),source_sha256=hashes(root),outputs_sha256={'cases.json':sha(out/'cases.json')}))
    print(dict(stage='preflight',passed=True,checks=dict(checks)),flush=True)


def run(root,out):
    start=time.perf_counter();source=root/BASE/'selection_v2';complete(source);complete(root/BASE/'preflight_v2')
    protocol=read(source/'protocol.json');assert protocol['source_sha256']==hashes(root)==read(root/BASE/'preflight_v2/summary.json')['source_sha256']
    save(out/'protocol.json',dict(selection_protocol_sha256=sha(source/'protocol.json'),source_sha256=hashes(root),query_targets_accessed=False))
    local=[]
    for c in protocol['cases']:
        directory=root/c['directory'];assert sha(directory/'selection.npz')==c['selection_sha256']
        with np.load(directory/'selection.npz',allow_pickle=False) as z:x=z['x'];v=z['v'];regs=z['regions']
        arrays,meta=model.probes(x,v,regs);target=out/directory.name;target.mkdir()
        np.savez_compressed(target/'screens.npz',**arrays);save(target/'screens.json',meta)
        local.append(dict(case=c,directory=str(target.relative_to(root)),files={f:sha(target/f) for f in ['screens.npz','screens.json']}))
    save(out/'screen_seal.json',dict(all_screens_completed=True,cases=local,query_targets_accessed=False,lp_labels_accessed=False))
    print(dict(stage='screens_sealed',cases=len(local),seconds=time.perf_counter()-start),flush=True)
    rows=[];checks=Counter()
    for item in local:
        c=item['case'];directory=root/c['directory'];target=root/item['directory']
        with np.load(directory/'selection.npz',allow_pickle=False) as z:x=z['x'];v=z['v'];regs=z['regions']
        with np.load(target/'screens.npz',allow_pickle=False) as z:c100=z['c100_reject'];pdhg=z['pdhg128_reject']
        results=[];counts=Counter();confirmed=Counter();meta=read(target/'screens.json')
        for i,reg in enumerate(regs):
            aa,rr=model.matrices(x,v,reg);answer=model.classify(aa,rr);answer['sample_index']=i
            checks['certificates']+=model.proof.verify_certificates(aa,rr,answer);counts[answer['classification']]+=1
            if c100[i]:confirmed['c100_'+answer['classification']]+=1
            if pdhg[i]:
                p=meta['pdhg']['certificates'][str(i)];bound=model.exact_box_bound(x,v,reg,np.array(p['p']),np.array(p['a']))
                assert F(p['lower'])<=bound and bound>0;checks['pdhg_exact']+=1
                confirmed['pdhg_'+answer['classification']]+=1
                assert answer['classification']!='positive_volume'
            if c100[i]:assert answer['classification']!='positive_volume'
            results.append(answer)
        save(target/'classification.json',results)
        row=dict(seed=c['seed'],method=c['method'],source_completed=c['source_completed'],k=c['k'],source_count=c['source_count'],samples=len(regs),counts=dict(counts),
            screening_counts=dict(confirmed),c100_rejected=int(c100.sum()),pdhg_rejected=int(pdhg.sum()),
            c100_seconds=meta['c100_seconds'],pdhg_seconds=meta['pdhg128_seconds'],
            classification_seconds=sum(t['seconds'] for t in results),lp_calls=sum(t['lp_calls'] for t in results),directory=item['directory'])
        rows.append(row);print(dict(stage='diagnosis',cases=len(rows),samples=sum(r['samples'] for r in rows),seconds=time.perf_counter()-start),flush=True)
    assert hashes(root)==protocol['source_sha256'];save(out/'rows.json',rows)
    files=[str(p.relative_to(out)).replace('\\','/') for p in out.rglob('*') if p.is_file()]
    save(out/'summary.json',dict(passed=True,cases=len(rows),samples=sum(r['samples'] for r in rows),checks=dict(checks),seconds=time.perf_counter()-start,
        classification_counts=dict(sum((Counter(r['counts']) for r in rows),Counter())),query_targets_accessed=False,core_research_goal_complete=False,
        outputs_sha256={f:sha(out/f) for f in files}))
    print(dict(stage='complete',cases=len(rows),counts=dict(sum((Counter(r['counts']) for r in rows),Counter()))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','preflight','run'],required=True);args=parser.parse_args()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    root=Path(__file__).resolve().parents[2];folder={'prepare':'selection_v2','preflight':'preflight_v2','run':'development_v2'}[args.stage]
    out=root/BASE/folder;out.mkdir(parents=True,exist_ok=False)
    try:globals()[args.stage](root,out)
    except Exception:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
