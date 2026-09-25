"""374 frozen matched-state history screening; LP labels evaluated only post-seal."""
import argparse,os,time,traceback
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import numpy as np
import history_certificate_v1 as model
import prefix_obstruction_v1 as proof
from run_prefix_obstruction_v3 import read,sha,save,complete

BASE='results/history_certificate';DESIGN='outputs/ttt-pc-alm-research/374_history_certificate_protocol_v1.md'


def hashes(root):
    p=read(root/'results/prefix_obstruction/development_v3/protocol.json')['source_sha256']
    for file,h in p.items():assert sha(root/file)==h
    out={**p}
    for file in [DESIGN]+['work/experiments/'+n for n in ['history_certificate_v1.py','run_history_certificate_v1.py','audit_history_certificate_v1.py']]:out[file]=sha(root/file)
    return out


def data(root,c):
    file=root/c['directory']/'selection.npz';assert sha(file)==c['selection_sha256']
    with np.load(file,allow_pickle=False) as z:return z['x'],z['v'],z['regions']


def cases(root):
    complete(root/'results/prefix_obstruction/selection_v3')
    return read(root/'results/prefix_obstruction/selection_v3/protocol.json')['cases']


def verify_proofs(x,v,regs,a):
    n=0
    for i in np.flatnonzero(a['first_step']>=0):
        bound=proof.exact_box_bound(x,v,regs[i],a['proof_p'][i],a['proof_a'][i])
        assert bound>0 and F(float(a['proof_lower'][i]))<=bound;n+=1
    return n


def preflight(root,out):
    begin=time.perf_counter();h=hashes(root);counts=Counter();c=cases(root)[0];x,v,regs=data(root,c)
    # Cold128 is exactly the old algorithm's original context and candidate batch.
    a,m=model.fit(x,v,regs,c['seed'],'cold_zero128');old,meta=model.local.pdhg(x,v,np.zeros((len(regs),4)),regs,steps=128,check_every=32)
    assert np.array_equal(old,a['first_step']>=0)
    for i,p in meta['certificates'].items():
        assert a['first_step'][i]==p['step'] and a['proof_lower'][i]==p['lower']
        assert a['proof_p'][i].tobytes()==p['p'].tobytes() and a['proof_a'][i].tobytes()==p['a'].tobytes();counts['legacy_proof_arrays']+=2
    states={}
    for name in model.METHODS:
        aa,mm=model.fit(x,v,regs,c['seed'],name);directory=out/name;directory.mkdir();np.savez_compressed(directory/'arrays.npz',**aa);save(directory/'metadata.json',mm)
        with np.load(directory/'arrays.npz',allow_pickle=False) as z:
            assert z.files==list(aa)
            for key in aa:assert z[key].tobytes()==aa[key].tobytes();counts['archive_arrays']+=1
        assert read(directory/'metadata.json')==mm
        counts['exact_proofs']+=verify_proofs(x,v,regs,aa);states[name]=aa
    for name in model.METHODS:
        if name.startswith('alm_'):
            for key in ['mother_b','mother_h','mother_u','start_b','start_h']:
                assert states[name][key].tobytes()==states['alm_dual128'][key].tobytes();counts['same_alm_state_arrays']+=1
    assert np.array_equal(states['cold_zero128']['first_step']>=0,(states['cold_zero1024']['first_step']>=0)&(states['cold_zero1024']['first_step']<=128))
    rng=np.random.default_rng(374071)
    for n in [4,16,24]:
        xx=rng.uniform(.05,.95,n);b=rng.uniform(-.1,.1,4);vv=model.cold.base.forward(xx,b);reg=model.cold.base.pattern(xx,b)[None].astype(np.uint8)
        for name in model.METHODS:
            aa,mm=model.fit(xx,vv,reg,374071,name);assert aa['first_step'][0]<0;counts['known_feasible_retained']+=1
    files=[str(p.relative_to(out)).replace('\\','/') for p in out.rglob('*') if p.is_file()]
    result=dict(passed=True,checks=dict(counts),seconds=time.perf_counter()-begin,source_sha256=h,outputs_sha256={f:sha(out/f) for f in files})
    assert hashes(root)==h;save(out/'summary.json',result);print(dict(stage='preflight',passed=True,checks=dict(counts),seconds=result['seconds']),flush=True)


def run(root,out):
    begin=time.perf_counter();h=hashes(root);pre=complete(root/BASE/'preflight_v1');assert pre['source_sha256']==h
    cc=cases(root);save(out/'protocol.json',dict(source_sha256=h,cases=cc,methods=model.METHODS,primary=model.PRIMARY,
        selection_protocol_sha256=sha(root/'results/prefix_obstruction/selection_v3/protocol.json'),query_targets_accessed=False))
    rows=[]
    for seed in sorted({c['seed'] for c in cc}):
        jobs=[(c,name) for c in cc if c['seed']==seed for name in model.METHODS]
        for j in np.random.default_rng(np.random.SeedSequence([374929,seed])).permutation(len(jobs)):
            c,name=jobs[int(j)];x,v,regs=data(root,c);original=(x.tobytes(),v.tobytes(),regs.tobytes());a,m=model.fit(x,v,regs,seed,name)
            assert original==(x.tobytes(),v.tobytes(),regs.tobytes())
            directory=out/f"{seed}_{c['method']}_{name}";directory.mkdir();np.savez_compressed(directory/'arrays.npz',**a);save(directory/'metadata.json',m)
            rows.append(dict(seed=seed,ordering=c['method'],method=name,k=c['k'],samples=len(regs),source_completed=c['source_completed'],
                certified=int((a['first_step']>=0).sum()),total_seconds=m['total_seconds'],generation_seconds=m['generation_seconds'],solver_seconds=m['solver_seconds'],
                directory=str(directory.relative_to(root)),files={f:sha(directory/f) for f in ['arrays.npz','metadata.json']}))
        print(dict(stage='screening',tasks=len({r['seed'] for r in rows}),calls=len(rows),seconds=time.perf_counter()-begin),flush=True)
    assert len(rows)==432 and hashes(root)==h;save(out/'rows.json',rows)
    save(out/'summary.json',dict(passed=True,calls=len(rows),seconds=time.perf_counter()-begin,all_methods_sealed_before_external_labels=True,
        query_targets_accessed=False,core_research_goal_complete=False,outputs_sha256={f:sha(out/f) for f in ['protocol.json','rows.json']}))
    print(dict(stage='all_screens_sealed',calls=len(rows),seconds=time.perf_counter()-begin),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','run'],required=True);args=p.parse_args()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    root=Path(__file__).resolve().parents[2];out=root/BASE/('preflight_v1' if args.stage=='preflight' else 'development_v1');out.mkdir(parents=True,exist_ok=False)
    try:globals()[args.stage](root,out)
    except Exception:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
