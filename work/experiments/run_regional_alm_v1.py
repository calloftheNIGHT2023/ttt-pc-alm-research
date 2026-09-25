"""376 exact block checks and immutable fixed-region component experiment."""
import argparse,os,time,traceback
from pathlib import Path
from collections import Counter
import numpy as np
import regional_alm_v1 as model
import run_history_certificate_v1 as prior
from run_prefix_obstruction_v3 import read,sha,save,complete

BASE='results/regional_alm';DESIGN='outputs/ttt-pc-alm-research/376_regional_alm_protocol_v1.md'


def hashes(root):
    h=read(root/'results/history_certificate/development_v1/protocol.json')['source_sha256']
    for p,s in h.items():assert sha(root/p)==s
    h={**h}
    for p in [DESIGN]+['work/experiments/'+n for n in ['regional_alm_v1.py','run_regional_alm_v1.py','audit_regional_alm_v1.py']]:h[p]=sha(root/p)
    return h


def block_tests():
    rng=np.random.default_rng(376071);counts=Counter();maximum=0.
    def clamp(x,l,h):return max(float(l),min(float(h),x))
    for n in [2,7,24]:
        for rep in range(3):
            x=rng.uniform(0,1,n);v=rng.uniform(0,1,n);regs=rng.integers(0,4,(3,4,n),dtype=np.uint8)
            zl,zh,hl,hh=model.screen.boxes(v,regs);s=np.array([0.,2.,-2.,0.])[regs];c=np.array([0.,0.,2.,0.])[regs]
            b=rng.uniform(-.12,.12,(3,4));z=rng.uniform(size=regs.shape)*(zh-zl)+zl;h=rng.uniform(size=regs.shape)*(hh-hl)+hl
            p=rng.normal(size=regs.shape);a=rng.normal(size=regs.shape)
            for label in ['b','z','h']:
                old={'b':b.copy(),'z':z.copy(),'h':h.copy()};before=model.energy(x,s,c,b,z,h,p,a)
                scalar=old[label].copy()
                if label=='b':
                    for r in range(3):
                        for j in range(4):scalar[r,j]=clamp((sum(float(z[r,j,i]-(x[i] if j==0 else h[r,j-1,i])+p[r,j,i]) for i in range(n))+.01*b[r,j])/(n+.01),-.12,.12)
                    b=model.block_b(x,b,z,h,p)
                elif label=='z':
                    for r in range(3):
                        for j in range(4):
                            for i in range(n):
                                prev=float(x[i]) if j==0 else float(h[r,j-1,i]);slope=float(s[r,j,i]);num=prev+b[r,j]-p[r,j,i]+slope*(h[r,j,i]-c[r,j,i]+a[r,j,i])+.01*z[r,j,i]
                                scalar[r,j,i]=clamp(num/(1+slope*slope+.01),zl[r,j,i],zh[r,j,i])
                    z=model.block_z(x,b,z,h,p,a,s,c,zl,zh)
                else:
                    for r in range(3):
                        for j in range(4):
                            for i in range(n):
                                num=s[r,j,i]*z[r,j,i]+c[r,j,i]-a[r,j,i]+.01*h[r,j,i];den=1.01
                                if j<3:num+=z[r,j+1,i]-b[r,j+1]+p[r,j+1,i];den+=1
                                scalar[r,j,i]=clamp(num/den,hl[r,j,i],hh[r,j,i])
                    h=model.block_h(b,z,h,p,a,s,c,hl,hh)
                value={'b':b,'z':z,'h':h}[label];maximum=max(maximum,float(abs(value-scalar).max()));assert maximum<2e-12
                change=((value-old[label])**2).reshape(3,-1).sum(1);after=model.energy(x,s,c,b,z,h,p,a)+.005*change
                assert np.all(after<=before+1e-10*(1+abs(before)));counts['block_energy_inequalities']+=3;counts['scalar_entries']+=value.size
    return dict(checks=dict(counts),maximum_scalar_gap=maximum)


def preflight(root,out):
    start=time.perf_counter();h=hashes(root);checks=Counter();block=block_tests();save(out/'block_tests.json',block)
    c=prior.cases(root)[0];x,v,regs=prior.data(root,c);states={}
    legacy,lm=model.screen.pdhg(x,v,np.zeros((len(regs),4)),regs,steps=128,check_every=32)
    for name in model.METHODS:
        a,m=model.solve(x,v,regs,name);directory=out/name;directory.mkdir();np.savez_compressed(directory/'arrays.npz',**a);save(directory/'metadata.json',m)
        with np.load(directory/'arrays.npz',allow_pickle=False) as z:
            for key in a:assert a[key].tobytes()==z[key].tobytes();checks['archive_arrays']+=1
        assert read(directory/'metadata.json')==m
        checks['exact_proofs']+=prior.verify_proofs(x,v,regs,a);states[name]=a
    assert np.all(~legacy|(states['pdhg_cold128']['first_step']>=0));checks['legacy_rejections_retained']=int(legacy.sum())
    for name in ['regional_active128','regional_passive128','regional_instant128','pdhg_box128']:
        for field in ['initial_b','initial_z','initial_h']:
            assert states[name][field].tobytes()==states['regional_active128'][field].tobytes();checks['same_initial_arrays']+=1
    for field in ['final_b','final_z','final_h']:
        assert states['regional_passive128'][field].tobytes()==states['regional_instant128'][field].tobytes();checks['passive_instant_primal_arrays']+=1
    rng=np.random.default_rng(376101)
    for n in [4,16,24]:
        xx=rng.uniform(.05,.95,n);b=rng.uniform(-.1,.1,4);vv=model.screen.base.forward(xx,b);reg=model.screen.base.pattern(xx,b)[None].astype(np.uint8)
        for name in model.METHODS:
            aa,mm=model.solve(xx,vv,reg,name);assert aa['first_step'][0]<0;checks['known_feasible_retained']+=1
    assert hashes(root)==h;files=[str(p.relative_to(out)).replace('\\','/') for p in out.rglob('*') if p.is_file()]
    save(out/'summary.json',dict(passed=True,checks=dict(checks),block_tests=block,source_sha256=h,seconds=time.perf_counter()-start,outputs_sha256={f:sha(out/f) for f in files}))
    print(dict(stage='preflight',passed=True,checks=dict(checks),block=block,seconds=time.perf_counter()-start),flush=True)


def run(root,out):
    start=time.perf_counter();h=hashes(root);assert complete(root/BASE/'preflight_v1')['source_sha256']==h;cases=prior.cases(root)
    save(out/'protocol.json',dict(source_sha256=h,methods=model.METHODS,primary=model.PRIMARY,cases=cases,query_targets_accessed=False))
    rows=[]
    for seed in sorted({c['seed'] for c in cases}):
        jobs=[(c,name) for c in cases if c['seed']==seed for name in model.METHODS]
        for j in np.random.default_rng(np.random.SeedSequence([376929,seed])).permutation(len(jobs)):
            c,name=jobs[int(j)];x,v,regs=prior.data(root,c);before=(x.tobytes(),v.tobytes(),regs.tobytes());a,m=model.solve(x,v,regs,name)
            assert before==(x.tobytes(),v.tobytes(),regs.tobytes())
            directory=out/f"{seed}_{c['method']}_{name}";directory.mkdir();np.savez_compressed(directory/'arrays.npz',**a);save(directory/'metadata.json',m)
            rows.append(dict(seed=seed,ordering=c['method'],method=name,k=c['k'],source_completed=c['source_completed'],samples=len(regs),
                certified=int((a['first_step']>=0).sum()),total_seconds=m['total_seconds'],steps=m['steps'],directory=str(directory.relative_to(root)),
                files={f:sha(directory/f) for f in ['arrays.npz','metadata.json']}))
        print(dict(stage='screening',tasks=len({r['seed'] for r in rows}),calls=len(rows),seconds=time.perf_counter()-start),flush=True)
    assert len(rows)==432 and hashes(root)==h;save(out/'rows.json',rows)
    result=dict(passed=True,calls=len(rows),seconds=time.perf_counter()-start,all_methods_sealed_before_external_labels=True,
        query_targets_accessed=False,core_research_goal_complete=False,outputs_sha256={f:sha(out/f) for f in ['protocol.json','rows.json']})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','run'],required=True);args=p.parse_args()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    root=Path(__file__).resolve().parents[2];out=root/BASE/('preflight_v1' if args.stage=='preflight' else 'development_v1');out.mkdir(parents=True,exist_ok=False)
    try:globals()[args.stage](root,out)
    except Exception:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
