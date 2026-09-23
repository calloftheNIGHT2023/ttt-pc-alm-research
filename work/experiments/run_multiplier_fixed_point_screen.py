"""Round245 support-only screen; unchanged frozen nodual512 arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import split_activity_mode_trace as trace
from run_independent_hybrid_memory import observations

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')

class Capture:
    def __init__(self):self.b=[];self.h=[]
    def parameter(self,b,step):pass
    def activity(self,b,h,u,step,phase):
        assert not np.any(u)
        if phase in ['initial','after_parameters_before_dual']:
            assert len(self.b)==step
            self.b.append(b.copy());self.h.append(h.copy())

def run(x,v,starts):
    observer=Capture();anchor=np.zeros(4)
    best=trace.refine(starts,x,v,anchor,'nodual',512,observer)
    b=np.array(observer.b);h=np.array(observer.h)
    assert b.shape==(513,17,4) and h.shape==(513,4,17,4)
    expected=trace.original(starts,x,v,anchor,'nodual',512)
    assert best.tobytes()==expected.tobytes()
    delta=np.maximum(np.max(abs(np.diff(b[-3:],axis=0)),axis=2),np.max(abs(np.diff(h[-3:],axis=0)),axis=(1,3)))
    residual=[];prev=np.broadcast_to(x,(len(starts),len(x)))
    for j in range(4):
        residual.append(h[-1,j]-trace.base.g(prev+b[-1,:,j,None]));prev=h[-1,j]
    residual=np.array(residual);maximum=np.max(abs(residual),axis=(0,2))
    err,_=trace.base.score(best,x,v,anchor)
    proposal=(np.max(delta,axis=0)<=1e-12)&(maximum>1e-6)&(err>trace.base.EPS+1e-6)
    arrays=dict(x=x,v=v,starts=starts,b=b,h=h,best=best,delta=delta,residual=residual,best_error=err,proposal=proposal)
    rows=[dict(restart=i,proposal=bool(proposal[i]),last_two_changes=delta[:,i].tolist(),residual_max=float(maximum[i]),best_error=float(err[i])) for i in range(17)]
    return arrays,rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    out=root/'results/multiplier_fixed_point/screen';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    src=Path(__file__).parent;parent=root/'results/round_244_audit.json'
    old=json.loads((root/'results/band_conditioned_online/development/protocol.json').read_text());audit=json.loads(parent.read_text())
    hashes=dict(old['source_sha256']);hashes.update(audit['extra_source_sha256'])
    for n,h in hashes.items():assert sha(src/n)==h,n
    hashes[Path(__file__).name]=sha(Path(__file__))
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=sha(root/'outputs/ttt-pc-alm-research/245_multiplier_fixed_point_escape_protocol.md'),
        seeds=list(range(5900000,5900016)),n=4,sweeps=512,restarts=17,start_seed=731,bound=.12,epsilon=.001,trust=.01,
        thresholds=dict(change=1e-12,residual=1e-6,best_error=.001001),query_targets_accessed=False)
    dump(out/'protocol.json',p);starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(16,4))]
    results=[];begin=time.perf_counter()
    with trace.old.core.pipeline.discovery_box(.12):
        for seed in p['seeds']:
            xx,vv=observations(seed);start=time.perf_counter();arrays,rows=run(xx[:4],vv[:4],starts)
            file=out/f'{seed}.npz';np.savez_compressed(file,**arrays)
            result=dict(seed=seed,seconds_including_original_replay=time.perf_counter()-start,file=file.name,sha256=sha(file),restarts=rows)
            results.append(result);dump(out/'rows.json',results)
            print(json.dumps(dict(seed=seed,proposals=sum(r['proposal'] for r in rows),seconds=result['seconds_including_original_replay'])),flush=True)
    for n,h in hashes.items():assert sha(src/n)==h,n
    summary=dict(passed=True,tasks=16,restarts=272,bitwise_original_replays=16,proposals=sum(r['proposal'] for t in results for r in t['restarts']),
        total_seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),query_targets_accessed=False)
    dump(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
