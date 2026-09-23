"""254: replay every state, support archive and multiplier conservation update."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import run_local_dual_jump_continuation as run
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/local_dual_jump';parent=base/'continuation';out=base/'continuation_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    summary=json.loads((parent/'summary.json').read_text());p=json.loads((parent/'protocol.json').read_text())
    for name,h in p['source_sha256'].items():assert sha(src/name)==h,name
    assert sha(parent/'support_rows.json')==summary['support_rows_sha256'];assert sha(parent/'protocol.json')==summary['protocol_sha256']
    for name,h in summary['geometry_sha256'].items():assert sha(parent/name)==h,name
    hashes={**p['source_sha256'],Path(__file__).name:sha(Path(__file__))};dump(out/'protocol.json',dict(source_sha256=hashes,parent_summary_sha256=sha(parent/'summary.json'),query_targets_accessed=False))
    lookup={(r['seed'],r['restart'],r['method']):r for r in json.loads((base/'transfer/support_rows.json').read_text())};configs={c['name']:c for c in p['configs']}
    counts=Counter();begin=time.perf_counter();max_credit_gap=0.
    with run.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for row in json.loads((parent/'support_rows.json').read_text()):
            assert sha(parent/row['file'])==row['sha256'];a=np.load(parent/row['file']);cfg=configs[row['method']]
            x,v,b,h,u,best,origin,visited=run.initial(base/'transfer',lookup,row['seed'],cfg)
            for name,value in dict(x=x,v=v,initial_b=b,initial_h=h,initial_u=u,incumbent=best,origin=origin,atomic_visited=visited).items():
                assert value.tobytes()==a[name].tobytes(),(row['file'],name);counts['initial_arrays']+=1
            if cfg.get('bp'):
                answer,history,_=run.cold.run_bp(b,x,v,cfg['method'],cfg['steps'],True);run.cold.frozen.retain(best,answer,x,v,np.zeros(4));answer=best.copy();history['best']=answer[None]
            else:answer,history,_=run.local_run(b,h,u,best,x,v,cfg)
            for name,value in history.items():assert value.tobytes()==a[name].tobytes(),(row['file'],name);counts['replayed_arrays']+=1
            assert answer.tobytes()==a['best_bank'].tobytes()
            retained=a['incumbent'].copy()
            for step,bb in enumerate(a['b']):
                run.cold.frozen.retain(retained,bb,x,v,np.zeros(4));counts['support_retention_batches']+=1
                if not cfg.get('bp'):assert retained.tobytes()==a['best'][step].tobytes(),(row['file'],step)
            assert retained.tobytes()==a['best_bank'].tobytes(),row['file']
            if not cfg.get('bp'):
                rate=.5 if cfg['method']=='alm' else 0.
                for step in range(1,len(a['b'])):
                    previous=x
                    for j in range(4):
                        residual=a['h'][step,j]-run.cold.base.g(previous+a['b'][step,:,j,None])
                        expected=a['u'][step-1,j]+rate*residual
                        gap=float(np.max(abs(expected-a['u'][step,j])));max_credit_gap=max(max_credit_gap,gap);assert gap==0.
                        previous=a['h'][step,j];counts['dual_update_batches']+=1
            counts['task_methods']+=1
            if counts['task_methods']%13==0:print(json.dumps(dict(seed=row['seed'],counts=counts,seconds=time.perf_counter()-begin)),flush=True)
    result=dict(passed=True,counts=counts,max_credit_update_gap=max_credit_gap,seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),query_targets_accessed=False)
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
