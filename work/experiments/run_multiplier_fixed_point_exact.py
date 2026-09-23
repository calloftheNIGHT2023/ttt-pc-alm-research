"""Certify every proposal; no query or teacher accesses."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import multiplier_fixed_point_exact as exact
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    inp=root/'results/multiplier_fixed_point/screen';out=inp.parent/'exact';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    old=json.loads((inp/'protocol.json').read_text());hashes=dict(old['source_sha256'])
    for n,h in hashes.items():assert sha(src/n)==h,n
    hashes.update({n:sha(src/n) for n in ['multiplier_fixed_point_exact.py',Path(__file__).name]})
    p=dict(source_sha256=hashes,screen_summary_sha256=sha(inp/'summary.json'),screen_rows_sha256=sha(inp/'rows.json'),
        recovery='binary-rational observations/bounds; fixed branch linear stationary equations; bound/kink proposals tolerance 1e-10; RREF with saved binary free variables; exact all-block verification',query_targets_accessed=False)
    dump(out/'protocol.json',p);rows=[];start=time.perf_counter()
    for task in json.loads((inp/'rows.json').read_text()):
        path=inp/task['file'];assert sha(path)==task['sha256'];a=np.load(path)
        for r in np.flatnonzero(a['proposal']):
            state,meta=exact.recover(a['x'],a['v'],a['b'][-1,r],a['h'][-1,:,r])
            proof=exact.verify(a['x'],a['v'],*state) if state is not None else dict(accepted=False,reason='recovery failed')
            row=dict(seed=task['seed'],restart=int(r),recovery=meta,proof=proof,state=exact.encode(state) if state is not None else None)
            rows.append(row)
        print(json.dumps(dict(seed=task['seed'],certified=sum(t['proof']['accepted'] for t in rows if t['seed']==task['seed']))),flush=True)
    dump(out/'certificates.json',rows)
    accepted=[dict(seed=r['seed'],restart=r['restart']) for r in rows if r['proof']['accepted']]
    dump(out/'accepted.json',accepted)
    result=dict(passed=True,proposals=len(rows),accepted=len(accepted),rejected=len(rows)-len(accepted),seconds=time.perf_counter()-start,
        max_reconstruction_distance=max(r['recovery'].get('distance_from_saved',0) for r in rows),
        rank_counts={str(k):sum(r['recovery']['rank']==k for r in rows) for k in sorted({r['recovery']['rank'] for r in rows})},
        protocol_sha256=sha(out/'protocol.json'),certificates_sha256=sha(out/'certificates.json'),accepted_sha256=sha(out/'accepted.json'),query_targets_accessed=False)
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
