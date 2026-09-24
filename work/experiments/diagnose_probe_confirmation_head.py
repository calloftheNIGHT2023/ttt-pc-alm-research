"""Identify failed independent head arithmetic from saved observations only."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import probe_credit_confirmation_suite as suite
import probe_confirmation_independent_heads as independent
from run_probe_credit_confirmation_v2 import verify_commit, exclusive_json
from run_multiplier_fixed_point_screen import sha


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve(); base=root/'results/probe_credit_confirmation'
    inp=base/'predictions'; audit=base/'prediction_audit'; out=base/'head_discrepancy_diagnosis'
    assert not out.exists() and not (audit/'RUNNING.lock').exists() and not (base/'evaluation_v2').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    p=json.loads((inp/'protocol.json').read_text()); completed=sorted(int(path.stem) for path in (audit/'tasks').glob('*.json'))
    assert completed==p['seeds'][:len(completed)]; seed=p['seeds'][len(completed)]
    for name,digest in p['source_sha256'].items(): assert sha(Path(__file__).parent/name)==digest
    committed=verify_commit(inp,seed,p['methods'],sha(inp/'protocol.json'))
    c=json.loads((inp/committed['file']).read_text()); rows=json.loads((inp/c['rows_file']).read_text())
    configs={r['name']:r for r in suite.catalogue(root)}
    loaded,manifest=suite.resources.legacy.oldfit.meta.load(root); assert manifest==p['checkpoint_manifest']
    results=[]; arrays={}; started=time.perf_counter()
    for row in rows:
        if row['metadata']['method_kind']!='head': continue
        name=row['method']; cfg=configs[name]
        with np.load(inp/row['file']) as z: a={k:z[k].copy() for k in z.files}
        x,v,q=a['x_observed'],a['v_observed'],a['q_observed']
        prediction,states=independent.replay(cfg,x,v,q,a,row['metadata'],loaded)
        gap=abs(prediction-a['prediction']); threshold=1e-9+1e-10*abs(a['prediction'])
        fresh,metadata=suite.fit(cfg,x,v,q,seed,loaded)
        equal={k:fresh[k].tobytes()==a[k].tobytes() for k in fresh}
        record=dict(method=name,configuration=cfg,saved_file=row['file'],saved_sha256=row['sha256'],
            failed_prediction_tolerance=bool(np.any(gap>threshold)),mismatched_points=int(np.sum(gap>threshold)),
            maximum_gap=float(gap.max()),worst_index=int(gap.argmax()),
            original_replay_array_equal=equal,
            state_gaps={k:float(np.max(abs(value-a[k]))) for k,value in states.items()},
            independent_replay_exception=None)
        assert all(equal.values()), (name,equal)
        if record['failed_prediction_tolerance']:
            arrays[name+'__reference']=prediction
            for k,value in a.items(): arrays[name+'__'+k]=value
        results.append(record); print(json.dumps(record),flush=True)
    out.mkdir(); exclusive_json(out/'heads.json',results)
    with (out/'arrays.npz').open('xb') as f: np.savez_compressed(f,**arrays)
    result=dict(diagnosis_complete=True,audit_gate_passed=False,seed=seed,completed_tasks=len(completed),
        failed_heads=[r['method'] for r in results if r['failed_prediction_tolerance']],
        original_prediction_unchanged=True,query_targets_accessed=False,seconds=time.perf_counter()-started,
        source_sha256=sha(Path(__file__)),prediction_commit_sha256=committed['sha256'],
        original_audit_protocol_sha256=sha(audit/'protocol.json'),
        outputs_sha256={n:sha(out/n) for n in ['heads.json','arrays.npz']})
    exclusive_json(out/'summary.json',result); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
