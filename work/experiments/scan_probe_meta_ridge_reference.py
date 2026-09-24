"""Complete arithmetic-only meta-ridge census before restarting audit gates.

Collect failures instead of loosening tolerances or choosing a favorable task.
No query targets, quality metrics, method updates, or formal gate release.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import probe_credit_confirmation_suite as suite
import probe_confirmation_precise_heads_v2 as reference
from run_probe_credit_confirmation_v2 import exclusive_json, acquire_lock
from run_multiplier_fixed_point_screen import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();base=root/'results/probe_credit_confirmation'
    inp=base/'predictions';out=base/'meta_ridge_reference_census';assert not out.exists()
    assert not (base/'evaluation_v2').exists() and not (inp/'RUNNING.lock').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    old=base/'precise_head_selftests_v2';tests=json.loads((old/'summary.json').read_text());assert tests['passed'] and tests['tests']==6
    assert sha(old/'tests.json')==tests['tests_sha256']
    for name,digest in tests['source_sha256'].items():assert sha(Path(__file__).parent/name)==digest
    run=json.loads((inp/'summary.json').read_text());assert run['passed'] and not run['query_targets_accessed']
    for name,digest in run['outputs_sha256'].items():assert sha(inp/name)==digest
    protocol=json.loads((inp/'protocol.json').read_text());before=json.loads((inp/'before_query_manifest.json').read_text())
    loaded,manifest=suite.resources.legacy.oldfit.meta.load(root);assert manifest==protocol['checkpoint_manifest']
    configs={r['name']:r for r in suite.catalogue(root) if r['name'] in ['cold__meta_ridge64','cold__meta_ridge128']}
    assert len(configs)==2;hashes=dict(protocol['source_sha256']);hashes.update(tests['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for name,digest in hashes.items():assert sha(Path(__file__).parent/name)==digest
    out.mkdir();lock,identity=acquire_lock(out);(out/'tasks').mkdir()
    exclusive_json(out/'protocol.json',dict(source_sha256=hashes,source_tests_sha256=sha(old/'summary.json'),
        prediction_summary_sha256=sha(inp/'summary.json'),seeds=protocol['seeds'],methods=list(configs),
        query_targets_accessed=False,audit_gate_passed=False,scope='All meta-ridge saved predictions, arithmetic-only; retain every discrepancy'))
    counts=Counter();files={};started=time.perf_counter();max_np=0.;max_ref=0.
    try:
        for seed,entry in zip(protocol['seeds'],before['task_commits']):
            assert entry['seed']==seed and sha(inp/entry['file'])==entry['sha256']
            commit=json.loads((inp/entry['file']).read_text());rowpath=inp/commit['rows_file']
            assert sha(rowpath)==commit['files'][commit['rows_file']]
            rows={r['method']:r for r in json.loads(rowpath.read_text())};records=[]
            for name,cfg in configs.items():
                row=rows[name];assert not row['metadata']['execution_failed']
                assert sha(inp/row['file'])==row['sha256']==commit['files'][row['file']]
                with np.load(inp/row['file']) as z:a={k:z[k].copy() for k in z.files}
                try:
                    prediction,states,note=reference.replay(cfg,a['x_observed'],a['v_observed'],a['q_observed'],a,row['metadata'],loaded)
                    np.testing.assert_allclose(prediction,a['prediction'],rtol=reference.RTOL,atol=reference.ATOL)
                    for k,state in states.items():np.testing.assert_allclose(state,a[k],rtol=reference.RTOL,atol=reference.ATOL)
                    max_np=max(max_np,note['original_numpy_prediction_gap']);max_ref=max(max_ref,float(np.max(abs(prediction-a['prediction']))))
                    counts['passed']+=1;counts['high_precision_references']+=int(note['high_precision_used'])
                    record=dict(method=name,passed=True,reference=note)
                except AssertionError as exc:
                    counts['failed']+=1;record=dict(method=name,passed=False,error=str(exc))
                    print(json.dumps(dict(seed=seed,**record)),flush=True)
                record.update(saved_file=row['file'],saved_sha256=row['sha256']);records.append(record);counts['heads']+=1
            path=out/'tasks'/f'{seed}.json';exclusive_json(path,dict(seed=seed,heads=records,query_targets_accessed=False,prediction_commit_sha256=entry['sha256']))
            files[str(path.relative_to(out))]=sha(path);counts['tasks']+=1
            if counts['tasks']%128==0:print(json.dumps(dict(counts=counts,seconds=time.perf_counter()-started,query_targets_accessed=False)),flush=True)
        assert counts['tasks']==8192 and counts['heads']==16384
        for name,digest in hashes.items():assert sha(Path(__file__).parent/name)==digest
        exclusive_json(out/'files.json',files)
        result=dict(diagnosis_complete=True,all_head_checks_passed=counts['failed']==0,audit_gate_passed=False,
            query_targets_accessed=False,counts=counts,maximum_original_numpy_gap_on_passes=max_np,
            maximum_accepted_reference_gap=max_ref,seconds=time.perf_counter()-started,
            outputs_sha256={n:sha(out/n) for n in ['protocol.json','files.json']})
        exclusive_json(out/'summary.json',result);print(json.dumps(result),flush=True)
    finally:
        if lock.exists() and json.loads(lock.read_text())==identity:lock.unlink()


if __name__=='__main__':main()
