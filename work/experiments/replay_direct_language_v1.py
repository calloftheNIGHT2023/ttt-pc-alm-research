"""Replay just the new direct baseline from packaged supports, not old archives."""
import argparse
from pathlib import Path
import os
import time
import numpy as np
import direct_language_inference_v1 as direct
from posterior_confirmation_pipeline import discovery_box
from budget_reinvestment_suite_v1 import read,sha,save
from run_support_language_online_v1 import load,same


def run(root,out):
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    source=root/'results/direct_language/development_predictions_v1'
    p=read(source/'protocol.json');rows=read(source/'rows.json');seal=read(source/'before_query_manifest.json')
    assert sha(source/'rows.json')==seal['rows_sha256'] and sha(source/'protocol.json')==seal['protocol_sha256']
    for n,h in p['source_sha256'].items():assert sha(root/n)==h
    selected=sorted([r for r in rows if r['method']=='direct_language_all'],key=lambda r:r['seed'])
    assert [r['seed'] for r in selected]==p['seeds']
    records=[];begin=time.perf_counter()
    with discovery_box(.12):
        for r in selected:
            file=root/r['file'];assert sha(file)==r['sha256']
            with np.load(file,allow_pickle=False) as z:
                x,v,q=(z[n].copy() for n in ['x_observed','v_observed','q_observed'])
            arrays,meta=direct.fit(x,v,q,r['seed'])
            # Only after fitting open archived predictions/candidates for comparison.
            original=load(file);checks=same(arrays,original,list(arrays))
            oldmeta=read(root/r['metadata_file'])['metadata']
            assert meta['positive_modes']==oldmeta['positive_modes'] and meta['pool']['enumerated']==oldmeta['pool']['enumerated']
            records.append(dict(seed=r['seed'],bitwise_arrays=checks,seconds=meta['charged_complete_seconds']))
    save(out/'rows.json',records)
    summary=dict(passed=True,tasks=len(records),bitwise_arrays=sum(r['bitwise_arrays'] for r in records),
        seconds=time.perf_counter()-begin,query_targets_accessed=False,source_sha256=sha(Path(__file__)),
        prediction_summary_sha256=sha(source/'summary.json'),outputs_sha256={'rows.json':sha(out/'rows.json')})
    save(out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/direct_language/replay_v1');args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];out=(root/args.output).resolve()
    assert out.is_relative_to(root) and out!=root
    out.mkdir(parents=True,exist_ok=False);run(root,out)
