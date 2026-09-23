"""Preflight all33 fresh pipelines on the old frozen preflight task."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import dual_jump_resource_suite as suite
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/dual_jump_query';out=base/'live_primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert json.loads((base/'audit/summary.json').read_text())['passed'];hashes=json.loads((base/'audit/protocol.json').read_text())['source_sha256']
    hashes.update({n:sha(src/n) for n in ['live_dual_jump_query.py','dual_jump_resource_suite.py',Path(__file__).name]})
    for n,h in hashes.items():assert sha(src/n)==h,n
    loaded,manifest=suite.oldfit.meta.load(root);p=dict(source_sha256=hashes,query_audit_sha256=sha(base/'audit/summary.json'),configs=suite.configs(),seed=5900001,checkpoint_manifest=manifest,query_targets_accessed=False)
    dump(out/'protocol.json',p);x,v=observations(p['seed']);q=np.linspace(0,1,257);rows=[];start=time.perf_counter()
    with suite.live.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for cfg in p['configs']:
            arrays,meta=suite.fit(cfg,x[:4],v[:4],q,p['seed'],loaded);suite.check(root,p['seed'],cfg,arrays,meta);rows.append(dict(method=cfg['name'],seconds=meta['complete_call_seconds']))
            print(json.dumps(rows[-1]),flush=True)
    dump(out/'rows.json',rows);result=dict(passed=True,bytewise_full_pipeline_predictors=len(rows),seconds=time.perf_counter()-start,protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'))
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
