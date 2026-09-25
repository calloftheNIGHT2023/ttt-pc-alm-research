import argparse,hashlib,json,time,tracemalloc
from pathlib import Path
import numpy as np
import psutil
import archived_region_memory as model
base=model.base
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
proto=json.loads(a.config.read_text());cfg=next(c for c in proto['configs'] if c['name']==a.method)
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto['source_sha256'].items())
seed=5900000;rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24)
v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24);q=rng.uniform(0,1,2048)
before=psutil.Process().memory_info().rss;tracemalloc.start();anchor=np.zeros(4);stages=[]
for n in [4,8,16,24]:
    start=time.perf_counter();predict,anchor,meta=model.fit(x[:n],v[:n],anchor,dict(**cfg,posterior_samples=512));predict(q)
    retained,peak=tracemalloc.get_traced_memory()
    stages.append(dict(n_context=n,tracked_peak_bytes=peak,tracked_retained_bytes=retained,instrumented_seconds=time.perf_counter()-start,
                       archive_numeric_key_bytes=meta['archive_numeric_key_bytes'],persistent_numeric_state_bytes=meta['persistent_state_bytes'],
                       combined_patterns=meta['combined_patterns'],geometry_bank_patterns=meta['geometry_bank_patterns']))
retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop();info=psutil.Process().memory_info()
result=dict(method=a.method,seed=seed,stages=stages,tracemalloc_peak_bytes=peak,tracemalloc_retained_bytes=retained,process_rss_before_bytes=before,
            process_rss_after_bytes=info.rss,process_peak_working_set_bytes=getattr(info,'peak_wset',None),source_hashes_match=True,
            scope='One old development stream in isolated process. Tracemalloc can omit native allocations; process peak includes interpreter/imports. Instrumented time is not a benchmark.')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
