"""Isolated full-pipeline resource audit, never concurrent with timed trials."""
import argparse,hashlib,json,tracemalloc,time
from pathlib import Path
import numpy as np
import psutil
import anchored_input_memory as model
base=model.base
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
assert not args.out.exists();config=json.loads(args.config.read_text());cfg=next(c for c in config['configs'] if c['name']==args.method)
seed=5400000;rng=np.random.default_rng(seed);truth=rng.uniform(-base.PRIOR,base.PRIOR,(config['depth'],config['width']))
weights=base.family.make_weights(config['depth'],config['width']);x=rng.uniform(-1,1,(max(config['stages']),config['width']));q=rng.uniform(-1,1,(config['queries'],config['width']))
v=base.forward(truth[None],x,weights)[0]+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,x.shape)
process=psutil.Process();before=process.memory_info().rss;tracemalloc.start();point=np.zeros_like(truth);stages=[]
for n in config['stages']:
    start=time.perf_counter()
    if cfg['method']=='input':predict,point,meta=model.fit(x[:n],v[:n],point,weights,cfg)
    elif cfg['method']=='prior':predict,meta=base.fit_closed(x[:n],v[:n],weights,cfg)
    else:predict,point,meta=base.fit_internal(x[:n],v[:n],point,weights,cfg)
    instrumented=time.perf_counter()-start;predict(q);current,peak=tracemalloc.get_traced_memory()
    stages.append({'n_context':n,'instrumented_fit_seconds':instrumented,'tracked_peak_so_far':peak,
        'reported_predictor_state_bytes':meta['persistent_state_bytes'],'common_context_bytes':x[:n].nbytes+v[:n].nbytes,
        'common_known_weight_bytes':weights.nbytes,'input_block_cache_bytes':meta.get('input_block_cache_bytes'),
        'major_arrays_subtotal':meta.get('major_arrays_bytes_subtotal')})
current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();info=process.memory_info()
result={'method':args.method,'audit_seed':seed,'stages':stages,'tracemalloc_peak_bytes':peak,'tracemalloc_retained_bytes':current,
    'rss_before_bytes':before,'rss_after_bytes':info.rss,'process_peak_working_set_bytes':getattr(info,'peak_wset',None),
    'scope':'One old development stream; tracemalloc may miss native allocations. Process peak includes interpreter/imports. Instrumented timing is not comparative timing.',
    'source_sha256':{Path(s).name:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in [__file__,model.__file__,model.exact.__file__,base.__file__]}}
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'method':args.method,'peak_bytes':peak}),flush=True)
