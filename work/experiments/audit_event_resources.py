import argparse,hashlib,json,tracemalloc,time
from pathlib import Path
import numpy as np
import psutil
import event_affine_memory as model
import anchored_input_memory as input_model
base=model.base
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();assert not args.out.exists()
cfgall=json.loads(args.config.read_text());cfg=next(c for c in cfgall['configs'] if c['name']==args.method);base.BOUND=cfgall['discovery_bound']
seed=5400000;rng=np.random.default_rng(seed);truth=rng.uniform(-base.PRIOR,base.PRIOR,(3,8));weights=base.family.make_weights(3,8)
x=rng.uniform(-1,1,(24,8));q=rng.uniform(-1,1,(512,8));v=base.forward(truth[None],x,weights)[0]+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,x.shape)
process=psutil.Process();before=process.memory_info().rss;tracemalloc.start();point=np.zeros_like(truth);stages=[]
for n in cfgall['stages']:
    start=time.perf_counter()
    if cfg['method']=='affine':predict,point,meta=model.fit(x[:n],v[:n],point,weights,cfg)
    elif cfg['method']=='input':predict,point,meta=input_model.fit(x[:n],v[:n],point,weights,cfg)
    elif cfg['method']=='prior':predict,meta=base.fit_closed(x[:n],v[:n],weights,cfg)
    else:predict,point,meta=base.fit_internal(x[:n],v[:n],point,weights,cfg)
    instrumented=time.perf_counter()-start;predict(q);current,peak=tracemalloc.get_traced_memory()
    stages.append({'n_context':n,'instrumented_fit_seconds':instrumented,'tracked_peak_so_far':peak,'reported_predictor_bytes':meta['persistent_state_bytes'],
        'common_context_bytes':x[:n].nbytes+v[:n].nbytes,'common_known_weights_bytes':weights.nbytes,
        'major_state_arrays_subtotal':meta.get('major_state_arrays_subtotal'),'bias_workspace_subtotal':meta.get('bias_workspace_bytes_subtotal')})
current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();info=process.memory_info()
result={'method':args.method,'seed':seed,'tracemalloc_peak_bytes':peak,'retained_bytes':current,'rss_before':before,'rss_after':info.rss,
    'process_peak_working_set_bytes':getattr(info,'peak_wset',None),'stages':stages,
    'scope':'one old stream; native allocations may be missed; instrumented time is not comparative timing',
    'source_sha256':{Path(s).name:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in [__file__,model.__file__,model.original.__file__,model.bias_solver.__file__,base.__file__]}}
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'method':args.method,'peak_bytes':peak}),flush=True)
