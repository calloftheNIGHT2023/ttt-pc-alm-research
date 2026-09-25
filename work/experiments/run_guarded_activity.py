import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import guarded_activity_memory as candidate
import common_cell_readout as geometry
base=candidate.base


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    cfg=json.loads(args.config.read_text());base.BOUND=cfg['discovery_bound'];args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists()
    sources=[Path(s) for s in [__file__,candidate.__file__,candidate.reference.__file__,candidate.reference.activity.__file__,
             candidate.original.__file__,candidate.original.original.__file__,candidate.original.first.__file__,candidate.original.bias_solver.__file__,
             base.__file__,base.family.__file__,geometry.__file__]]
    proto=dict(**cfg,source_sha256={s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
               config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),scope='observed development; bounded joint local energy safeguards; common global readout')
    (args.out/'protocol.json').write_text(json.dumps(proto,indent=2),encoding='utf-8');weights=base.family.make_weights(cfg['depth'],cfg['width']);rows=[];order=np.random.default_rng(746619)
    for seed in range(cfg['seed0'],cfg['seed0']+cfg['count']):
        rng=np.random.default_rng(seed);truth=rng.uniform(-base.PRIOR,base.PRIOR,(cfg['depth'],cfg['width']))
        x=rng.uniform(-1,1,(max(cfg['stages']),cfg['width']));q=rng.uniform(-1,1,(cfg['queries'],cfg['width']))
        v=base.forward(truth[None],x,weights)[0]+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,x.shape);target=base.forward(truth[None],q,weights)[0]
        for ci in order.permutation(len(cfg['configs'])):
            c=cfg['configs'][ci];anchor=np.zeros_like(truth)
            for n in cfg['stages']:
                before=time.perf_counter()
                if c['method']=='guarded':_,raw,fit_meta=candidate.fit(x[:n],v[:n],anchor,weights,c)
                elif c['method']=='memory':_,raw,fit_meta=candidate.reference.fit(x[:n],v[:n],anchor,weights,c)
                else:_,raw,fit_meta=base.fit_internal(x[:n],v[:n],anchor,weights,c)
                point,region,pm=geometry.project(raw,anchor,x[:n],v[:n],weights);state,sm=geometry.fiber(point,region,x[:n],v[:n],weights)
                adaptation=time.perf_counter()-before;before=time.perf_counter();prediction,rm=geometry.read(q,weights,point,state,'mean');read_seconds=time.perf_counter()-before
                support,_=geometry.read(x[:n],weights,point,state,'mean');error=float(np.max(np.abs(support-v[:n])))
                anchor=point.copy();rows.append(dict(method=c['name'],seed=seed,n_context=n,query_mse=float(np.mean((prediction-target)**2)),
                    support_max_error=error,support_feasible=bool(error<=base.EPS+base.TOL),adaptation_seconds=adaptation,read_queries_seconds=read_seconds,
                    selected_raw=raw.tolist(),anchor_output=point.tolist(),fit_meta=fit_meta,**pm,**sm,**rm))
        (args.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(completed=seed-cfg['seed0']+1,total=cfg['count'],rows=len(rows))),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
