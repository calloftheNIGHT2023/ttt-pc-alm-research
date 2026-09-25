import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import affine_eliminated_memory as candidate
import anchored_input_memory as input_model
base=candidate.base


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    cfg=json.loads(args.config.read_text());args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists();base.BOUND=cfg['discovery_bound']
    sources=[Path(__file__),Path(candidate.__file__),Path(candidate.first.__file__),Path(candidate.bias_solver.__file__),Path(input_model.__file__),Path(base.__file__),Path(base.family.__file__)]
    proto={**cfg,'source_sha256':{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'scope':'observed development streams; all algorithms use same known prior box; not confirmation'}
    (args.out/'protocol.json').write_text(json.dumps(proto,indent=2),encoding='utf-8');weights=base.family.make_weights(cfg['depth'],cfg['width']);rows=[];order=np.random.default_rng(644619)
    for seed in range(cfg['seed0'],cfg['seed0']+cfg['count']):
        rng=np.random.default_rng(seed);truth=rng.uniform(-base.PRIOR,base.PRIOR,(cfg['depth'],cfg['width']))
        x=rng.uniform(-1,1,(max(cfg['stages']),cfg['width']));q=rng.uniform(-1,1,(cfg['queries'],cfg['width']))
        v=base.forward(truth[None],x,weights)[0]+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,x.shape);target=base.forward(truth[None],q,weights)[0]
        for ci in order.permutation(len(cfg['configs'])):
            c=cfg['configs'][ci];point=np.zeros_like(truth)
            for n in cfg['stages']:
                before=time.perf_counter()
                if c['method']=='affine':predict,point,meta=candidate.fit(x[:n],v[:n],point,weights,c)
                elif c['method']=='input':predict,point,meta=input_model.fit(x[:n],v[:n],point,weights,c)
                elif c['method']=='prior':predict,meta=base.fit_closed(x[:n],v[:n],weights,c)
                else:predict,point,meta=base.fit_internal(x[:n],v[:n],point,weights,c)
                fit=time.perf_counter()-before;before=time.perf_counter();pred=predict(q);read=time.perf_counter()-before;error=float(np.max(np.abs(predict(x[:n])-v[:n])))
                rows.append({'method':c['name'],'seed':seed,'n_context':n,'query_mse':float(np.mean((pred-target)**2)),
                    'support_max_error':error,'support_feasible':bool(error<=base.EPS+base.TOL),'adaptation_seconds':fit,'read_queries_seconds':read,
                    'common_context_bytes':x[:n].nbytes+v[:n].nbytes,**meta})
        (args.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps({'completed':seed-cfg['seed0']+1,'total':cfg['count']}),flush=True)
    print(json.dumps({'complete':True,'rows':len(rows)}),flush=True)


if __name__=='__main__':main()
