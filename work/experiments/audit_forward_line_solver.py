import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import forward_line_solver as model
base=model.base
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();assert not args.out.exists()
base.BOUND=.2;rng=np.random.default_rng(757619);weights=base.family.make_weights(3,8)
gaps=[];representation=[];coverage=[];minimum_curvature=[];time_rows=[]
for repeat in range(6):
    old=rng.uniform(-.2,.2,(4,3,8));new=rng.uniform(-.2,.2,old.shape);x=rng.uniform(-1,1,(8,8))
    truth=rng.uniform(-.2,.2,(1,3,8));v=base.forward(truth,x,weights)[0]+rng.uniform(-base.EPS,base.EPS,x.shape)
    state=model.segments(old,new,x,weights);pieces,_=model.loss_pieces(state,v,len(x),len(old))
    before=time.perf_counter();answer,alpha,meta=model.solve(old,new,x,v,weights);time_rows.append(time.perf_counter()-before)
    minimum_curvature.append(meta['minimum_accumulated_curvature']);actual=model.values(answer,x,v,weights)
    for i,piece in enumerate(pieces):
        assert abs(piece['lower'][0])<1e-14 and abs(piece['upper'][-1]-1)<1e-14
        assert np.max(np.abs(piece['upper'][:-1]-piece['lower'][1:]))<1e-14
        def objective(t):return model.values((old[i]+t*(new[i]-old[i]))[None],x,v,weights)[0]
        independent=min(objective(0),objective(1))
        for lo,hi,coeff in zip(piece['lower'],piece['upper'],piece['coefficients']):
            midpoint=(lo+hi)/2;pred=coeff[0]*midpoint**2+2*coeff[1]*midpoint+coeff[2]
            representation.append(abs(pred-objective(midpoint)))
            numeric=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-12})
            independent=min(independent,objective(lo),objective(hi),float(numeric.fun))
        gap=float(actual[i]-independent);gaps.append(gap);coverage.append(len(piece['lower']))
        assert abs(gap)<1e-9,gap
    assert np.max(actual-np.minimum(model.values(old,x,v,weights),model.values(new,x,v,weights)))<1e-12
    assert np.max(np.abs(answer))<=.2+1e-12
assert max(representation)<1e-9
# Constant direction and an entirely flat support loss must remain well defined.
x=np.zeros((2,8));old=np.zeros((2,3,8));v=base.forward(old[:1],x,weights)[0]
answer,alpha,_=model.solve(old,old,x,v,weights);assert np.array_equal(answer,old) and np.all(alpha==0)
result=dict(passed=True,random_parameter_segments=len(gaps),objective_piece_checks=len(representation),
            maximum_absolute_global_line_gap=max(abs(g) for g in gaps),maximum_piece_representation_gap=max(representation),
            intervals_per_segment=coverage,minimum_curvature=min(minimum_curvature),batch4_seconds=time_rows,
            source_sha256={Path(s).name:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in [__file__,model.__file__,base.__file__,base.family.__file__]})
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
