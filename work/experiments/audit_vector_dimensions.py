"""Dimension-specific derivative/orthogonality audit before width experiments."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import vector_interval_memory as task

p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); args=p.parse_args()
checks=[]; rng=np.random.default_rng(440388)
for width in [8,32]:
    weights=task.family.make_weights(3,width); x=rng.uniform(-1,1,(8,width)); bank=rng.uniform(-.2,.2,(2,3,width)); v=rng.uniform(-1,1,(8,width))
    _,grad,_=task.evaluate(bank,x,v,weights); _,error,jac,_=task.evaluate(bank,x,v,weights,True)
    fromjac=np.einsum("rnwp,rnw->rp",jac,error,optimize=True).reshape(bank.shape)/len(x)
    delta=float(np.max(np.abs(grad-fromjac))); assert delta<1e-10
    fd=[]
    for coord in np.linspace(0,3*width-1,12,dtype=int):
        e=np.eye(3*width)[coord].reshape(3,width)*1e-7
        plus=task.evaluate((bank[0]+e)[None],x,v,weights)[0][0]; minus=task.evaluate((bank[0]-e)[None],x,v,weights)[0][0]
        fd.append(abs((plus-minus)/2e-7-grad[0].ravel()[coord]))
    assert max(fd)<1e-6
    checks.append({"width":width,"fast_biases":3*width,"chain_gradient_error":delta,"finite_difference_max_error":float(max(fd)),
        "orthogonality_error":float(np.max(np.abs(weights@weights.transpose(0,2,1)-np.eye(width))))})
result={"passed":True,"checks":checks,"source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in [Path(__file__),Path(task.__file__),Path(task.family.__file__)]}}
args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result))
