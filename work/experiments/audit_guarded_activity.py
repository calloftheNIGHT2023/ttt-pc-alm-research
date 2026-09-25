import argparse,hashlib,json
from pathlib import Path
import numpy as np
import guarded_activity_memory as model
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();assert not args.out.exists()
rng=np.random.default_rng(739619);weight=model.base.family.make_weights(1,8)[0];records=[]
for gradient in [False,True]:
    for _ in range(20):
        low=rng.uniform(-1,0,(4,11,8));high=rng.uniform(0,1,(4,11,8));old=rng.uniform(low,high)
        center=rng.normal(size=old.shape)*2;target=rng.normal(size=old.shape)*2;bias=rng.uniform(-.2,.2,(4,1,8))
        result,meta=model.update(old,center,target,weight,bias,low,high,gradient=gradient)
        assert np.all(result>=low-1e-12) and np.all(result<=high+1e-12)
        assert meta['max_old_energy_increase']<=1e-10 and meta['max_clipped_energy_increase']<=1e-10
        records.append(meta)
model.base.BOUND=.2;weights=model.base.family.make_weights(3,8);truth=rng.uniform(-.2,.2,(3,8));x=rng.uniform(-1,1,(8,8));v=model.base.forward(truth[None],x,weights)[0]
old_evaluate=model.base.evaluate
def forbidden(*args,**kwargs):raise AssertionError('full-chain gradient called')
model.base.evaluate=forbidden
try:model.fit(x,v,np.zeros_like(truth),weights,dict(sweeps=8))
finally:model.base.evaluate=old_evaluate
result=dict(passed=True,sample_blocks=sum(r['block_samples'] for r in records),
            maximum_old_energy_increase=max(r['max_old_energy_increase'] for r in records),
            maximum_clipped_energy_increase=max(r['max_clipped_energy_increase'] for r in records),no_full_chain_gradient=True,
            source_sha256={Path(s).name:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in [__file__,model.__file__,model.reference.__file__,model.reference.activity.__file__]})
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
