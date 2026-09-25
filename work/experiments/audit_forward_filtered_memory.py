import argparse,hashlib,json
from pathlib import Path
import numpy as np
import forward_filtered_memory as model
base=model.base
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();assert not args.out.exists()
base.BOUND=.2;rng=np.random.default_rng(5400000);truth=rng.uniform(-.2,.2,(3,8));x=rng.uniform(-1,1,(24,8))[:8]
weights=base.family.make_weights(3,8);v=base.forward(truth[None],x,weights)[0]+np.random.default_rng(24400000).uniform(-base.EPS,base.EPS,x.shape)
anchor=np.zeros_like(truth);starts,_=base.proposals(x,v,anchor,weights);records=[]
for name in ['orthogonal','guard','adam']:
    if name=='adam':observed,_=model.adam(starts,x,v,weights,anchor,steps=16,mode='none');expected,_=base.bp(starts,x,v,weights,anchor,solver='adam',steps=16,lr=.01)
    else:
        observed,_=model.local(starts,x,v,weights,anchor,sweeps=16,mode='none',guard=name=='guard')
        expected,_=(model.guarded.local if name=='guard' else model.original.local)(starts,x,v,weights,anchor,sweeps=16)
    assert np.array_equal(observed,expected),name
    for mode in ['grid','exact']:
        history=[];old_evaluate=base.evaluate
        def forbidden(*args,**kwargs):raise AssertionError('BP credit entered local proposal/filter')
        if name!='adam':base.evaluate=forbidden
        try:
            if name=='adam':_,meta=model.adam(starts,x,v,weights,anchor,steps=16,mode=mode,history=history)
            else:_,meta=model.local(starts,x,v,weights,anchor,sweeps=16,mode=mode,guard=name=='guard',history=history)
        finally:base.evaluate=old_evaluate
        increase=max(max(np.array(t['after'])-t['before']) for t in history);assert increase<=1e-10,(name,mode,increase)
        records.append(dict(method=name,mode=mode,parameter_equivalence_without_filter=True,maximum_true_write_loss_increase=float(increase),**meta))
result=dict(passed=True,records=records,scope='support-only correctness; instrumentation timing not a performance comparison',
            source_sha256={Path(s).name:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in [__file__,model.__file__,model.line.__file__,base.__file__,base.family.__file__]})
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
