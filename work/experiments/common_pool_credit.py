"""Same candidate universe, same solver, unchanged method-specific credits."""
import time
import numpy as np
from online_endpoint_h2 import EndpointIntervalBank
from verify_credit_moment_step import guarded_solve

METHODS=('alm_native','alm_residual','adam60_native','adam60_residual','pc_native','nodual_native')


def assemble(items):
    assert set(items)==set(METHODS)
    first=items[METHODS[0]];x=first['x'];v=first['v'];shape=first['regs'].shape[1:]
    keys=set();original={}
    for method in METHODS:
        item=items[method];assert item['x'].tobytes()==x.tobytes() and item['v'].tobytes()==v.tobytes()
        assert item['regs'].dtype==np.uint8 and item['regs'].shape[1:]==shape
        original[method]={r.tobytes() for r in item['regs']};keys.update(original[method])
    keys=sorted(keys);regs=np.array([np.frombuffer(k,np.uint8).reshape(shape) for k in keys],dtype=np.uint8)
    membership=np.array([[k in original[m] for m in METHODS] for k in keys],dtype=bool)
    assert membership.any(1).all() and len(keys)==len(set(keys))
    assert all(int(membership[:,i].sum())==len(original[m]) for i,m in enumerate(METHODS))
    return x,v,regs,membership


def solve(x,v,regs,bank):
    start=time.perf_counter();screen=EndpointIntervalBank(x,v,bank);construction=time.perf_counter()-start
    start=time.perf_counter();mask,proofs,_=screen.screen(regs);screen_seconds=time.perf_counter()-start
    indices=np.flatnonzero(~mask);arrays,meta=guarded_solve(x,v,regs[indices],bank)
    accepted=mask.copy();accepted[indices[arrays['first_step']>0]]=True
    arrays.update(indices=indices,old_positive=mask,accepted=accepted)
    meta.update(old_proofs=proofs,old_count=int(mask.sum()),total_positive=int(accepted.sum()),
        construction_seconds=construction,old_screen_seconds=screen_seconds,
        charged_component_seconds=construction+screen_seconds+meta['total_seconds'],bank_bytes=bank.nbytes,
        scope='Precomputed common pool and credits excluded; all screening and new joint solve charged; not online time')
    return arrays,meta
