"""260 all36 fresh pipelines; old grid also gets its verified short scan."""
import time
import numpy as np
import live_minimum_dual as live
import live_dual_jump_short as oldwarm
import dual_jump_resource_suite as oldsuite
from run_minimum_dual_continuation import CONFIGS
oldfit=oldsuite.oldfit

def configs():
    ans=[dict(name=c['name'],family='minimum',config={**c,'initial':'activity_only' if c['initial']=='minimum_activity' else 'dual_jump'}) for c in CONFIGS]+oldsuite.configs()
    assert len(ans)==36 and len({c['name'] for c in ans})==36
    return ans

def fit(cfg,x,v,q,seed,loaded):
    start=time.perf_counter()
    if cfg['family']=='minimum':arrays,meta=live.fit(cfg['config'],x,v,q,seed)
    elif cfg['family']=='new':arrays,meta=oldwarm.fit(cfg['config'],x,v,q,seed)
    else:arrays,meta=oldsuite.fit(cfg,x,v,q,seed,loaded)
    meta['complete_call_seconds']=time.perf_counter()-start
    return arrays,meta

def check(root,seed,cfg,arrays,meta):
    if cfg['family']!='minimum':return oldsuite.check(root,seed,cfg,arrays,meta)
    base=root/'results/minimum_dual_query/development'
    with np.load(base/f'point_{seed}_{cfg["name"]}.npz') as a:
        assert arrays['selected_b'].tobytes()==a['b'].tobytes();assert arrays['point_prediction'].tobytes()==a['prediction'].tobytes()
    with np.load(base/f'mode_{seed}_{cfg["name"]}_0.npz') as a:
        for k in ['points','allocation','prediction']:assert arrays[k].tobytes()==a[k].tobytes(),(seed,cfg['name'],k)
    with np.load(root/'results/minimum_sufficient_dual/continuation'/f'{seed}_{cfg["name"]}.npz') as a:
        for k in ['best_bank','initial_b','initial_h','initial_u']:assert arrays[k].tobytes()==a[k].tobytes(),(seed,cfg['name'],k)
