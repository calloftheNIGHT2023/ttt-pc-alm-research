"""Fresh complete pipeline for13 candidates/controls,10 old solvers,10 heads."""
import time
import numpy as np
import live_dual_jump_query as live
import live_shared_mode_readout as oldlive
import run_cold_stagnation_switch as oldfit
from run_local_dual_jump_continuation import CONFIGS

def configs():
    ans=[dict(name=c['name'],family='new',config=c) for c in CONFIGS]
    ans.extend(dict(name='cold__'+c['name'],family='cold' if c['family'] in ['local','bp'] else 'head',config=c) for c in live.cold.configs())
    assert len(ans)==33 and len({c['name'] for c in ans})==33
    return ans

def fit(cfg,x,v,q,seed,loaded):
    start=time.perf_counter()
    if cfg['family']=='new':arrays,meta=live.fit(cfg['config'],x,v,q,seed)
    elif cfg['family']=='cold':
        arrays,meta=oldlive.fit(cfg['config'],x,v,q,seed);arrays['point_prediction']=live.cold.base.forward(q,arrays['selected_b'])
    else:
        arrays,meta=oldfit.fit(cfg['config'],x,v,q,loaded,False);arrays['point_prediction']=arrays['prediction'].copy()
    meta['complete_call_seconds']=time.perf_counter()-start
    return arrays,meta

def check(root,seed,cfg,arrays,meta):
    if cfg['family']=='new':
        base=root/'results/dual_jump_query/development'
        with np.load(base/f'point_{seed}_{cfg["name"]}.npz') as a:
            assert arrays['selected_b'].tobytes()==a['b'].tobytes();assert arrays['point_prediction'].tobytes()==a['prediction'].tobytes()
        with np.load(base/f'mode_{seed}_{cfg["name"]}_0.npz') as a:
            for key in ['points','allocation','prediction']:assert arrays[key].tobytes()==a[key].tobytes(),(seed,cfg['name'],key)
        with np.load(root/'results/local_dual_jump/continuation'/f'{seed}_{cfg["name"]}.npz') as a:assert arrays['best_bank'].tobytes()==a['best_bank'].tobytes()
    else:
        method=cfg['config']['name']
        with np.load(root/'results/cold_stagnation_switch/development'/f'{seed}_{method}.npz') as a:assert arrays['point_prediction'].tobytes()==a['prediction'].tobytes()
        if cfg['family']=='cold':
            with np.load(root/'results/shared_mode_readout/development'/f'state_{seed}_{method}_0.npz') as a:
                for key in ['points','allocation','prediction']:assert arrays[key].tobytes()==a[key].tobytes(),(seed,method,key)

