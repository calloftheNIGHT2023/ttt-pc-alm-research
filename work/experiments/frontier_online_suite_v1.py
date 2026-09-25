"""363 fixed main, full controls, source gate and failure policy."""
import time
import numpy as np
import cross_region_online_suite_v1 as previous
import frontier_online_bridge_v1 as candidate
from test_frontier_reallocation_v1 import hashes as component_hashes
from verify_known_range_projection_v1 import project

BASE='results/frontier_online';SEEDS=list(range(328000000,328000032));PRIMARY='frontier_dual_frontier_g8'
DESIGN='outputs/ttt-pc-alm-research/363_frontier_online_protocol_v1.md'
SOURCES=['frontier_online_bridge_v1.py','frontier_online_suite_v1.py','run_frontier_online_v1.py',
         'evaluate_frontier_online_v1.py','audit_frontier_online_v1.py']
budget=previous.budget;read,sha,save,complete=previous.read,previous.sha,previous.save,previous.complete


def gate(root):
    complete(root/'results/frontier_reallocation/audit_v1');complete(root/'results/cross_region_online/development_audit_v1')
    h=previous.gate(root);h.update(component_hashes(root))
    h.update({p:sha(root/p) for p in [DESIGN]+['work/experiments/'+n for n in SOURCES]})
    return h


def catalogue(root):
    result=[dict(name='frontier_'+c+'_'+s+'_g8',channel=c,strategy=s,reference_name='strong_language_'+c)
            for c in previous.bridge.CHANNELS for s in ['frontier','uniform']]
    result += [dict(name='frontier_c20_'+('all' if g is None else 'g'+str(g)),channel='zero',
                    settings=dict(screen=False,geometry_budget=g),reference_name='cross_c20_'+('all' if g is None else 'g'+str(g))) for g in [8,16,None]]
    result += [dict(name='frontier_native_alm64',kwargs={},reference_name='cross_native_alm64'),
               dict(name='frontier_native_adam240',kwargs=dict(solver='adam',steps=240),reference_name='cross_native_adam240')]
    assert len(result)==17;return result


def invoke(cfg,x,v,q,seed):
    if 'strategy' not in cfg:return previous.invoke(cfg,x,v,q,seed)
    begin=time.perf_counter()
    try:a,m=candidate.fit(x,v,q,seed,channel=cfg['channel'],strategy=cfg['strategy'])
    except (ArithmeticError,ValueError,RuntimeError) as exc:
        import traceback
        b=np.zeros(4);prediction=project(previous.bridge.cold.base.forward(q,b))
        a=dict(prediction=prediction,point_prediction=prediction.copy(),selected_b=b,points=b[None],allocation=np.empty(0,int),best_bank=b[None])
        m=dict(execution_failed=True,failure_type=type(exc).__name__,failure_message=str(exc),failure_traceback=traceback.format_exc(),
               query_targets_accessed=False,failure_policy='Fixed zero fallback; all cost kept; no retry or task removal')
    seconds=time.perf_counter()-begin;m.update(charged_complete_seconds=seconds,query_targets_accessed=False,resources_matched=False)
    return a,m,seconds
