"""358 fixed online configurations and hashes; no query answer access."""
import time
import numpy as np
import budget_reinvestment_suite_v1 as budget
import unvisited_online_suite_v1 as previous
import strong_pool_online_suite_v1 as native
import strong_pool_credit_bridge_v1 as bridge
import cross_region_online_bridge_v1 as candidate
from test_cross_region_credit_v1 import hashes as component_hashes
from verify_known_range_projection_v1 import project

BASE='results/cross_region_online';SEEDS=list(range(328000000,328000032));PRIMARY='cross_dual_reuse_g8'
DESIGN='outputs/ttt-pc-alm-research/358_cross_region_online_protocol_v1.md'
SOURCES=['cross_region_online_bridge_v1.py','cross_region_online_suite_v1.py','run_cross_region_online_v1.py',
         'evaluate_cross_region_online_v1.py','audit_cross_region_online_v1.py']
read,sha,save,complete=budget.read,budget.sha,budget.save,budget.complete


def gate(root):
    complete(root/'results/cross_region_credit/audit_v1')
    complete(root/'results/unvisited_online/development_audit_v1')
    h=previous.gate(root);h.update(component_hashes(root))
    for p in [DESIGN]+['work/experiments/'+n for n in SOURCES]:h[p]=sha(root/p)
    for p,digest in h.items():assert sha(root/p)==digest
    return h


def catalogue(root):
    result=[dict(name='cross_'+c+'_reuse_g8',channel=c,settings=dict(propagate=True),reference_name='strong_language_'+c) for c in bridge.CHANNELS]
    result += [dict(name='cross_'+c+'_independent_g8',channel=c,settings=dict(propagate=False),reference_name='strong_language_'+c) for c in ['dual','zero']]
    result.append(dict(name='cross_zero_independent256_g8',channel='zero',settings=dict(propagate=False,steps=256),reference_name='strong_language_zero'))
    result += [dict(name='cross_c20_'+('all' if g is None else 'g'+str(g)),channel='zero',
                    settings=dict(screen=False,geometry_budget=g),reference_name='strong_language_zero') for g in [8,16,None]]
    result += [dict(name='cross_native_alm64',kwargs={},reference_name='strong_native_alm64'),
               dict(name='cross_native_adam240',kwargs=dict(solver='adam',steps=240),reference_name='strong_native_adam240')]
    assert len(result)==14;return result


def invoke(cfg,x,v,q,seed):
    if 'channel' not in cfg:return native.invoke(cfg,x,v,q,seed)
    begin=time.perf_counter()
    try:a,m=candidate.fit(x,v,q,seed,channel=cfg['channel'],**cfg['settings'])
    except (ArithmeticError,ValueError,RuntimeError) as exc:
        import traceback
        b=np.zeros(4);prediction=project(bridge.cold.base.forward(q,b))
        a=dict(prediction=prediction,point_prediction=prediction.copy(),selected_b=b,points=b[None],allocation=np.empty(0,int),best_bank=b[None])
        m=dict(execution_failed=True,failure_type=type(exc).__name__,failure_message=str(exc),failure_traceback=traceback.format_exc(),
               query_targets_accessed=False,failure_policy='Fixed zero fallback; all cost kept; no retry or task removal')
    seconds=time.perf_counter()-begin;m.update(charged_complete_seconds=seconds,query_targets_accessed=False,resources_matched=False)
    return a,m,seconds
