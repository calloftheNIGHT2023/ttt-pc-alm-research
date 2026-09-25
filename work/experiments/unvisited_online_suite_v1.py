"""353 frozen support-only catalogue, source gate and charged invocation."""
import time
import numpy as np
import budget_reinvestment_suite_v1 as budget
import strong_pool_credit_bridge_v1 as bridge
import strong_pool_online_suite_v1 as parent
import unvisited_online_bridge_v1 as candidate
from verify_known_range_projection_v1 import project

BASE = 'results/unvisited_online'
DESIGN = 'outputs/ttt-pc-alm-research/353_unvisited_online_protocol_v1.md'
SEEDS = list(range(328000000, 328000032))
PRIMARY = 'unvisited_dual'
SOURCES = ['unvisited_online_bridge_v1.py', 'unvisited_online_suite_v1.py', 'run_unvisited_online_v1.py',
           'evaluate_unvisited_online_v1.py', 'audit_unvisited_online_v1.py', 'continue_unvisited_online_v1.ps1']
read, sha, save, complete = budget.read, budget.sha, budget.save, budget.complete


def gate(root):
    prior = root/'results/unvisited_language/audit_v1'; complete(prior)
    complete(root/'results/strong_pool_online/development_audit_v1')
    hashes = read(prior/'protocol.json')['source_sha256'].copy()
    for p, digest in hashes.items(): assert sha(root/p) == digest
    for p in [DESIGN]+['work/experiments/'+n for n in SOURCES]: hashes[p] = sha(root/p)
    return hashes


def catalogue(root):
    result = [dict(name='unvisited_'+ch, channel=ch, reference_name='strong_language_'+ch) for ch in bridge.CHANNELS]
    result += [dict(name='unvisited_native_alm64', kwargs={}, reference_name='strong_native_alm64'),
               dict(name='unvisited_native_adam240', kwargs=dict(solver='adam', steps=240), reference_name='strong_native_adam240')]
    return result


def invoke(cfg, x, v, q, seed):
    if 'channel' not in cfg: return parent.invoke(cfg, x, v, q, seed)
    begin = time.perf_counter()
    try: a, m = candidate.fit(x, v, q, seed, channel=cfg['channel'])
    except (ArithmeticError, ValueError, RuntimeError) as exc:
        import traceback
        b = np.zeros(4); pred = project(bridge.cold.base.forward(q, b))
        a = dict(prediction=pred, point_prediction=pred.copy(), selected_b=b,
                 points=b[None], allocation=np.empty(0, dtype=int), best_bank=b[None])
        m = dict(execution_failed=True, failure_type=type(exc).__name__, failure_message=str(exc),
            failure_traceback=traceback.format_exc(), query_targets_accessed=False,
            failure_policy='Charge attempt; retain fixed zero-model fallback; no retry or task removal')
    seconds = time.perf_counter()-begin
    m.update(charged_complete_seconds=seconds, query_targets_accessed=False, resources_matched=False)
    return a, m, seconds
