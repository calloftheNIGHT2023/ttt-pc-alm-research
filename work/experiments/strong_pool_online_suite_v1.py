"""348 fixed strong-pool development methods; no evaluator or teacher access."""
import copy
import time
import numpy as np
import budget_reinvestment_suite_v1 as budget
import strong_pool_credit_bridge_v1 as bridge
import cold_stagnation_switch as cold
from verify_known_range_projection_v1 import project

BASE = 'results/strong_pool_online'
DESIGN = 'outputs/ttt-pc-alm-research/348_strong_pool_online_protocol_v1.md'
SEEDS = list(range(328000000, 328000032))
PRIMARY = 'strong_language_dual'
SOURCES = ['strong_pool_online_suite_v1.py', 'run_strong_pool_online_v1.py',
           'evaluate_strong_pool_online_v1.py', 'audit_strong_pool_online_v1.py', 'continue_strong_pool_online_v1.ps1']
read, sha, save, complete = budget.read, budget.sha, budget.save, budget.complete


def gate(root):
    tested = root/'results/strong_pool_credit_bridge/preflight_v1'; complete(tested)
    previous = root/'results/support_language_online'
    complete(previous/'development_predictions_v1'); complete(previous/'development_evaluation_v1'); complete(previous/'development_audit_v1')
    hashes = read(tested/'protocol.json')['source_sha256'].copy()
    for path, digest in hashes.items(): assert sha(root/path) == digest
    for path in [DESIGN]+['work/experiments/'+n for n in SOURCES]: hashes[path] = sha(root/path)
    return hashes


def catalogue(root):
    result = [dict(name='strong_native_alm64', kwargs={}, reference_347='native'),
              dict(name='strong_watch_alm64', kwargs=dict(watch=True), reference_347='watch')]
    result += [dict(name='strong_language_'+ch, kwargs=dict(channel=ch), reference_347=ch) for ch in bridge.CHANNELS]
    for solver, steps in [('alm', [128, 256]), ('pc', [64, 128, 256]), ('nodual', [64, 128, 256]), ('adam', [240, 960, 3840])]:
        result += [dict(name=f'strong_native_{solver}{s}', kwargs=dict(solver=solver, steps=s)) for s in steps]
    assert len(result) == len({c['name'] for c in result}) == 19
    return result


def invoke(cfg, x, v, q, seed):
    begin = time.perf_counter()
    try:
        a, m = bridge.fit(x, v, q, seed, trace_hash=True, **cfg['kwargs'])
    except (ArithmeticError, ValueError, RuntimeError) as exc:
        # Assertions and programming errors stay fatal; no silent structural fallback.
        import traceback
        b = np.zeros(4); pred = project(cold.base.forward(q, b))
        a = dict(prediction=pred, point_prediction=pred.copy(), selected_b=b,
                 points=b[None], allocation=np.empty(0, dtype=int), best_bank=b[None])
        m = dict(execution_failed=True, failure_type=type(exc).__name__, failure_message=str(exc),
                 failure_traceback=traceback.format_exc(), query_targets_accessed=False,
                 failure_policy='Charge failed attempt, return original zero model, no retry and no removed task')
    seconds = time.perf_counter()-begin
    m.update(charged_complete_seconds=seconds, query_targets_accessed=False, resources_matched=False)
    return a, m, seconds


def old_reference(root, cfg, x, v, q, seed):
    """Actual frozen legacy implementation, for preflight equivalence only."""
    base = copy.deepcopy(next(c for c in budget.resources.catalogue(root) if c['name'] == 'probe_all_alm64'))
    settings = base['config']['config']; kwargs = cfg['kwargs']; solver = kwargs.get('solver', 'alm')
    settings.update(method=solver, steps=kwargs.get('steps', 64), bp=solver == 'adam')
    old_step = cold.Local.step; fingerprints = []
    def observe(state):
        if state.method == solver and len(state.b) == 629:
            if not fingerprints: fingerprints.append(bridge.state_digest(state))
            result = old_step(state); fingerprints.append(bridge.state_digest(state)); return result
        return old_step(state)
    try:
        cold.Local.step = observe
        arrays, meta, _ = budget.resources.invoke(base, x, v, q, seed, None)
    finally: cold.Local.step = old_step
    return arrays, meta, fingerprints
