"""345 complete online calls with only the branch selector substituted."""
from contextlib import contextmanager
import time
import budget_reinvestment_suite_v1 as budget
import support_language_chain_v1 as language

BASE = 'results/support_language_online'
DESIGN = 'outputs/ttt-pc-alm-research/345_support_language_online_protocol_v1.md'
SEEDS = list(range(328000000, 328000032))
CHANNELS = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']
PRIMARY = 'language_first_fit_dual'
SOURCES = ['support_language_online_suite_v1.py', 'run_support_language_online_v1.py',
           'evaluate_support_language_online_v1.py', 'audit_support_language_online_v1.py',
           'continue_support_language_online_v1.ps1']
read, sha, save, complete = budget.read, budget.sha, budget.save, budget.complete


def gate(root):
    tested = root/'results/support_language_primitive/preflight_v1'; complete(tested)
    old = root/'results/search_radius_development'
    complete(old/'development_predictions_v1'); complete(old/'development_evaluation_v1'); complete(old/'development_audit_v1')
    hashes = read(tested/'protocol.json')['source_sha256'].copy()
    for path, digest in hashes.items(): assert sha(root/path) == digest
    for path in [DESIGN]+['work/experiments/'+name for name in SOURCES]: hashes[path] = sha(root/path)
    return hashes


def catalogue(root):
    originals = {c['name']: c for c in budget.resources.catalogue(root)}
    return [dict(name='language_first_fit_'+ch, channel=ch,
                 base_config=originals['online_first_fit_'+ch]) for ch in CHANNELS]


@contextmanager
def selector():
    with budget.fast.arithmetic('integer_dp_trigger'):
        saved = budget.fast.original.chain.propose
        budget.fast.original.chain.propose = language.propose
        try: yield
        finally: budget.fast.original.chain.propose = saved


def invoke(cfg, x, v, q, seed):
    begin = time.perf_counter()
    with selector():
        a, m, _ = budget.resources.invoke(cfg['base_config'], x, v, q, seed, None)
    seconds = time.perf_counter()-begin
    if m['execution_failed'] and any(s in m.get('failure_message', '') for s in [
        'Global BP called by local-only candidate', 'Global solver or BP entered branch selector']):
        raise RuntimeError('Forbidden credit or solver use is not a numerical fallback')
    m.update(charged_complete_seconds=seconds, support_language_constrained=True,
             query_targets_accessed=False, resources_matched=False)
    return a, m, seconds
