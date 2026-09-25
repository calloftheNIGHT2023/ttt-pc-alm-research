"""337 immutable trajectory, credit and readout; only exact K-best width changes."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import online_credit_resource_suite_v1 as resources
import online_credit_dyadic_v1 as fast
from exact_quadratic_events_v1 import encode

BASE = 'results/budget_reinvestment'
DESIGN = 'outputs/ttt-pc-alm-research/337_budget_reinvestment_calibration_protocol_v1.md'
SEEDS = list(range(328000000, 328000008))
KS = [8, 16, 32, 64]
REFERENCE = 'reference_fraction_first_fit_dual'
SOURCES = ['budget_reinvestment_suite_v1.py', 'test_budget_reinvestment_v1.py',
           'calibrate_budget_reinvestment_v1.py', 'audit_budget_reinvestment_v1.py',
           'continue_budget_reinvestment_v1.ps1']


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    payload = json.dumps(encode(value), indent=2, ensure_ascii=False, allow_nan=False)
    with path.open('x', encoding='utf-8') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())


def complete(folder):
    s = read(folder/'summary.json'); assert s['passed'], folder
    for n, h in s.get('outputs_sha256', {}).items(): assert sha(folder/n) == h
    return s


def gate(root):
    previous = root/'results/online_credit_fresh_pilot/dyadic_online_v1'
    s = complete(previous); assert s['calls'] == 576 and s['bitwise_array_checks'] == 11520
    p = read(previous/'protocol.json')
    hashes = {'work/experiments/'+n: h for n, h in p['frozen_source_sha256'].items()}
    hashes.update(p['source_sha256'])
    for n, h in hashes.items(): assert sha(root/n) == h, n
    hashes[DESIGN] = sha(root/DESIGN)
    for name in SOURCES: hashes['work/experiments/'+name] = sha(root/'work/experiments'/name)
    return hashes


def catalogue(root):
    old = resources.catalogue(root); online = [c for c in old if c['family'] == 'online_credit']
    configs = [c for c in old if c['family'] != 'online_credit']
    assert len(configs) == 39 and len(online) == 12
    for c in online:
        for k in KS:
            configs.append(dict(name=c['name']+f'__k{k}', family='budgeted_online_credit',
                group=c['group'], base_config=c, k=k))
    primary = next(c for c in online if c['name'] == resources.PRIMARY)
    configs.append(dict(name=REFERENCE, family='fraction_reference', group='budget_reference', base_config=primary))
    assert len(configs) == len({c['name'] for c in configs}) == 88
    return configs


@contextmanager
def width(k):
    assert k in KS
    with fast.arithmetic('integer_dp_trigger'):
        original_propose = fast.original.chain.propose
        def propose(x, v, credit, original, *, k=8):
            assert k == 8, 'Frozen caller width changed unexpectedly'
            return original_propose(x, v, credit, original, k=selected_width)
        selected_width = k
        fast.original.chain.propose = propose
        try: yield
        finally: fast.original.chain.propose = original_propose


def invoke(cfg, x, v, q, seed, loaded):
    begin = time.perf_counter()
    if cfg['family'] == 'budgeted_online_credit':
        with width(cfg['k']): a, m, _ = resources.invoke(cfg['base_config'], x, v, q, seed, loaded)
    elif cfg['family'] == 'fraction_reference':
        a, m, _ = resources.invoke(cfg['base_config'], x, v, q, seed, loaded)
    else: a, m, _ = resources.invoke(cfg, x, v, q, seed, loaded)
    seconds = time.perf_counter()-begin
    if m['execution_failed'] and any(s in m.get('failure_message', '') for s in [
        'Global BP called by local-only candidate', 'Global solver or BP entered branch selector',
        'Frozen caller width changed unexpectedly']):
        raise RuntimeError('Forbidden credit or solver access cannot be a numerical fallback')
    m.update(charged_complete_seconds=seconds, reinvestment_k=cfg.get('k'),
        reinvestment_arithmetic='integer_dp_trigger' if cfg['family'] == 'budgeted_online_credit' else 'frozen_original',
        query_targets_accessed=False, resources_matched=False)
    return a, m, seconds


def observed(root, seeds):
    folder = root/'results/online_credit_fresh_pilot/pilot_predictions_v2'
    s = complete(folder); assert s['tasks'] == 512
    rows = read(folder/'rows.json'); index = {(r['seed'], r['method']): r for r in rows}
    result = {}; manifests = {}
    for seed in seeds:
        row = index[seed, resources.PRIMARY]; p = folder/row['file']
        assert sha(p) == row['sha256']
        with np.load(p, allow_pickle=False) as z:
            result[seed] = tuple(z[n].copy() for n in ['x_observed', 'v_observed', 'q_observed'])
        manifests[str(seed)] = dict(file=str(p.relative_to(root)), sha256=row['sha256'])
    return result, manifests, index


def array_checks(a, b):
    assert set(a) == set(b)
    for key, value in a.items():
        assert value.shape == b[key].shape and value.dtype == b[key].dtype, key
        assert value.tobytes() == b[key].tobytes(), key
    return len(a)


def frozen_check(root, cfg, seed, a, m, index):
    if cfg['family'] == 'budgeted_online_credit' and cfg['k'] != 8: return 0
    name = cfg['base_config']['name'] if 'base_config' in cfg else cfg['name']
    row = index[seed, name]; folder = root/'results/online_credit_fresh_pilot/pilot_predictions_v2'
    assert sha(folder/row['file']) == row['sha256'] and sha(folder/row['metadata_file']) == row['metadata_sha256']
    expected_meta = read(folder/row['metadata_file'])['metadata']
    assert m['execution_failed'] == expected_meta['execution_failed']
    with np.load(folder/row['file'], allow_pickle=False) as z:
        b = {n: z[n] for n in z.files if n not in ['x_observed', 'v_observed', 'q_observed']}
    count = array_checks(a, b)
    if 'positive_modes' in m: assert m['positive_modes'] == expected_meta['positive_modes']
    return count


def select(configs, means, failures):
    cap = means[REFERENCE]
    base_names = sorted({c['base_config']['name'] for c in configs if c['family'] == 'budgeted_online_credit'})
    allowed = {name: [k for k in KS if means[name+f'__k{k}'] <= cap and failures[name+f'__k{k}'] == 0]
               for name in base_names}
    common = [k for k in KS if all(k in allowed[name] for name in base_names)]
    assert common, 'No common K satisfies the fixed resource rule; do not expand budget'
    k = max(common)
    own = {name: max(values) for name, values in allowed.items()}
    memory_names = {REFERENCE}
    memory_names.update(c['name'] for c in configs if c['family'] not in ['budgeted_online_credit', 'fraction_reference'])
    memory_names.update(name+f'__k{k}' for name in base_names)
    memory_names.update(name+f'__k{own[name]}' for name in base_names)
    return dict(budget_reference=REFERENCE, budget_seconds=cap, common_k=k, admissible_common_k=common,
        per_credit_largest_k=own, per_credit_admissible_k=allowed,
        memory_methods=sorted(memory_names), within_reference_budget=[c['name'] for c in configs if means[c['name']] <= cap],
        sensitivity_110_percent=[c['name'] for c in configs if means[c['name']] <= 1.1*cap],
        highest_adam_still_within=means['probe_then_adam3840_33'] <= cap,
        highest_adam_within_110_percent=means['probe_then_adam3840_33'] <= 1.1*cap,
        query_quality_used=False, all_methods_retained=True, selection_scope='time and execution validity; not equal peak memory')
