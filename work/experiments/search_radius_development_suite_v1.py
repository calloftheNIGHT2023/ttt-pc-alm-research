"""340 fixed development panel; wider radius, resource-width controls and stronger solvers."""
from contextlib import contextmanager
import time
import budget_reinvestment_suite_v1 as budget
import branch_image_chain_radius_v1 as radius

BASE = 'results/search_radius_development'
DESIGN = 'outputs/ttt-pc-alm-research/340_radius_development_protocol_v1.md'
SEEDS = list(range(328000000, 328000032))
PRIMARY = 'radius_first_fit_dual__k8__sall'
SOURCES = ['search_radius_development_suite_v1.py', 'run_search_radius_development_v1.py',
           'evaluate_search_radius_development_v1.py', 'audit_search_radius_development_v1.py',
           'continue_search_radius_development_v1.ps1']
read, sha, save, complete = budget.read, budget.sha, budget.save, budget.complete


def gate(root):
    cal = root/budget.BASE/'calibration_v1'; complete(cal)
    audit = complete(root/budget.BASE/'audit_v1')
    assert audit['calibration_summary_sha256'] == sha(cal/'summary.json')
    hashes = budget.gate(root); assert hashes == read(cal/'protocol.json')['source_sha256']
    reach = root/'results/search_radius_reachability/audit_v1'; complete(reach)
    union = next(m for m in read(reach/'methods.json') if m['method'] == 'strong_union')
    assert union['certified_categories'].get('above_window', 0) > 0
    tested = root/'results/search_radius_primitive/preflight_v1'; complete(tested)
    tp = read(tested/'protocol.json')
    assert tp['calibration_summary_sha256'] == sha(cal/'summary.json')
    assert tp['reachability_summary_sha256'] == sha(reach/'summary.json')
    for path, digest in tp['source_sha256'].items(): assert sha(root/path) == digest
    hashes.update(tp['source_sha256'])
    for path in [DESIGN]+['work/experiments/'+n for n in SOURCES]: hashes[path] = sha(root/path)
    return hashes


def catalogue(root):
    originals = budget.resources.catalogue(root)
    online = [c for c in originals if c['family'] == 'online_credit']
    result = []
    for c in online:
        if c['policy'] == 'first_fit':
            for k in [1, 2, 4, 8]:
                result.append(dict(name=f'radius_first_fit_{c["channel"]}__k{k}__sall',
                    family='radius_online', group='radius_first_fit', base_config=c, k=k, span='all'))
    selection = read(root/budget.BASE/'calibration_v1/selection.json')
    for c in online:
        for k in sorted({selection['common_k'], selection['per_credit_largest_k'][c['name']]}):
            result.append(dict(name=c['name']+f'__k{k}', family='budgeted_online_credit',
                group=c['group'], base_config=c, k=k))
    for steps in [7680, 15360]:
        name = f'probe_then_adam{steps}_33'
        result.append(dict(name=name, family='probe', group='extended_adam',
            config=dict(name=name, atomic='branch_probe', solver='adam', certificate=False, action='Q', steps=steps)))
    for solver in ['pc', 'nodual', 'alm']:
        for steps in [512, 1024]:
            plain = solver == 'alm'; name = f'plain_alm{steps}_33' if plain else f'probe_{solver}{steps}_33'
            result.append(dict(name=name, family='local', group='extended_'+solver,
                config=dict(name=name, atomic='alm1' if plain else 'branch_probe', solver=solver,
                            certificate=False, action='L' if plain else 'Q', prefix=steps, extra=steps)))
    for steps in [128, 256, 512]:
        name = f'probe_alm{steps}_33'
        result.append(dict(name=name, family='local', group='extended_probe_alm',
            config=dict(name=name, atomic='branch_probe', solver='alm', certificate=False,
                        action='Q', prefix=steps, extra=steps)))
    names = [c['name'] for c in originals+result]
    assert len(names) == len(set(names)) and PRIMARY in names
    return result


@contextmanager
def extraction(width, span):
    with budget.fast.arithmetic('integer_dp_trigger'):
        saved = budget.fast.original.chain.propose
        def propose(x, v, credit, original, *, k=8):
            assert k == 8, 'Frozen caller width changed unexpectedly'
            return radius.propose(x, v, credit, original, k=width, span=span)
        budget.fast.original.chain.propose = propose
        try: yield
        finally: budget.fast.original.chain.propose = saved


def invoke(cfg, x, v, q, seed, loaded):
    begin = time.perf_counter()
    if cfg['family'] == 'radius_online':
        with extraction(cfg['k'], cfg['span']):
            a, m, _ = budget.resources.invoke(cfg['base_config'], x, v, q, seed, loaded)
    else:
        a, m, _ = budget.invoke(cfg, x, v, q, seed, loaded)
    seconds = time.perf_counter()-begin
    if m['execution_failed'] and any(s in m.get('failure_message', '') for s in [
        'Global BP called by local-only candidate', 'Global solver or BP entered branch selector',
        'Frozen caller width changed unexpectedly']):
        raise RuntimeError('Forbidden credit or solver use is not a numerical fallback')
    m.update(charged_complete_seconds=seconds, development_extraction_span=cfg.get('span', 3),
             development_k=cfg.get('k'), query_targets_accessed=False, resources_matched=False)
    return a, m, seconds
