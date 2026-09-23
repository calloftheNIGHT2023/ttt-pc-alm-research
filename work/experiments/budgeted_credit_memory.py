"""Actual finite-call writes; local candidates and explicit BP-credit hybrids.

No saved states, reference patterns, queries or teacher enter this runtime.
Hybrid controls are explicitly global-BP users, never mislabeled BP-free.
"""
import time
import numpy as np
import batched_credit_bank_memory as old
base = old.base


class Multiplex:
    def __init__(self, collectors): self.collectors = collectors
    def parameter(self, *args, **kwargs):
        for obs in self.collectors: obs.parameter(*args, **kwargs)
    def activity(self, *args, **kwargs):
        for obs in self.collectors: obs.activity(*args, **kwargs)


def credit_keep(x, v, coarse, collector):
    accepted, _ = collector.finalize(); keys = [r.tobytes() for r in coarse]
    regs = np.array([r for r, key in zip(coarse, keys) if key not in accepted], np.uint8).reshape(-1, 4, len(x))
    start = time.perf_counter(); bank = old.direction_bank(accepted, 32); cross = {}; temporary = 0
    if len(bank) and len(regs):
        count = len(bank); repeated = np.repeat(regs, count, axis=0); aa = np.tile(bank, (len(regs), 1, 1))
        values, scales = collector.rough(repeated, aa); values = values.reshape(len(regs), count); scales = scales.reshape(values.shape)
        choices = values.argmax(1); ids = np.flatnonzero(values[np.arange(len(regs)), choices] > 1e-10 * scales[np.arange(len(regs)), choices]); flat = ids * count + choices[ids]
        pp = base.SLOPES[repeated[flat]] * aa[flat]; lower = old.screen.certified_lower_bound(x, v, repeated[flat], pp, aa[flat]) if len(ids) else []
        for index, bound in zip(ids, lower):
            if bound > 0: cross[regs[index].tobytes()] = float(bound)
        temporary = repeated.nbytes + aa.nbytes + values.nbytes + scales.nbytes + pp.nbytes
    kept = {r.tobytes() for r in regs if r.tobytes() not in cross}
    return kept, dict(matching_keys=[k.hex() for k in accepted], matching_bounds=[item[0] for item in accepted.values()],
        cross_keys=[k.hex() for k in cross], cross_bounds=list(cross.values()), directions=len(bank), direction_array_bytes=bank.nbytes,
        pending_numeric_bytes=sum(len(k) + 8 + value[1].nbytes for k, value in collector.pending.items()), cross_array_bytes_subtotal=temporary,
        collection_seconds=collector.credit_seconds, strict_matching_seconds=collector.strict_seconds, cross_seconds=time.perf_counter() - start)


def prepare(x, v, cfg):
    start = time.perf_counter(); kinds = cfg.get('credits', []); collectors = [old.Collector(x, v, kind) for kind in kinds]
    if not kinds:
        bank, regs, meta = old.previous.discover(x, v, cfg)
    else:
        anchor = np.zeros(4); pool = old.old.interface.make_pool(4, 'prior256'); starts, meta = old.old.interface.select_pool(x, v, anchor, pool, cfg.get('restarts', 64)); obs = Multiplex(collectors)
        with old.old.core.pipeline.discovery_box(.12):
            if cfg['generator'] in ['alm', 'nodual', 'pc']:
                best = old.trace.refine(starts, x, v, anchor, cfg['generator'], cfg['sweeps'], obs)
            elif cfg['generator'] == 'adam':
                evaluate = old.old.core.batched.evaluate
                def wrapped(b, *args, **kwargs):
                    obs.parameter(b); obs.activity(b, old.previous.one_pass_relaxation(b, x, v)); return evaluate(b, *args, **kwargs)
                try:
                    old.old.core.batched.evaluate = wrapped
                    best, _ = old.old.core.batched.refine(starts, x, v, anchor, solver='adam', steps=cfg['steps'], lr=.003)
                finally: old.old.core.batched.evaluate = evaluate
            else: raise ValueError(cfg['generator'])
            bank = old.old.core.contextual.deduplicate(np.vstack([starts, best]), x, v, anchor)
        first = [r.tobytes() for r in old.old.interface.archived.signatures(x, bank)]
        forward = list(dict.fromkeys(first + list(collectors[0].forward))); forward_set = set(forward)
        keys = forward + [k for k in collectors[0].split if k not in forward_set]
        regs = np.array([np.frombuffer(k, np.uint8).reshape(4, len(x)) for k in keys])
        for collector in collectors[1:]: assert set(collector.forward) == set(collectors[0].forward) and set(collector.split) == set(collectors[0].split)
    discovery = time.perf_counter() - start; begin = time.perf_counter()
    coarse = regs[~old.screen.contract(x, v, regs, 5)]; keep = {r.tobytes() for r in coarse}; screens = {}
    for kind, collector in zip(kinds, collectors):
        accepted, detail = credit_keep(x, v, coarse, collector); keep &= accepted; screens[kind] = detail
    if cfg.get('contract_rounds', 5) != 5:
        stronger = old.screen.contract(x, v, coarse, cfg['contract_rounds']); keep -= {r.tobytes() for r, reject in zip(coarse, stronger) if reject}
    retained = np.array([r for r in coarse if r.tobytes() in keep], np.uint8).reshape(-1, 4, len(x))
    return bank, retained, dict(effective_generator=cfg['generator'], effective_pool='prior256', original_pattern_count=len(regs),
        coarse_geometry_candidates=len(coarse), retained_geometry_candidates=len(retained), unbudgeted_pattern_keys=[r.tobytes().hex() for r in retained],
        discovery_seconds=discovery, screen_total_seconds=time.perf_counter() - begin, credit_details=screens,
        global_bp_credit_used='bp' in kinds, global_bp_parameter_update=cfg['generator'] == 'adam',
        metadata_scope='all online collection, screening and discovery charged; detailed array bytes are subtotals')


def fit(x, v, state, cfg):
    if state is not None or cfg.get('passthrough', False):
        predict, new, meta = old.fit(x, v, state, cfg)
        return predict, new, dict(**meta, budget_active=False, global_bp_credit_used=False, global_bp_parameter_update=meta['effective_generator'] == 'adam')
    bank, regs, meta = prepare(x, v, cfg); cap = cfg.get('geometry_budget'); evaluated = regs if cap is None else regs[:cap]
    predict, new, detail = old.materialize_retained(x, v, bank, evaluated, cfg.get('posterior_samples', 512))
    return predict, new, dict(**meta, **detail, budget_active=cap is not None, geometry_budget=cap,
        evaluated_pattern_keys=[r.tobytes().hex() for r in evaluated], skipped_for_budget=len(regs) - len(evaluated))


def verify():
    checked = 0; prefix_checks = 0
    for seed in [5900000, 5900001]:
        rng = np.random.default_rng(seed); truth = rng.uniform(-.12, .12, 4); x = rng.uniform(0, 1, 24)
        v = base.forward(x, truth) + np.random.default_rng(seed + 19000000).uniform(-base.EPS, base.EPS, 24)
        for gen, credits, rounds in [('alm', [], 5), ('alm', ['residual'], 5), ('alm', ['local'], 5), ('alm', ['bp'], 20), ('alm', ['bp', 'local'], 20), ('adam', ['bp'], 5)]:
            cfg = dict(generator=gen, sweeps=16, steps=16, restarts=64, split_proposals=True, credits=credits, contract_rounds=rounds)
            _, expected, _ = old.previous.fit(x[:4], v[:4], None, cfg)
            _, actual, meta = fit(x[:4], v[:4], None, cfg)
            assert np.array_equal(expected.anchor, actual.anchor) and np.array_equal(expected.samples, actual.samples); checked += 1
            for cap in [0, 8, 24]:
                _, limited, lm = fit(x[:4], v[:4], None, dict(**cfg, geometry_budget=cap))
                assert lm['evaluated_pattern_keys'] == meta['evaluated_pattern_keys'][:cap] and lm['geometry_calls'] <= cap
                assert set(lm['positive_mode_keys']) <= set(meta['positive_mode_keys']); prefix_checks += 1
    before_jac = base.forward_jacobian; before_bp = old.old.core.batched.refine
    def forbidden(*args, **kwargs): raise AssertionError('global BP entered pure-local budget candidate')
    try:
        base.forward_jacobian = forbidden; old.old.core.batched.refine = forbidden
        _, _, m = fit(x[:4], v[:4], None, dict(generator='alm', sweeps=16, restarts=64, credits=['local'], geometry_budget=24))
        assert not m['global_bp_credit_used'] and not m['global_bp_parameter_update']
    finally: base.forward_jacobian = before_jac; old.old.core.batched.refine = before_bp
    return dict(passed=True, full_old_state_bitwise_cases=checked, finite_budget_prefix_cases=prefix_checks, local_candidate_no_global_bp=True, hybrids_explicitly_global_bp=True)


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
