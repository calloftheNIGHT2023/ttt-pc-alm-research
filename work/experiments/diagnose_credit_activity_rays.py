"""Local-credit ray diagnostic with same-parent residual/random/BP controls."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import credit_activity_ray_proposals as model
from audit_bp_credit_reuse import Observer as BPObserver
from diagnose_split_activity_modes import risk
from diagnose_missing_mode_graph import distances


class Shadow:
    """Evaluator-only BP control; never returned to the local candidate."""
    def __init__(self, x, v):
        self.obs = BPObserver(x, v, model.old.old.core.batched.evaluate); self.events = {}

    def parameter(self, b, step=0): self.obs.ensure(b)

    def activity(self, b, h, u=None, step=0, phase=''):
        self.obs.ensure(b)
        self.events[step, phase] = (b.copy(), self.obs.current.copy(), self.obs.history.copy())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); a = p.parse_args()
    root = a.project / 'results/credit_activity_rays/diagnostic'; refroot = a.project / 'results/posterior_state_reuse'
    inherited = json.loads((a.project / 'results/activity_path_events/diagnostic/protocol.json').read_text()); hashes = inherited['source_sha256'].copy()
    for name, h in hashes.items(): assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == h, name
    for name in [Path(__file__).name, Path(model.__file__).name]: hashes[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    online = json.loads((a.project / 'results/batched_credit_bank/development/episodes.json').read_text())
    oldrows = {(r['seed'], r['method']): r for r in online if r['repetition'] == 0 and r['n_context'] == 4}
    configs = [dict(name='alm16', generator='alm', sweeps=16, restarts=64, credit_kind='local', reference='alm16_batch_matching'),
        dict(name='adam8', generator='adam', steps=8, restarts=64, credit_kind='bp', reference='adam8_bp_matching'),
        dict(name='adam16', generator='adam', steps=16, restarts=64, credit_kind='bp', reference='adam16_bp_matching')]
    verification = model.verify()
    # Candidate capture has a separate global-BP forbidden check; Shadow does not run here.
    rng = np.random.default_rng(816444); x = rng.uniform(0, 1, 4); v = model.base.forward(x, np.zeros(4))
    previous_jac = model.base.forward_jacobian; previous_refine = model.old.old.core.batched.refine
    def forbidden(*args, **kwargs): raise AssertionError('global BP in local ray candidate')
    try:
        model.base.forward_jacobian = forbidden; model.old.old.core.batched.refine = forbidden
        model.capture(x, v, dict(generator='alm', sweeps=2, restarts=64, credit_kind='local'))
    finally: model.base.forward_jacobian = previous_jac; model.old.old.core.batched.refine = previous_refine
    verification['local_capture_no_global_bp'] = True
    protocol = dict(seeds=list(range(5900000, 5900016)), configs=configs, source_sha256=hashes, verification=verification,
        primary_diagnostic='new positive local-credit ray modes beyond identical-parent residual, fixed random and all three shadow-BP ray channels',
        selection='unchanged 109 accepted direction; first matching parent for identical accepted direction; canonical pattern order; random rng 329221+seed',
        scope='old support diagnostic only; accepted online certificate keys/bounds unchanged; no query target; no speed claim')
    root.mkdir(parents=True, exist_ok=True); assert not (root / 'protocol.json').exists()
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8'); rows = []; audits = []
    refs = {r['seed']: r for r in json.loads((refroot / 'first_write_reference/coverage.json').read_text())}
    for seed in protocol['seeds']:
        rng = np.random.default_rng(seed); truth = rng.uniform(-.12, .12, 4); allx = rng.uniform(0, 1, 24)
        allv = model.base.forward(allx, truth) + np.random.default_rng(seed + 19000000).uniform(-model.base.EPS, model.base.EPS, 24)
        x, v = allx[:4], allv[:4]; completed = []
        for cfg in configs:
            start = time.perf_counter(); chosen, endpoint, meta = model.capture(x, v, cfg); ref = oldrows[seed, cfg['reference']]
            for field in ['matching_mode_keys', 'matching_lower_bounds']: assert meta[field] == ref[field]
            shadow = None
            if cfg['generator'] == 'alm':
                shadow = Shadow(x, v); anchor = np.zeros(4); pool = model.old.old.interface.make_pool(4, 'prior256'); starts, _ = model.old.old.interface.select_pool(x, v, anchor, pool, 64)
                with model.old.old.core.pipeline.discovery_box(.12):
                    actual = model.old.trace.refine(starts, x, v, anchor, 'alm', 16, shadow)
                    expected = model.old.trace.original(starts, x, v, anchor, 'alm', 16)
                assert np.array_equal(actual, expected)
            groups = {name: set() for name in ['selected', 'residual', 'random', 'bp_current', 'bp_history', 'bp_combined']}; counts = {name: 0 for name in groups}; labels = {}
            random = np.random.default_rng(329221 + seed); snapshots = []
            for key in sorted(chosen):
                snap = chosen[key]; b, h = snap['b'], snap['h']; labels[snap['label']] = labels.get(snap['label'], 0) + 1
                directions = dict(selected=snap['a'], residual=snap['residual'], random=random.standard_normal(h.shape))
                if shadow is not None:
                    bb, current, history = shadow.events[snap['step'], snap['phase']]; i = snap['restart']; assert np.array_equal(bb[i], b)
                    directions.update(bp_current=current[i], bp_history=history[i], bp_combined=current[i] + history[i])
                for name, direction in directions.items():
                    groups[name].update(model.ray_patterns(x, b, h, direction)); counts[name] += int(np.any(direction[:-1]))
                snapshots.append(dict(pattern=key.hex(), **{name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in snap.items()}))
            elapsed = time.perf_counter() - start; allgroups = dict(endpoint=endpoint, **groups)
            file = root / f'rays_{seed}_{cfg["name"]}.json'
            file.write_text(json.dumps(dict(seed=seed, method=cfg['name'], groups={name: sorted(k.hex() for k in members) for name, members in allgroups.items()}, snapshots=snapshots), indent=2), encoding='utf-8')
            audits.append(dict(seed=seed, method=cfg['name'], matching_keys_and_lower_bounds_exact=True, accepted_parents=len(chosen), file=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                shadow_gradient_checks=0 if shadow is None else shadow.obs.gradient_checks, shadow_maximum_gradient_error=0 if shadow is None else shadow.obs.max_gradient_error))
            completed.append((cfg, allgroups, counts, labels, elapsed, sum(sum(value.nbytes for value in s.values() if isinstance(value, np.ndarray)) for s in chosen.values())))
        # Reference applied only after all candidate and control channels have been generated.
        path = refroot / 'first_write_reference' / refs[seed]['reference_file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == refs[seed]['reference_sha256']
        full = json.loads(path.read_text()); assert full['numerical_volume_reference_complete'] and np.array_equal(full['x'], x) and np.array_equal(full['v'], v)
        keys = sorted(c['pattern'] for c in full['positive_regions']); patterns = np.array([np.frombuffer(bytes.fromhex(k), np.uint8) for k in keys]); positive = set(keys)
        adjacency = np.abs(patterns[:, None].astype(int) - patterns[None, :].astype(int)).sum(-1) == 1
        ca = json.loads((refroot / f'conditional_risk/audit_{seed}.json').read_text()); path = refroot / 'conditional_risk' / ca['curve_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ca['curve_sha256']
        with np.load(path) as curve:
            assert np.array_equal(curve['patterns'], np.array(keys)); weights, means, grid = curve['weights'], curve['region_means'], curve['q']
            for cfg, groups, counts, labels, elapsed, statebytes in completed:
                endpoint = groups['endpoint']; bp = set().union(*(groups[name] for name in ['bp_current', 'bp_history', 'bp_combined']))
                variants = dict(endpoint=endpoint, selected=endpoint | groups['selected'], residual=endpoint | groups['residual'], random=endpoint | groups['random'])
                if cfg['generator'] == 'alm': variants['bp_union'] = endpoint | bp
                scores = {}; masks = {}
                for name, members in variants.items():
                    mask = np.array([bytes.fromhex(k) in members for k in keys]); masks[name] = mask
                    regs = np.array([np.frombuffer(k, np.uint8).reshape(4, 4) for k in sorted(members)]); rejected = model.old.screen.contract(x, v, regs, 5)
                    assert not {r.tobytes().hex() for r in regs[rejected]} & positive
                    scores[name] = dict(**risk(mask, weights, means, grid), proposals=len(members), after_c5=int((~rejected).sum()))
                controls = endpoint | groups['residual'] | groups['random'] | bp; extra = (groups['selected'] - controls) & {bytes.fromhex(k) for k in keys}
                steps = distances(adjacency, masks['endpoint']); new = masks['selected'] & ~masks['endpoint']
                rows.append(dict(seed=seed, method=cfg['name'], scores=scores, nonzero_direction_counts=counts, accepted_labels=labels, independent_extra_keys=sorted(k.hex() for k in extra),
                    selected_new_positive=int(new.sum()), selected_new_beyond_h2=int((new & ((steps < 0) | (steps > 2))).sum()),
                    diagnostic_capture_and_proposal_seconds=elapsed, parent_numeric_array_bytes=statebytes,
                    proposal_key_numeric_bytes=sum(sum(map(len, group)) for group in groups.values())))
        (root / 'audits.json').write_text(json.dumps(audits, indent=2), encoding='utf-8'); (root / 'diagnostics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(dict(completed=seed - protocol['seeds'][0] + 1, total=16, paths=len(audits))), flush=True)
    summary = []
    for cfg in configs:
        rr = [r for r in rows if r['method'] == cfg['name']]
        scores = {label: {field: float(np.mean([r['scores'][label][field] for r in rr])) for field in ['mass', 'truncation_risk', 'positive_modes', 'proposals', 'after_c5']} for label in rr[0]['scores']}
        summary.append(dict(method=cfg['name'], scores=scores, selected_new_positive=sum(r['selected_new_positive'] for r in rr), independent_extra_positive=sum(len(r['independent_extra_keys']) for r in rr),
            selected_new_beyond_h2=sum(r['selected_new_beyond_h2'] for r in rr), mean_nonzero_directions={name: float(np.mean([r['nonzero_direction_counts'][name] for r in rr])) for name in rr[0]['nonzero_direction_counts']},
            mean_diagnostic_seconds=float(np.mean([r['diagnostic_capture_and_proposal_seconds'] for r in rr])), max_parent_numeric_array_bytes=max(r['parent_numeric_array_bytes'] for r in rr), max_proposal_key_numeric_bytes=max(r['proposal_key_numeric_bytes'] for r in rr)))
    result = dict(summary=summary, matching_online_certificate_replays=len(audits), source_hashes=len(hashes), scope=protocol['scope'])
    (root / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
