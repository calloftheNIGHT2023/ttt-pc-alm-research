"""Fixed-order safe-screen prefix audit. No learner reads complete references."""
import argparse, hashlib, json
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import batched_credit_bank_memory as model
from certified_branch_solver import exact_certificate
from diagnose_split_activity_modes import risk


def filter_collector(x, v, regs, collector):
    accepted, proofs = collector.finalize(); keys = [r.tobytes() for r in regs]
    kept = [r for r, key in zip(regs, keys) if key not in accepted]
    rr = np.array(kept, np.uint8).reshape(-1, 4, len(x)); rr = rr[~model.screen.contract(x, v, rr, 5)]
    bank = model.direction_bank(accepted, 32); cross = {}
    if len(bank) and len(rr):
        k = len(bank); repeated = np.repeat(rr, k, axis=0); aa = np.tile(bank, (len(rr), 1, 1))
        value, scale = collector.rough(repeated, aa); value = value.reshape(len(rr), k); scale = scale.reshape(value.shape)
        choice = value.argmax(1); selected = value[np.arange(len(rr)), choice]; ss = scale[np.arange(len(rr)), choice]
        ids = np.flatnonzero(selected > 1e-10 * ss); flat = ids * k + choice[ids]; pp = model.base.SLOPES[repeated[flat]] * aa[flat]
        lower = model.screen.certified_lower_bound(x, v, repeated[flat], pp, aa[flat]) if len(ids) else []
        for index, fi, p, lb in zip(ids, flat, pp, lower):
            if lb <= 0: continue
            key = rr[index].tobytes(); cross[key] = float(lb)
            proofs.append(dict(pattern=key.hex(), a=aa[fi].copy(), p=p.copy(), lower=float(lb), stage='cross'))
    return [r.tobytes() for r in rr if r.tobytes() not in cross], proofs


def check_prefix(base, filtered, positive):
    assert len(base) == len(set(base)) and len(filtered) == len(set(filtered))
    allowed = set(filtered); assert filtered == [k for k in base if k in allowed]
    assert set(base) & positive == set(filtered) & positive
    old_ranks = {key: i + 1 for i, key in enumerate(base) if key in positive}
    new_ranks = {key: i + 1 for i, key in enumerate(filtered) if key in positive}; area = 0
    for k in range(len(base) + 1):
        left, right = set(base[:k]) & positive, set(filtered[:k]) & positive
        assert left <= right; area += len(right) - len(left)
    advance = sum(old_ranks[key] - new_ranks[key] for key in old_ranks); assert area == advance
    return dict(positive_rank_sum_advance=advance, positive_modes_advanced=sum(new_ranks[key] < old_ranks[key] for key in old_ranks),
        first_positive_call=min(new_ranks.values(), default=None), last_positive_call=max(new_ranks.values(), default=None),
        rank_details=[dict(pattern=key, base_rank=old_ranks[key], new_rank=new_ranks[key]) for key in old_ranks])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); a = p.parse_args()
    root = a.project / 'results/finite_geometry_prefix/diagnostic'; refroot = a.project / 'results/posterior_state_reuse'
    inherited = json.loads((a.project / 'results/joint_credit_primal_rays/diagnostic/protocol.json').read_text()); hashes = inherited['source_sha256'].copy()
    for name, h in hashes.items(): assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == h, name
    hashes[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sourcepath = a.project / 'results/batched_credit_bank/development/episodes.json'; online = json.loads(sourcepath.read_text())
    oldrows = {(r['seed'], r['method']): r for r in online if r['repetition'] == 0 and r['n_context'] == 4}
    protocol = dict(seeds=list(range(5900000, 5900016)), display_budgets=[8, 16, 24, 32, 48, 64], source_sha256=hashes,
        online_source_sha256=hashlib.sha256(sourcepath.read_bytes()).hexdigest(),
        primary='all-positive rank advancement and exact every-K inclusion; local additions beyond same-path BP bank + C20',
        scope='old support diagnostic; geometry calls not equal-time; ideal numerical conditional risks, not actual query readout; no confirmation')
    root.mkdir(parents=True, exist_ok=True); assert not (root / 'protocol.json').exists(); (root / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    refs = {r['seed']: r for r in json.loads((refroot / 'first_write_reference/coverage.json').read_text())}; rows = []; curves = []; audits = []; proofs = []
    for seed in protocol['seeds']:
        rng = np.random.default_rng(seed); truth = rng.uniform(-.12, .12, 4); allx = rng.uniform(0, 1, 24)
        allv = model.base.forward(allx, truth) + np.random.default_rng(seed + 19000000).uniform(-model.base.EPS, model.base.EPS, 24); x, v = allx[:4], allv[:4]; completed = []
        for name, gen, steps in [('alm16', 'alm', 16), ('adam8', 'adam', 8), ('adam16', 'adam', 16)]:
            cfg = dict(generator=gen, sweeps=steps, steps=steps, restarts=64)
            bank, regs, _ = model.previous.discover(x, v, cfg); keys = [r.tobytes().hex() for r in regs]
            base = [key for key, skip in zip(keys, model.screen.contract(x, v, regs, 5)) if not skip]; sequences = dict(none=base)
            kind = 'local' if gen == 'alm' else 'bp'; source_name = f'{name}_batch_bank' if gen == 'alm' else f'{name}_bp_bank'
            _, retained, meta, _ = model.prepare(x, v, dict(**cfg, credit_kind=kind, bank_size=32))
            for field in ['matching_mode_keys', 'matching_lower_bounds', 'cross_mode_keys', 'cross_lower_bounds']: assert meta[field] == oldrows[seed, source_name][field]
            sequences['local_bank' if gen == 'alm' else 'bp_bank'] = [r.tobytes().hex() for r in retained]
            if gen == 'alm':
                _, remaining, rm, _ = model.prepare(x, v, dict(**cfg, credit_kind='residual', bank_size=32))
                for field in ['matching_mode_keys', 'matching_lower_bounds', 'cross_mode_keys', 'cross_lower_bounds']: assert rm[field] == oldrows[seed, 'alm16_batch_residual'][field]
                sequences['residual_bank'] = [r.tobytes().hex() for r in remaining]
                sequences['contract20'] = [key for key, skip in zip(keys, model.screen.contract(x, v, regs, 20)) if not skip]
                shadow = model.Collector(x, v, 'bp', True); anchor = np.zeros(4); starts, _ = model.old.interface.select_pool(x, v, anchor, model.old.interface.make_pool(4, 'prior256'), 64)
                with model.old.core.pipeline.discovery_box(.12):
                    actual = model.trace.refine(starts, x, v, anchor, 'alm', 16, shadow)
                    expected = model.trace.original(starts, x, v, anchor, 'alm', 16)
                assert np.array_equal(actual, expected)
                assert set(shadow.forward) | set(shadow.split) == {bytes.fromhex(k) for k in keys}
                shadow_kept, captured = filter_collector(x, v, regs, shadow); sequences['bp_shadow_bank'] = [k.hex() for k in shadow_kept]
                bp20 = set(sequences['bp_shadow_bank']) & set(sequences['contract20'])
                sequences['bp_shadow_c20'] = [k for k in base if k in bp20]
                local = set(sequences['local_bank']); sequences['bp_shadow_c20_plus_local'] = [k for k in base if k in bp20 and k in local]
                for proof in captured:
                    reg = np.frombuffer(bytes.fromhex(proof['pattern']), np.uint8).reshape(4, 4)
                    with model.old.core.pipeline.discovery_box(.12): exact = exact_certificate(x, v, reg, proof['p'], proof['a'])
                    value = Fraction(int(exact['numerator']), int(exact['denominator'])); assert value > 0 and Fraction(proof['lower']) <= value
                    _, _, g, rhs = model.previous.neighbor.pattern_matrix(x, v, reg); lp = linprog(np.zeros(4), A_ub=g, b_ub=rhs, bounds=[(-.12, .12)] * 4); assert lp.status == 2
                    proofs.append(dict(seed=seed, pattern=proof['pattern'], a=proof['a'].tolist(), p=proof['p'].tolist(), lower=proof['lower'], exact=exact, stage=proof['stage'], lp_status=int(lp.status)))
            completed.append((name, sequences, source_name)); audits.append(dict(seed=seed, method=name, original_online_credit_keys_and_bounds_exact=True, proposal_keys=keys, sequences=sequences))
        # All candidate and shadow sequences above were formed without the following reference.
        path = refroot / 'first_write_reference' / refs[seed]['reference_file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == refs[seed]['reference_sha256']; full = json.loads(path.read_text())
        positive = {r['pattern'] for r in full['positive_regions']}; ca = json.loads((refroot / f'conditional_risk/audit_{seed}.json').read_text()); path = refroot / 'conditional_risk' / ca['curve_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ca['curve_sha256']
        with np.load(path) as curve:
            pattern_keys = list(curve['patterns']); weights, means, grid = curve['weights'], curve['region_means'], curve['q']; assert set(pattern_keys) == positive
            for name, sequences, source_name in completed:
                base = sequences['none']; assert set(base) & positive == set(oldrows[seed, source_name]['positive_mode_keys'])
                for label, seq in sequences.items():
                    stats = check_prefix(base, seq, positive)
                    rows.append(dict(seed=seed, method=name, screen=label, total_geometry_calls=len(seq), **stats))
                    for budget in range(max(len(base), 64) + 1):
                        found = set(seq[:budget]); mask = np.array([k in found for k in pattern_keys]); score = risk(mask, weights, means, grid)
                        curves.append(dict(seed=seed, method=name, screen=label, budget=budget, **score))
                if name == 'alm16':
                    stats = check_prefix(sequences['bp_shadow_c20'], sequences['bp_shadow_c20_plus_local'], positive)
                    rows.append(dict(seed=seed, method=name, screen='local_increment_over_bp_c20', total_geometry_calls=len(sequences['bp_shadow_c20_plus_local']), **stats))
        (root / 'audits.json').write_text(json.dumps(audits, indent=2), encoding='utf-8'); (root / 'ranks.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(dict(completed=seed - protocol['seeds'][0] + 1, total=16, exact_shadow_proofs=len(proofs))), flush=True)
    summary = []
    for method, label in sorted({(r['method'], r['screen']) for r in rows}):
        rr = [r for r in rows if r['method'] == method and r['screen'] == label]
        summary.append(dict(method=method, screen=label, tasks=len(rr), tasks_positive_advanced=sum(r['positive_rank_sum_advance'] > 0 for r in rr),
            positive_rank_sum_advance=sum(r['positive_rank_sum_advance'] for r in rr), positive_modes_advanced=sum(r['positive_modes_advanced'] for r in rr),
            mean_geometry_calls=float(np.mean([r['total_geometry_calls'] for r in rr])), mean_first_positive_call=float(np.mean([r['first_positive_call'] for r in rr])), mean_last_positive_call=float(np.mean([r['last_positive_call'] for r in rr]))))
    displayed = []
    for method, label in sorted({(r['method'], r['screen']) for r in curves}):
        for budget in protocol['display_budgets']:
            cc = [r for r in curves if r['method'] == method and r['screen'] == label and r['budget'] == budget]
            assert len(cc) == 16
            displayed.append(dict(method=method, screen=label, budget=budget, tasks_with_positive=sum(r['mass'] > 0 for r in cc), mean_mass=float(np.mean([r['mass'] for r in cc])),
                mean_truncation_risk=None if any(r['truncation_risk'] is None for r in cc) else float(np.mean([r['truncation_risk'] for r in cc]))))
    result = dict(summary=summary, display_budgets=displayed, audited_sequences=len(rows), exact_shadow_proofs=len(proofs), source_hashes=len(hashes), scope=protocol['scope'])
    for name, data in [('summary.json', result), ('curves.json', curves), ('shadow_proofs.json', proofs)]: (root / name).write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps(dict(summary=summary, exact_shadow_proofs=len(proofs), audited_sequences=len(rows)), indent=2))


if __name__ == '__main__': main()
