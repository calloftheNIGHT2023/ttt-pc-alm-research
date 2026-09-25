"""Replay online bank certificates, plus independent BP current/history checks."""
import argparse, hashlib, json
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import batched_credit_bank_memory as model
from certified_branch_solver import exact_certificate
from audit_bp_credit_reuse import Observer


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); a = p.parse_args()
    root = a.project / 'results/batched_credit_bank'; inp = root / 'development'; out = root / 'certificate_audit'
    protocol = json.loads((inp / 'protocol.json').read_text()); rows = json.loads((inp / 'episodes.json').read_text())
    lookup = {(r['seed'], r['method']): r for r in rows if r['repetition'] == 0 and r['n_context'] == 4}
    for name, h in protocol['source_sha256'].items(): assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == h, name
    proofs = []; matches = []; semantics = []
    for seed in range(protocol['seed0'], protocol['seed0'] + protocol['count']):
        rng = np.random.default_rng(seed); truth = rng.uniform(-.12, .12, 4); x = rng.uniform(0, 1, 24)
        v = model.base.forward(x, truth) + np.random.default_rng(seed + 19000000).uniform(-model.base.EPS, model.base.EPS, 24)
        for cfg in protocol['configs']:
            if cfg['backend'] != 'batch': continue
            _, regs, meta, captured = model.prepare(x[:4], v[:4], dict(**cfg, capture_proofs=True)); row = lookup[seed, cfg['name']]
            for prefix in ['matching', 'cross']:
                for suffix in ['_mode_keys', '_lower_bounds']: assert meta[prefix + suffix] == row[prefix + suffix]
            assert len(regs) == row['geometry_calls']; positive = set(row['positive_mode_keys'])
            for proof in captured:
                reg = np.frombuffer(bytes.fromhex(proof['pattern']), np.uint8).reshape(4, 4)
                with model.old.core.pipeline.discovery_box(.12): exact = exact_certificate(x[:4], v[:4], reg, proof['p'], proof['a'])
                value = Fraction(int(exact['numerator']), int(exact['denominator'])); assert value > 0 and Fraction(proof['lower']) <= value
                _, _, g, rhs = model.previous.neighbor.pattern_matrix(x[:4], v[:4], reg)
                lp = linprog(np.zeros(4), A_ub=g, b_ub=rhs, bounds=[(-.12, .12)] * 4)
                assert lp.status == 2 and proof['pattern'] not in positive
                proofs.append(dict(seed=seed, method=cfg['name'], pattern=proof['pattern'], p=proof['p'].tolist(), a=proof['a'].tolist(), lower=proof['lower'], exact=exact, stage=proof['stage'], lp_status=int(lp.status)))
            matches.append(dict(seed=seed, method=cfg['name'], exact_cut_keys_and_bounds=True, proofs=len(captured)))
        for steps in [8, 16]:
            evaluate = model.old.core.batched.evaluate; original = model.Collector.ensure_forward; observers = {}; checks = [0]
            def checked(collector, b):
                changed = collector.last is None or not np.array_equal(b, collector.last)
                original(collector, b)
                if collector.kind != 'bp': return
                obs = observers.setdefault(id(collector), Observer(collector.x, collector.v, evaluate)); obs.ensure(b)
                assert np.array_equal(collector.codes, obs.codes)
                assert np.array_equal(collector.current, obs.current) and np.array_equal(collector.history, obs.history)
                if changed: checks[0] += 1
            try:
                model.Collector.ensure_forward = checked
                cfg = next(c for c in protocol['configs'] if c['name'] == f'adam{steps}_bp_bank')
                model.prepare(x[:4], v[:4], cfg)
            finally: model.Collector.ensure_forward = original
            assert len(observers) == 1; obs = next(iter(observers.values())); assert obs.gradient_checks == checks[0]
            semantics.append(dict(seed=seed, method=f'adam{steps}', changed_parameter_events=checks[0], gradient_checks=obs.gradient_checks, maximum_gradient_error=obs.max_gradient_error, adjoints_history_and_codes_bitwise=True))
        print(json.dumps(dict(completed=seed - protocol['seed0'] + 1, total=protocol['count'], proofs=len(proofs))), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    result = dict(exact_replayed_first_writes=len(matches), exact_rational_and_lp_checks=len(proofs), matches=matches, bp_semantics=semantics,
        bp_gradient_checks=sum(r['gradient_checks'] for r in semantics), maximum_bp_gradient_error=max(r['maximum_gradient_error'] for r in semantics),
        source_sha256=protocol['source_sha256'], audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='support-only reconstruction; independent exact rational and LP rejection checks; BP histories compared to frozen 105 implementation')
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); (out / 'proofs.json').write_text(json.dumps(proofs, indent=2), encoding='utf-8')
    print(json.dumps(dict(complete=True, first_writes=len(matches), proofs=len(proofs), bp_gradient_checks=result['bp_gradient_checks'])))


if __name__ == '__main__': main()
