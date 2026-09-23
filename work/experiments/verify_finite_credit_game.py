"""Old-seed primitive: independent LP only outside the forbidden-call solve."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import scipy.optimize
import finite_credit_game as model
from verify_joint_credit_minimax import reduced_lp


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    raise AssertionError('LP/BP call forbidden during the finite credit game')


def guarded_solve(x, v, regs, bank):
    # Patch all already-imported aliases of the public LP entry, not only scipy.
    target = scipy.optimize.linprog; locations = []
    for module in list(sys.modules.values()):
        if module is None: continue
        for name, value in list(vars(module).items()):
            if value is target: locations.append((module, name))
    patches = [patch.object(module, name, forbidden) for module, name in locations]
    torch = sys.modules.get('torch')
    if torch is not None:
        patches += [patch.object(torch.autograd, 'backward', forbidden),
                    patch.object(torch.autograd, 'grad', forbidden), patch.object(torch.Tensor, 'backward', forbidden)]
    for p in patches: p.start()
    try: return model.solve(x, v, regs, bank)
    finally:
        for p in reversed(patches): p.stop()


def verify():
    rng = np.random.default_rng(5900001); teacher = rng.uniform(-.12, .12, 4); xx = rng.uniform(0, 1, 24)
    vv = model.original.base.forward(xx, teacher) + np.random.default_rng(24900001).uniform(-.001, .001, 24)
    counts = dict(oracle_cases=0, independent_lp=0, box_consistency=0, repeat_arrays=0, guarded_solves=0)
    maximum = 0.; positive = 0
    for n in (4, 8):
        x = xx[:n]; v = vv[:n]
        regs = np.array([model.original.base.pattern(x, b) for b in np.r_[teacher[None], rng.uniform(-.12, .12, (7, 4))]], np.uint8)
        bank = rng.normal(size=(9, 4, n)); bank /= abs(bank).max((1, 2), keepdims=True)
        zl, zh, hl, hh = model.original.screen.boxes(v, regs)
        for credit in np.r_[bank, np.zeros((1, 4, n))]:
            a = np.broadcast_to(credit, regs.shape); out = model.response(x, v, regs, a)
            assert out['valid'].all()
            reference = model.original.float_optimum(x, v, regs, a)
            maximum = max(maximum, float(np.max(abs(reference-out['objective']))))
            assert np.max(abs(reference-out['objective'])) < 1e-10
            assert np.all(out['z'] >= zl-1e-12) and np.all(out['z'] <= zh+1e-12)
            assert np.all(out['h'] >= hl-1e-12) and np.all(out['h'] <= hh+1e-12)
            assert np.max(abs(out['b'])) <= .12+1e-12
            prev = np.concatenate([np.broadcast_to(x, (len(regs), 1, n)), out['h'][:, :-1]], axis=1)
            assert np.max(abs(out['z']-prev-out['b'][:, :, None])) < 1e-12
            for i, reg in enumerate(regs):
                lp = reduced_lp(x, v, reg, credit[None]); assert lp.success
                maximum = max(maximum, abs(lp.fun-out['objective'][i]))
                assert abs(lp.fun-out['objective'][i]) < 1e-9; counts['independent_lp'] += 1
            counts['oracle_cases'] += len(regs); counts['box_consistency'] += len(regs)
        arrays, meta = guarded_solve(x, v, regs, bank)
        again, repeated = guarded_solve(x, v, regs, bank); counts['guarded_solves'] += 2
        for name, values in arrays.items():
            assert values.tobytes() == again[name].tobytes(); counts['repeat_arrays'] += 1
        assert meta['proofs'] == repeated['proofs']; positive += meta['positive']
    return dict(passed=True,checks=counts,max_objective_error=maximum,positive=positive,
                scope='Old 5900001 primitive, no query or prior LP solution supplied to finite solver')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--project', type=Path, required=True); root = ap.parse_args().project.resolve()
    parent = root/'results/joint_credit_minimax/development/protocol.json'
    hashes = dict(json.loads(parent.read_text())['source_sha256'])
    for name, value in hashes.items(): assert sha(Path(__file__).with_name(name)) == value, name
    audit = root/'results/round_221_audit.json'; assert json.loads(audit.read_text())['passed']
    result = verify()
    for name in ['audit_joint_credit_minimax.py','plot_joint_credit_minimax.py','audit_joint_credit_resources.py',
                 'analyze_joint_credit_margins.py','audit_research_round_221.py','finite_credit_game.py',Path(__file__).name]:
        hashes[name] = sha(Path(__file__).with_name(name))
    result.update(source_sha256=hashes,parent_protocol_sha256=sha(parent),parent_audit_sha256=sha(audit))
    out = root/'results/finite_credit_game/primitive'; out.mkdir(parents=True, exist_ok=True)
    assert not (out/'summary.json').exists()
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,sources=len(hashes),checks=result['checks'],max_error=result['max_objective_error'])),flush=True)


if __name__ == '__main__': main()
