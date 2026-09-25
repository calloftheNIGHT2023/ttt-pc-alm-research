"""Synthetic tests only: common geometry classification, not 308 task scoring."""
import argparse
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import complete_credit_mode_geometry_v1 as target
from exact_quadratic_events_v1 import encode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(encode(value), stream, indent=2, ensure_ascii=False, allow_nan=False)


def run(out):
    start = time.perf_counter()
    src = Path(__file__).parent
    names = [Path(__file__).name, 'complete_credit_mode_geometry_v1.py',
             'conditioned_mode_geometry.py', 'region_posterior_memory.py',
             'enumerate_support_modes.py', 'shared_mode_readout.py', 'streaming_branch_projection.py']
    save(out / 'protocol.json', dict(source_sha256={n: sha(src / n) for n in names},
         synthetic_only=True, real_task_geometry_accessed=False, query_targets_accessed=False,
         numerical_volume_relative_check=1e-9, seed=308713))
    ba, br = target.box_rows(4)
    e = [F(1), F(0), F(0), F(0)]
    cases = [('cube', [], [], 'positive_volume'),
             ('half_cube', [e], [F(0)], 'positive_volume'),
             ('empty', [e], [F(-1)], 'infeasible'),
             ('zero_width', [e, [-x for x in e]], [F(0), F(0)], 'zero_volume_or_empty'),
             ('constant_empty', [[F(0)] * 4], [F(-1, 2**80)], 'infeasible'),
             ('very_thin_nonempty', [e, [-x for x in e]], [F(1, 2**40), F(0)], None)]
    rows = []
    for name, extra_a, extra_r, expected in cases:
        a, r = extra_a + ba, extra_r + br
        result = target.classify_constraints(a, r)
        count = target.verify_certificates(a, r, encode(result))
        if expected is not None:
            assert result['classification'] == expected, (name, result)
        else:
            # The set is analytically full dimensional. A numerical failure
            # may remain unresolved, but must not become a false rejection.
            assert result['classification'] in ['positive_volume', 'unresolved'], result
        if name in ['cube', 'half_cube']:
            volume = float((2 * target.BOUND)**4) / (2 if name == 'half_cube' else 1)
            assert result['numerical_volume_available'] and abs(result['volume'] / volume - 1) < 1e-9
        rows.append(dict(name=name, expected=expected, certificate_checks=count, result=result))
    rng = np.random.default_rng(308713)
    for index in range(12):
        b, x = rng.uniform(-.1, .1, 4), rng.uniform(0., 1., 4)
        v = target.geometry.base.forward(x, b) + rng.uniform(-.0005, .0005, len(x))
        pattern = target.geometry.base.pattern(x, b)
        key = pattern.astype(np.uint8).tobytes().hex()
        _, a, r, na, nr = target.matrices(x, v, key)
        _, _, original_a, original_r = target.geometry.base.branch_polytope(x, v, b)
        assert np.array_equal(na, original_a) and np.array_equal(nr, original_r)
        proof = target.point_certificate(a, r, b)
        assert proof['strict_interior']
        result = target.classify_mode(x, v, key)
        assert result['positive_volume_certified'] and result['numerical_volume_available'], result
        checks = target.verify_certificates(a, r, encode(result))
        rows.append(dict(name=f'tent_{index}', mode=key, x_observed=x.tolist(), v_observed=v.tolist(),
                         generator_point=b.tolist(), certificate_checks=checks, result=result))
    # The verifier must reject a forged cube and an altered rejection bound.
    cube = deepcopy(encode(rows[0]['result']))
    cube['certificates'][0]['radius'] = '1'
    rejected = 0
    try:
        target.verify_certificates(ba, br, cube)
    except AssertionError:
        rejected += 1
    empty = deepcopy(encode(rows[2]['result']))
    weight_cert = next(c for c in empty['certificates'] if c['type'] == 'exact_box_separation')
    weight_cert['lower'] = '123'
    try:
        target.verify_certificates([e] + ba, [F(-1)] + br, empty)
    except AssertionError:
        rejected += 1
    assert rejected == 2
    save(out / 'rows.json', rows)
    summary = dict(passed=True, synthetic_geometry_cases=len(rows), independently_recomputed_float_matrices=12,
                   certificate_checks=sum(r['certificate_checks'] for r in rows), forged_certificates_rejected=rejected,
                   real_task_geometry_accessed=False, query_targets_accessed=False,
                   seconds=time.perf_counter() - start,
                   outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    save(out / 'summary.json', summary)
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args.out)
    except Exception:
        save(args.out / 'failure.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
