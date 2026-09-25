"""Synthetic tests before inspecting any stage 309 real-state outcomes."""
import argparse
from fractions import Fraction as F
from pathlib import Path
import random

import layer_credit_interaction_v1 as method
from test_complete_credit_amplitude_events_v2 import fixtures
from evaluate_complete_credit_mode_geometry_v1 import save, sha


def run(out):
    rows = []
    for name, state in fixtures():
        result, cost = method.combine(state)
        assert all(not r['float_value_discrepancy'] for r in result)
        rows.append(dict(name=name, masks=len(result), costs=cost,
                         float_mode_differences=sum(r['float_mode_discrepancy'] for r in result)))
    rng = random.Random(309912)
    derivative_checks = 0
    for _ in range(256):
        u = [F(rng.randint(-16, 16), 8) for _ in range(4)]
        slopes = [rng.choice([-2, 0, 2]) for _ in range(4)]
        clipped = [bool(rng.randrange(2)) for _ in range(4)]
        h = method.branch_response(u, slopes, clipped)
        tau = F(1, 100)
        def recurrence(a):
            values = [F(0)] * 4
            for j in range(3, -1, -1):
                if clipped[j]:
                    values[j] = F(1, 3)  # any constant fixed endpoint
                elif j == 3:
                    values[j] = (F(2, 7) - a[j]*u[j]) / (1+tau)
                else:
                    s = slopes[j]
                    values[j] = (F(1, 7) - a[j]*u[j] + s*(values[j+1] + a[j+1]*u[j+1]))/(1+s*s+tau)
            return values
        a0 = [F(1)]*4
        original = recurrence(a0)
        for j in range(4):
            aa = a0.copy(); aa[j] += F(1, 128)
            varied = recurrence(aa)
            assert varied == [v + h[k][j]/128 for k, v in enumerate(original)]
            derivative_checks += 4
    # A conditional, exact cancellation witness with a fixed/clipped output.
    h = method.branch_response([F(2), F(1)], [2, 0], [False, True])
    assert sum(h[0]) == 0 and h[0][0] != 0 and h[0][1] != 0
    assert method.branch_response([F(2), F(1)], [2, 0], [True, True]) == [[0, 0], [0, 0]]
    root = Path(__file__).resolve().parents[2]
    sources = ['layer_credit_interaction_v1.py', Path(__file__).name]
    result = dict(passed=True, fixtures=rows, exact_derivative_scalar_checks=derivative_checks,
                  conditional_cancellation_witness=True, all_clipped_counterexample=True,
                  real_task_data_accessed=False,
                  source_sha256={n: sha(root/'work/experiments'/n) for n in sources})
    save(out/'summary.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    run(args.out)
