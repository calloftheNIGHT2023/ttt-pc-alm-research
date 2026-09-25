"""Small synthetic tests before any new old-task selections are inspected."""
import argparse
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from support_consistency_trigger_v1 import choose_first_fit, exact_forward, support_loss, locations, uniform_index
from evaluate_complete_credit_mode_geometry_v1 import save, sha


def run(out):
    x = np.array([.15, .32])
    bs = np.array([[0.], [.02], [.04]])
    pred, _ = exact_forward(bs[1], x)
    v = np.array(list(map(float, pred)))
    a = choose_first_fit(bs, x, v)
    assert a['support_fit'] and a['index'] == 1 and a['exact_forward_calls'] == 2
    b = choose_first_fit(bs, x, np.array([2., 2.]))
    assert b['fallback'] and b['index'] == 2
    c = choose_first_fit(np.array([[.04], [.04]]), x, np.array([2., 2.]))
    assert c['fallback'] and c['index'] == 0
    d = choose_first_fit(np.array([[.02], [.02]]), x, v)
    assert d['support_fit'] and d['index'] == 0
    locs = locations()
    assert len(locs) == len(set(locs)) == 1088
    assert locs[0] == (0,1,0) and locs[1055] == (0,32,32) and locs[1056] == (1,1,0) and locs[-1] == (1,32,0)
    for seed in range(30):
        assert uniform_index(seed) == uniform_index(seed) and 0 <= uniform_index(seed) < 1088
    # Feasible forward support is compatible with a nonzero stored dual.
    assert support_loss([0.], [.2], [.4]) == 0
    h = F(.4); u = F(.1); f = exact_forward([0.], [.2])[0][0]
    assert u + h-f != 0
    result = dict(passed=True, first_fit_cases=2, fallback_cases=2, enumerated_locations=1088,
                  deterministic_uniform_cases=30, zero_loss_nonzero_stored_credit_cases=1,
                  source_sha256={n:sha(Path(__file__).with_name(n)) for n in [Path(__file__).name, 'support_consistency_trigger_v1.py']},
                  real_data_accessed=False, query_targets_accessed=False)
    save(out/'summary.json', result)
    print(result)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); a=p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False); run(a.out)
