"""Independent algebra/constraint/integration checks; not performance evidence."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.integrate import quad_vec
import common_cell_readout as geometry
base = geometry.base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    base.BOUND = .2
    rng = np.random.default_rng(713619)
    weights = base.family.make_weights(3, 8)
    affine_gaps, projections, fibers = [], [], []
    for seed in range(5400000, 5400012):
        task = np.random.default_rng(seed)
        truth = task.uniform(-.2, .2, (3, 8))
        x = task.uniform(-1, 1, (24, 8))
        v = base.forward(truth[None], x, weights)[0] + np.random.default_rng(seed + 19000000).uniform(-base.EPS, base.EPS, x.shape)
        for n in [8, 16, 24]:
            cell = geometry.cell(truth, x[:n], weights)
            for _ in range(8):
                point = truth + rng.normal(size=truth.shape) * 1e-9
                assert np.max(cell['A'] @ point.ravel() - cell['b']) < 1e-7
                true = base.forward(point[None], x[:n], weights)[0].ravel()
                affine_gaps.append(float(np.max(np.abs(true - cell['J'] @ point.ravel() - cell['c']))))
            point, region, pm = geometry.project(truth, np.zeros_like(truth), x[:n], v[:n], weights)
            assert pm['projection_status'] == 'projected', pm
            assert pm['qp_stationarity_residual'] < 1e-6, pm
            state, fm = geometry.fiber(point, region, x[:n], v[:n], weights)
            projections.append(dict(seed=seed, n=n, **pm))
            if state is not None:
                query = task.uniform(-1, 1, (17, 8))
                exact, im = geometry.integrate_line(query, weights, state)
                independent, _ = quad_vec(lambda t: base.forward((point + t * state['direction'])[None], query, weights)[0],
                                          state['lower'], state['upper'], epsabs=1e-10, epsrel=1e-10, limit=10000)
                independent /= state['upper'] - state['lower']
                gap = float(np.max(np.abs(exact - independent)))
                assert gap < 1e-8, gap
                observed, _ = geometry.integrate_line(x[:n], weights, state)
                assert np.max(np.abs(observed - v[:n])) <= base.EPS + geometry.CHECK_TOL
                fibers.append(dict(seed=seed, n=n, integration_gap=gap, **fm, **im))
    # Cross many query activation branches; this is independent of task accuracy.
    quadrature_gaps = []
    for _ in range(6):
        point = rng.uniform(-.2, .2, (3, 8))
        direction = rng.normal(size=point.shape) * .5
        query = rng.uniform(-1, 1, (7, 8))
        state = dict(point=point, direction=direction, lower=-.4, upper=.6)
        exact, meta = geometry.integrate_line(query, weights, state)
        numeric, _ = quad_vec(lambda t: base.forward((point + t * direction)[None], query, weights)[0],
                              -.4, .6, epsabs=1e-10, epsrel=1e-10, limit=10000)
        gap = float(np.max(np.abs(exact - numeric)))
        assert gap < 1e-8, gap
        quadrature_gaps.append(gap)
    # At exact kinks, explicit cell branch maps must still equal true output.
    small_weights = [np.eye(4)]
    point = np.zeros((1, 4)); x = np.array([[-1., 0., 1., 2.]])
    region = geometry.cell(point, x, small_weights)
    assert np.max(np.abs(region['J'] @ point.ravel() + region['c'] - base.forward(point[None], x, small_weights)[0].ravel())) == 0
    result = dict(passed=True, affine_tests=len(affine_gaps), affine_max_gap=max(affine_gaps),
                  projections=projections, fiber_checks=fibers, quadrature_gaps=quadrature_gaps,
                  scope='teacher regions only for audit; truth prohibited from adaptation APIs',
                  source_sha256={Path(s).name: hashlib.sha256(Path(s).read_bytes()).hexdigest()
                                 for s in [__file__, geometry.__file__, base.__file__, base.family.__file__]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['projections', 'source_sha256']}), flush=True)


if __name__ == '__main__':
    main()
