"""Frozen development control: raw and common-projection causal trajectories."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import event_affine_memory as candidate
import common_cell_readout as geometry
base = candidate.base


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); args = p.parse_args()
    cfg = json.loads(args.config.read_text()); base.BOUND = cfg['discovery_bound']
    args.out.mkdir(parents=True, exist_ok=True); assert not (args.out / 'protocol.json').exists()
    sources = [Path(s) for s in [__file__, candidate.__file__, candidate.original.__file__,
               candidate.first.__file__, candidate.bias_solver.__file__, base.__file__,
               base.family.__file__, geometry.__file__]]
    protocol = dict(**cfg, source_sha256={s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    scope='12 observed development streams; global common geometry; no confirmation',
                    anchor_rule='raw keeps raw point; common point/midpoint/mean share projected point anchor',
                    time_rule='shared discovery/projection charged fully to each readout; fiber construction charged to midpoint/mean only',
                    selection='raw selected cell only; no extra restarts or query-target selection')
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    weights = base.family.make_weights(cfg['depth'], cfg['width']); rows = []
    order = np.random.default_rng(721619)
    for seed in range(cfg['seed0'], cfg['seed0'] + cfg['count']):
        rng = np.random.default_rng(seed); truth = rng.uniform(-base.PRIOR, base.PRIOR, (cfg['depth'], cfg['width']))
        x = rng.uniform(-1, 1, (max(cfg['stages']), cfg['width'])); q = rng.uniform(-1, 1, (cfg['queries'], cfg['width']))
        v = base.forward(truth[None], x, weights)[0] + np.random.default_rng(seed + 19000000).uniform(-base.EPS, base.EPS, x.shape)
        target = base.forward(truth[None], q, weights)[0]
        jobs = [(config, common) for config in cfg['configs'] for common in [False, True]]
        for ci in order.permutation(len(jobs)):
            config, common = jobs[ci]; anchor = np.zeros_like(truth)
            for n in cfg['stages']:
                before = time.perf_counter()
                if config['method'] == 'affine':
                    _, raw, fit_meta = candidate.fit(x[:n], v[:n], anchor, weights, config)
                else:
                    _, raw, fit_meta = base.fit_internal(x[:n], v[:n], anchor, weights, config)
                fit_seconds = time.perf_counter() - before
                if common:
                    point, region, projection_meta = geometry.project(raw, anchor, x[:n], v[:n], weights)
                    state, fiber_meta = geometry.fiber(point, region, x[:n], v[:n], weights)
                    modes = ['point', 'midpoint', 'mean']
                else:
                    point, state, projection_meta, fiber_meta, modes = raw, None, {}, {}, ['raw']
                anchor = point.copy()
                for mode in modes:
                    before = time.perf_counter()
                    prediction, read_meta = geometry.read(q, weights, point, state, 'point' if mode == 'raw' else mode)
                    read_seconds = time.perf_counter() - before
                    support, _ = geometry.read(x[:n], weights, point, state, 'point' if mode == 'raw' else mode)
                    error = float(np.max(np.abs(support - v[:n])))
                    adaptation = fit_seconds + projection_meta.get('projection_seconds', 0.)
                    if mode in ['midpoint', 'mean']:
                        adaptation += fiber_meta['fiber_seconds']
                    rows.append(dict(method=config['name'], mode=mode, seed=seed, n_context=n,
                                     query_mse=float(np.mean((prediction - target) ** 2)),
                                     support_max_error=error, support_feasible=bool(error <= base.EPS + base.TOL),
                                     adaptation_seconds=adaptation, discovery_seconds=fit_seconds,
                                     read_queries_seconds=read_seconds, selected_raw=raw.tolist(), anchor_output=point.tolist(),
                                     base_fit_meta=fit_meta, common_context_bytes=x[:n].nbytes + v[:n].nbytes,
                                     **projection_meta, **fiber_meta, **read_meta))
        (args.out / 'episodes.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(dict(completed=seed-cfg['seed0']+1, total=cfg['count'], rows=len(rows))), flush=True)
    print(json.dumps(dict(complete=True, rows=len(rows))), flush=True)


if __name__ == '__main__':
    main()
