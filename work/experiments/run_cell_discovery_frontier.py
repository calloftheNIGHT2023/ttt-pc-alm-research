"""Finite discovery budgets; identical global refinement/readout for every solver."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import event_affine_memory as candidate
import common_cell_readout as geometry
base = candidate.base


def discover(config, x, v, anchor, weights):
    if config['method'] == 'prior_cells':
        starts, meta = base.proposals(x, v, anchor, weights, features=256, restarts=config['restarts'])
        records, projected = [], []
        for point in starts:
            trial, region, pm = geometry.project(point, anchor, x, v, weights)
            records.append((trial, region, pm)); projected.append(trial)
        # Same support-only feasible/minimum-movement selection as internal fits.
        selected = base.select(np.array(projected), x, v, weights, anchor)
        index = next(i for i, point in enumerate(projected) if np.array_equal(point, selected))
        point, region, pm = records[index]
        meta.update(prior_cells_examined=len(starts), prior_cells_feasible=sum(r[2]['projection_status']=='projected' for r in records))
        return point, region, pm, meta
    if config['method'] == 'affine':
        _, raw, meta = candidate.fit(x, v, anchor, weights, config)
    else:
        _, raw, meta = base.fit_internal(x, v, anchor, weights, config)
    point, region, pm = geometry.project(raw, anchor, x, v, weights)
    return point, region, pm, meta


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); args = p.parse_args()
    cfg = json.loads(args.config.read_text()); base.BOUND = cfg['discovery_bound']
    args.out.mkdir(parents=True, exist_ok=True); assert not (args.out/'protocol.json').exists()
    sources = [Path(s) for s in [__file__, candidate.__file__, candidate.original.__file__, candidate.first.__file__,
               candidate.bias_solver.__file__, base.__file__, base.family.__file__, geometry.__file__]]
    protocol = dict(**cfg, source_sha256={s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    scope='observed development frontier; no query selection; same common geometry; not official PC/TTT reproduction',
                    anchor='projected point, even when reading a fiber function mean')
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    weights = base.family.make_weights(cfg['depth'], cfg['width']); rows = []; order = np.random.default_rng(726619)
    for seed in range(cfg['seed0'], cfg['seed0']+cfg['count']):
        rng = np.random.default_rng(seed); truth = rng.uniform(-base.PRIOR, base.PRIOR, (cfg['depth'], cfg['width']))
        x = rng.uniform(-1,1,(max(cfg['stages']),cfg['width'])); q = rng.uniform(-1,1,(cfg['queries'],cfg['width']))
        v = base.forward(truth[None], x, weights)[0] + np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,x.shape)
        target = base.forward(truth[None], q, weights)[0]
        for ci in order.permutation(len(cfg['configs'])):
            config = cfg['configs'][ci]; anchor = np.zeros_like(truth)
            for n in cfg['stages']:
                before = time.perf_counter(); point, region, pm, fm = discover(config,x[:n],v[:n],anchor,weights)
                state, sm = geometry.fiber(point,region,x[:n],v[:n],weights); adaptation = time.perf_counter()-before
                before = time.perf_counter(); prediction, rm = geometry.read(q,weights,point,state,'mean'); read_seconds=time.perf_counter()-before
                support,_=geometry.read(x[:n],weights,point,state,'mean'); error=float(np.max(np.abs(support-v[:n])))
                anchor=point.copy(); point_prediction=base.forward(point[None],q,weights)[0]
                rows.append(dict(method=config['name'],seed=seed,n_context=n,query_mse=float(np.mean((prediction-target)**2)),
                                 point_query_mse=float(np.mean((point_prediction-target)**2)),support_max_error=error,
                                 support_feasible=bool(error<=base.EPS+base.TOL),adaptation_seconds=adaptation,read_queries_seconds=read_seconds,
                                 anchor_output=point.tolist(),fit_meta=fm,**pm,**sm,**rm))
        (args.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=seed-cfg['seed0']+1,total=cfg['count'],rows=len(rows))),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__': main()
