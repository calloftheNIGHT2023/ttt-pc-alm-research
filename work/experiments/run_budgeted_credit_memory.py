"""Causal four-stage finite-geometry experiment with complete online costs."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import budgeted_credit_memory as model


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--out', type=Path, required=True); args = p.parse_args()
    cfg = json.loads(args.config.read_text()); inherited = json.loads(Path('results/finite_geometry_prefix/diagnostic/protocol.json').read_text()); hashes = inherited['source_sha256'].copy()
    for name, expected in hashes.items(): assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected, name
    for name in [Path(__file__).name, Path(model.__file__).name]: hashes[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    verification = model.verify(); diagnostic = {(r['seed'], r['method']): r for r in json.loads(Path('results/finite_geometry_prefix/diagnostic/audits.json').read_text())}; checks = 0
    for seed in range(cfg['seed0'], cfg['seed0'] + cfg['count']):
        rng = np.random.default_rng(seed); truth = rng.uniform(-.12, .12, 4); x = rng.uniform(0, 1, 24)
        v = model.base.forward(x, truth) + np.random.default_rng(seed + 19000000).uniform(-model.base.EPS, model.base.EPS, 24)
        for c in cfg['configs']:
            if 'diagnostic_method' not in c: continue
            _, regs, _ = model.prepare(x[:4], v[:4], c)
            assert [r.tobytes().hex() for r in regs] == diagnostic[seed, c['diagnostic_method']]['sequences'][c['diagnostic_screen']]; checks += 1
    verification['frozen_diagnostic_order_exact_cases'] = checks
    protocol = dict(**cfg, source_sha256=hashes, config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(), verification=verification,
        primary_endpoint='mean unseen-query MSE across four causal stages; paired by task, not timing repeats',
        primary_increment='explicit BP-credit hybrid plus local history vs identical BP-credit hybrid without local history at K24',
        resource_endpoint='full fit plus query read, both timing repetitions averaged within task; full state and solver work charged',
        scope='16 old tasks; K24 is a development budget, not a fresh confirmation; later stages own saved state plus common direct128; local-only candidate has no global BP')
    args.out.mkdir(parents=True, exist_ok=True); assert not (args.out / 'protocol.json').exists(); (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rows = []; order = np.random.default_rng(942619); blocks = [(rep, seed) for rep in range(cfg['repetitions']) for seed in range(cfg['seed0'], cfg['seed0'] + cfg['count'])]; order.shuffle(blocks)
    for block, (rep, seed) in enumerate(blocks):
        rng = np.random.default_rng(seed); truth = rng.uniform(-.12, .12, 4); x = rng.uniform(0, 1, 24); q = rng.uniform(0, 1, cfg['queries']); target = model.base.forward(q, truth)
        v = model.base.forward(x, truth) + np.random.default_rng(seed + 19000000).uniform(-model.base.EPS, model.base.EPS, 24)
        for index in order.permutation(len(cfg['configs'])):
            c = cfg['configs'][index]; state = None
            for n in cfg['stages']:
                oldbytes = 0 if state is None else state.anchor.nbytes + state.samples.nbytes
                begin = time.perf_counter(); predict, state, meta = model.fit(x[:n], v[:n], state, dict(**c, archive=True, pool='posterior_mix', posterior_samples=cfg['posterior_samples'], proposal_budget=cfg['proposal_budget']))
                fit = time.perf_counter() - begin; begin = time.perf_counter(); prediction = predict(q); read = time.perf_counter() - begin
                support = float(np.max(np.abs(predict(x[:n]) - v[:n]))); filename = f'r{rep}_{seed}_{c["name"]}_n{n}.npz'; np.savez_compressed(args.out / filename, anchor=state.anchor, samples=state.samples)
                rows.append(dict(repetition=rep, seed=seed, method=c['name'], n_context=n, query_mse=float(np.mean((np.clip(prediction, 0, 1) - target) ** 2)), raw_query_mse=float(np.mean((prediction - target) ** 2)),
                    support_max_error=support, support_feasible=bool(support <= model.base.EPS + model.base.TOL), adaptation_seconds=fit, read_queries_seconds=read, previous_state_bytes=oldbytes,
                    state_file=filename, state_sha256=hashlib.sha256((args.out / filename).read_bytes()).hexdigest(), **meta))
        (args.out / 'episodes.json').write_text(json.dumps(rows, indent=2), encoding='utf-8'); print(json.dumps(dict(completed_blocks=block + 1, total_blocks=len(blocks), rows=len(rows))), flush=True)
    print(json.dumps(dict(complete=True, rows=len(rows))), flush=True)


if __name__ == '__main__': main()
