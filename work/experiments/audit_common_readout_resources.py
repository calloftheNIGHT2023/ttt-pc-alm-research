"""Separate-process memory audit. Do not use instrumented timing as a benchmark."""
import argparse, hashlib, json, time, tracemalloc
from pathlib import Path
import numpy as np
import psutil
import event_affine_memory as candidate
import common_cell_readout as geometry
base = candidate.base
p = argparse.ArgumentParser(); p.add_argument('--method', required=True); p.add_argument('--out', type=Path, required=True)
args = p.parse_args(); assert not args.out.exists()
configs = json.loads(Path(__file__).with_name('common_cell_development.json').read_text())
cfg = next(c for c in configs['configs'] if c['name'] == args.method); base.BOUND = .2
seed = 5400000; rng = np.random.default_rng(seed); truth = rng.uniform(-.2, .2, (3, 8))
weights = base.family.make_weights(3, 8); x = rng.uniform(-1, 1, (24, 8)); q = rng.uniform(-1, 1, (512, 8))
v = base.forward(truth[None], x, weights)[0] + np.random.default_rng(seed + 19000000).uniform(-base.EPS, base.EPS, x.shape)
process = psutil.Process(); before_rss = process.memory_info().rss; tracemalloc.start()
anchor = np.zeros_like(truth); stages = []
for n in configs['stages']:
    before = time.perf_counter()
    if cfg['method'] == 'affine': _, raw, fm = candidate.fit(x[:n], v[:n], anchor, weights, cfg)
    else: _, raw, fm = base.fit_internal(x[:n], v[:n], anchor, weights, cfg)
    point, region, pm = geometry.project(raw, anchor, x[:n], v[:n], weights)
    state, sm = geometry.fiber(point, region, x[:n], v[:n], weights)
    _, rm = geometry.read(q, weights, point, state, 'mean'); anchor = point.copy()
    current, peak = tracemalloc.get_traced_memory()
    stages.append(dict(n_context=n, instrumented_seconds=time.perf_counter()-before, tracked_peak_so_far=peak,
                       predictor_state_bytes=point.nbytes + (point.nbytes + 16 if state is not None else 0),
                       geometry_matrix_bytes=pm['geometry_matrix_bytes'], common_context_bytes=x[:n].nbytes+v[:n].nbytes,
                       known_weights_bytes=weights.nbytes, **rm))
current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop(); info = process.memory_info()
sources = [__file__, candidate.__file__, candidate.original.__file__, candidate.first.__file__, candidate.bias_solver.__file__,
           geometry.__file__, base.__file__, base.family.__file__]
result = dict(method=args.method, seed=seed, tracemalloc_peak_bytes=peak, retained_bytes=current,
              rss_before=before_rss, rss_after=info.rss, process_peak_working_set_bytes=getattr(info, 'peak_wset', None), stages=stages,
              scope='one old stream; global geometry included; native allocations may be missed; not timing evidence',
              source_sha256={Path(s).name: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources})
args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(dict(method=args.method, tracked_peak=peak)), flush=True)
