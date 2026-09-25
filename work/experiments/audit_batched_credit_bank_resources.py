"""Fresh-process complete-stream resource accounting, not timed performance."""
import argparse, hashlib, json, tracemalloc
from pathlib import Path
import numpy as np
import psutil
import batched_credit_bank_memory as model


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True); p.add_argument('--method', required=True); a = p.parse_args()
    protocol = json.loads((a.input / 'protocol.json').read_text()); cfg = next(c for c in protocol['configs'] if c['name'] == a.method)
    for name, h in protocol['source_sha256'].items(): assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == h, name
    seed = protocol['seed0']; refs = {r['n_context']: r for r in json.loads((a.input / 'episodes.json').read_text()) if r['seed'] == seed and r['method'] == a.method and r['repetition'] == 0}
    rng = np.random.default_rng(seed); truth = rng.uniform(-.12, .12, 4); x = rng.uniform(0, 1, 24); q = rng.uniform(0, 1, protocol['queries']); target = model.base.forward(q, truth)
    v = model.base.forward(x, truth) + np.random.default_rng(seed + 19000000).uniform(-model.base.EPS, model.base.EPS, 24)
    process = psutil.Process(); before = process.memory_info()._asdict(); state = None; stages = []; tracemalloc.start()
    for n in protocol['stages']:
        oldbytes = 0 if state is None else state.anchor.nbytes + state.samples.nbytes
        pred, state, meta = model.fit(x[:n], v[:n], state, dict(**cfg, archive=True, pool='posterior_mix', posterior_samples=512, proposal_budget=1024))
        mse = float(np.mean((pred(q) - target) ** 2)); row = refs[n]
        assert mse == row['raw_query_mse'] and meta['positive_mode_keys'] == row['positive_mode_keys'] and meta['anchor_output'] == row['anchor_output']
        current, peak = tracemalloc.get_traced_memory()
        entry = dict(n_context=n, original_outputs_exact=True, current_bytes=current, peak_bytes=peak, old_plus_new_state_bytes=oldbytes + meta['persistent_state_bytes'], geometry_numeric_arrays_subtotal=meta['geometry_numeric_arrays_subtotal'], process_memory=process.memory_info()._asdict())
        for field in ['pending_credit_numeric_bytes', 'direction_bank_numeric_bytes', 'main_cross_batch_array_bytes_subtotal', 'matching_certified_patterns', 'cross_certified_patterns']: entry[field] = meta.get(field, 0)
        stages.append(entry)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    result = dict(method=a.method, seed=seed, tracked_peak_bytes=peak, stages=stages, baseline_process_memory=before, source_sha256=protocol['source_sha256'], audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='one old stream in a fresh process; tracked allocation peak can omit native memory, process peak includes imports; instrumented time is not benchmark')
    (a.input / f'resources_{a.method}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(method=a.method, tracked_peak_bytes=peak, original_outputs_exact=True)))


if __name__ == '__main__': main()
