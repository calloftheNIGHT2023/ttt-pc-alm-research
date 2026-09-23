"""264 common correction: fresh all-method timings and predictions, no truth."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import tracemalloc
import numpy as np
import torch
import conditioned_mode_geometry as corrected
import matched_budget_suite as suite
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha, dump


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve(); src = Path(__file__).parent
    base = root/'results/matched_budget_confirmation'; old = base/'confirmation'; pre = base/'geometry_preflight'
    out = base/'conditioned_confirmation'; out.mkdir(parents=True, exist_ok=True)
    assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert json.loads((pre/'summary.json').read_text())['passed']
    pp = json.loads((pre/'protocol.json').read_text()); hashes = dict(pp['source_sha256'])
    hashes[Path(__file__).name] = sha(Path(__file__))
    for n, h in hashes.items(): assert sha(src/n) == h, n
    design = root/'outputs/ttt-pc-alm-research/264_common_geometry_prequery_protocol.md'
    assert sha(design) == pp['design_sha256']
    p0 = json.loads((old/'protocol.json').read_text()); cfgs = p0['configs']
    oldsummary = json.loads((old/'summary.json').read_text()); before = json.loads((old/'before_query_manifest.json').read_text())
    assert oldsummary['passed'] and not oldsummary['query_targets_accessed']
    assert sha(old/'before_query_manifest.json') == oldsummary['before_query_manifest_sha256']
    assert sha(old/'rows.json') == before['rows_sha256']
    references = {(r['seed'], r['method']): r for r in json.loads((old/'rows.json').read_text())}
    start = time.perf_counter(); loaded, manifest = suite.oldfit.meta.load(root); loading = time.perf_counter()-start
    assert manifest == p0['checkpoint_manifest'] and suite.hashes_of_priors() == p0['prior_sha256']
    p = dict(source_sha256=hashes, design_sha256=sha(design), preflight_sha256=sha(pre/'summary.json'),
        original_before_query_manifest_sha256=sha(old/'before_query_manifest.json'), configs=cfgs,
        checkpoint_manifest=manifest, prior_sha256=p0['prior_sha256'], seeds=p0['seeds'],
        memory_seeds=p0['memory_seeds']+[5910048], additional_memory_seed_scope='known shared numerical failure, not chosen by query risk',
        primary=p0['primary'], query_points=257, n_context=4, mode_particles=2048,
        repetition_seed=249911, order_seed=264930, model_loading_seconds=loading,
        failure_policy=p0['failure_policy'], query_targets_accessed=False,
        scope='universal conditional geometry correction; same64 tasks and46 methods, not a second independent task sample')
    dump(out/'protocol.json', p); q = np.linspace(0, 1, 257); rows = []; memory = []; files = {}
    equal = 0; repaired = 0; begin = time.perf_counter()
    with suite.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        x, v = observations(5900001); start = time.perf_counter()
        for cfg in cfgs:
            a, m = corrected.fit(cfg, x[:4], v[:4], q, 5900001, loaded)
            assert not m['execution_failed'] and m['geometry_repair_count'] == 0
        warm = time.perf_counter()-start; jobs = [(s, c) for s in p['seeds'] for c in cfgs]
        for i in np.random.default_rng(p['order_seed']).permutation(len(jobs)):
            seed, cfg = jobs[i]; x, v = observations(seed)
            a, m = corrected.fit(cfg, x[:4], v[:4], q, seed, loaded)
            ref = references[seed, cfg['name']]; oldpath = old/ref['file']
            assert sha(oldpath) == ref['sha256'] == before['prediction_files'][ref['file']]
            if not ref['metadata']['execution_failed']:
                assert not m['execution_failed'] and m['geometry_repair_count'] == 0
                with np.load(oldpath) as z:
                    for k, value in a.items(): assert value.tobytes() == z[k].tobytes(), (seed, cfg['name'], k)
                equal += 1
            else:
                assert m['geometry_repair_count'] > 0
                repaired += not m['execution_failed']
            path = out/f"{seed}_{cfg['name']}.npz"; assert not path.exists()
            np.savez_compressed(path, x_observed=x[:4], v_observed=v[:4], q_observed=q, **a); files[path.name] = sha(path)
            rows.append(dict(seed=seed, method=cfg['name'], family=cfg['family'], readout=ref['readout'],
                file=path.name, sha256=files[path.name], metadata=m, seconds=m['charged_complete_seconds'],
                original_execution_failed=ref['metadata']['execution_failed']))
            if len(rows) % len(cfgs) == 0:
                dump(out/'rows.json', rows)
                print(json.dumps(dict(predictions_done=len(rows), total=len(jobs), failures=sum(r['metadata']['execution_failed'] for r in rows),
                    unchanged_successes=equal, repaired_original_failures=repaired, seconds=time.perf_counter()-begin)), flush=True)
        dump(out/'rows.json', rows)
        for seed in p['memory_seeds']:
            x, v = observations(seed)
            for cfg in cfgs:
                gc.collect(); tracemalloc.start(); a, m = corrected.fit(cfg, x[:4], v[:4], q, seed, loaded)
                current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
                with np.load(out/f"{seed}_{cfg['name']}.npz") as ref:
                    for k, value in a.items(): assert value.tobytes() == ref[k].tobytes(), (seed, cfg['name'], k)
                memory.append(dict(seed=seed, method=cfg['name'], traced_current_bytes=current, traced_peak_bytes=peak, metadata=m))
                if len(memory) % 10 == 0:
                    dump(out/'memory.json', memory); print(json.dumps(dict(memory_done=len(memory), total=len(p['memory_seeds'])*len(cfgs))), flush=True)
        dump(out/'memory.json', memory)
    for n, h in hashes.items(): assert sha(src/n) == h, n
    assert equal == 2912
    before = dict(protocol_sha256=sha(out/'protocol.json'), rows_sha256=sha(out/'rows.json'), memory_sha256=sha(out/'memory.json'),
                  prediction_files=files, query_targets_accessed=False)
    dump(out/'before_query_manifest.json', before)
    ans = dict(passed=True, predictions_complete=True, tasks=64, configs=len(cfgs), predictions=len(rows), memory_replays=len(memory),
        failures=sum(r['metadata']['execution_failed'] for r in rows), unchanged_successes=equal, repaired_original_failures=repaired,
        warmup_seconds=warm, seconds=time.perf_counter()-begin, before_query_manifest_sha256=sha(out/'before_query_manifest.json'),
        query_targets_accessed=False, next='independent corrected prediction and geometry audit before opening any query truth')
    dump(out/'summary.json', ans); print(json.dumps(ans), flush=True)


if __name__ == '__main__': main()
