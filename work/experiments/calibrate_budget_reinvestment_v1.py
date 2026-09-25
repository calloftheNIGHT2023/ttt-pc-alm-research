"""337 resource-only K selection, same complete fit accounting for all controls."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import tracemalloc
import numpy as np
import psutil
import torch
import budget_reinvestment_suite_v1 as suite
from posterior_confirmation_pipeline import discovery_box


def setup(root):
    hashes = suite.gate(root); preflight = root/suite.BASE/'preflight_v1'
    s = suite.complete(preflight); assert s['real_calls'] == 48 and s['exact_nested_pairs'] == 36
    assert suite.read(preflight/'protocol.json')['source_sha256'] == hashes
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    start = time.perf_counter(); loaded, checkpoint = suite.resources.old.resources.legacy.oldfit.meta.load(root)
    loading = time.perf_counter()-start
    prior = suite.read(root/'results/online_credit_fresh_pilot/pilot_predictions_v2/protocol.json')
    assert checkpoint == prior['checkpoint_manifest']
    return hashes, loaded, checkpoint, loading


def worker(root, out, name, seed, parent):
    hashes, loaded, checkpoint, loading = setup(root)
    environment = suite.resources.old.environment_snapshot(root)
    assert all(p['pid'] == parent for p in environment['other_research_or_git_pack_processes'])
    cfg = next(c for c in suite.catalogue(root) if c['name'] == name)
    inputs, manifest, _ = suite.observed(root, [seed]); process = psutil.Process(); gc.collect()
    before = process.memory_info()._asdict()
    with discovery_box(.12):
        tracemalloc.start()
        try:
            a, m, seconds = suite.invoke(cfg, *inputs[seed], seed, loaded)
            current, peak = tracemalloc.get_traced_memory()
        finally: tracemalloc.stop()
    after = process.memory_info()._asdict()
    with (out/'arrays.npz').open('xb') as f: np.savez_compressed(f, **a)
    suite.save(out/'record.json', dict(seed=seed, method=name, metadata=m,
        instrumented_seconds_not_benchmark=seconds, process_before=before, process_after=after,
        traced_current_bytes=current, traced_peak_bytes=peak, checkpoint_manifest=checkpoint,
        model_loading_seconds=loading, preloaded_model_bytes=sum(v['shared_model_bytes'] for v in checkpoint.values()),
        source_sha256=hashes, observed_inputs=manifest, arrays_sha256=suite.sha(out/'arrays.npz'),
        pid=os.getpid(), parent_pid=parent, environment=environment, query_targets_accessed=False,
        memory_scope='Absolute process lifetime peak includes imports and preloaded models; tracemalloc may omit native allocations; no peak subtraction'))


def run(root, out):
    hashes, loaded, checkpoint, loading = setup(root); configs = suite.catalogue(root)
    inputs, manifest, index = suite.observed(root, suite.SEEDS)
    start_environment = suite.resources.old.environment_snapshot(root)
    assert not start_environment['other_research_or_git_pack_processes'], start_environment
    protocol = dict(source_sha256=hashes, configs=configs, seeds=suite.SEEDS, repeats=2,
        warmup_seed=suite.SEEDS[0], memory_seeds=[suite.SEEDS[0], suite.SEEDS[-1]], order_seed=337929,
        observed_inputs=manifest, checkpoint_manifest=checkpoint, model_loading_seconds=loading,
        preloaded_model_bytes=sum(v['shared_model_bytes'] for v in checkpoint.values()),
        preflight_summary_sha256=suite.sha(root/suite.BASE/'preflight_v1/summary.json'),
        threads=dict(blas=1, omp=1, torch=1), query_targets_accessed=False, posterior_reference_accessed=False,
        environment_start=start_environment, time_scope='Complete fit plus projection; I/O, model loading, GC and memory instrumentation excluded')
    suite.save(out/'protocol.json', protocol); (out/'calls').mkdir(); (out/'memory_calls').mkdir()
    begin = time.perf_counter(); timings = []; warmup = []; files = {}; checks = 0; environments = []; memory = []
    def environment():
        e = suite.resources.old.environment_snapshot(root); environments.append(e)
        suite.save(out/'calls'/f'environment_{len(environments):04d}.json', e)
        assert not e['other_research_or_git_pack_processes'], e
    def deterministic(seed, cfg, a, m):
        nonlocal checks
        checks += suite.frozen_check(root, cfg, seed, a, m, index)
        file = f'{seed}_{cfg["name"]}.npz'
        if file not in files:
            with (out/file).open('xb') as f: np.savez_compressed(f, **a)
            files[file] = suite.sha(out/file)
        else:
            assert suite.sha(out/file) == files[file]
            with np.load(out/file, allow_pickle=False) as z: expected = {n: z[n] for n in z.files}
            checks += suite.array_checks(a, expected)
        for field in ['prediction', 'point_prediction']:
            assert a[field].shape == (257,) and np.isfinite(a[field]).all() and np.all((a[field] >= 0) & (a[field] <= 1))
        return file
    with discovery_box(.12):
        for cfg in configs:
            gc.collect(); a, m, seconds = suite.invoke(cfg, *inputs[suite.SEEDS[0]], suite.SEEDS[0], loaded)
            file = deterministic(suite.SEEDS[0], cfg, a, m)
            row = dict(seed=suite.SEEDS[0], method=cfg['name'], seconds=seconds, metadata=m, file=file, sha256=files[file])
            suite.save(out/'calls'/f'warmup_{len(warmup):03d}.json', row); warmup.append(row); del a, m
            if len(warmup) % 22 == 0: print(dict(warmups=len(warmup), total=88, seconds=time.perf_counter()-begin), flush=True)
        suite.save(out/'warmup.json', warmup); environment()
        jobs = [(seed, cfg, rep) for seed in suite.SEEDS for cfg in configs for rep in range(2)]
        assert len(jobs) == 1408
        for order, j in enumerate(np.random.default_rng(337929).permutation(len(jobs))):
            seed, cfg, rep = jobs[int(j)]; gc.collect()
            a, m, seconds = suite.invoke(cfg, *inputs[seed], seed, loaded)
            file = deterministic(seed, cfg, a, m)
            row = dict(seed=seed, method=cfg['name'], group=cfg['group'], repeat=rep, order=order,
                seconds=seconds, metadata=m, file=file, sha256=files[file])
            suite.save(out/'calls'/f'timing_{order:04d}.json', row); timings.append(row); del a, m
            if len(timings) % 44 == 0:
                environment(); print(dict(timing_calls=len(timings), total=1408, seconds=time.perf_counter()-begin), flush=True)
    means = {c['name']: sum(r['seconds'] for r in timings if r['method'] == c['name'])/16 for c in configs}
    failures = {c['name']: sum(r['metadata']['execution_failed'] for r in timings if r['method'] == c['name']) for c in configs}
    selection = suite.select(configs, means, failures)
    suite.save(out/'timings.json', timings); suite.save(out/'selection.json', selection)
    print(dict(time_selection_complete=True, common_k=selection['common_k'], reference_seconds=selection['budget_seconds'],
               memory_methods=len(selection['memory_methods']), highest_adam_still_within=selection['highest_adam_still_within']), flush=True)
    memory_configs = [c for c in configs if c['name'] in selection['memory_methods']]
    for seed in protocol['memory_seeds']:
        for cfg in memory_configs:
            environment(); directory = out/'memory_calls'/f'{seed}_{cfg["name"]}'
            cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', cfg['name'], '--seed', str(seed),
                   '--parent', str(os.getpid()), '--out', str(directory)]
            child = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if child.returncode:
                suite.save(out/'worker_failure.json', dict(command=cmd, returncode=child.returncode, stdout=child.stdout, stderr=child.stderr))
                raise RuntimeError('Memory worker failed; preserve outputs')
            row = suite.read(directory/'record.json')
            assert row['source_sha256'] == hashes and row['checkpoint_manifest'] == checkpoint
            assert suite.sha(directory/'arrays.npz') == row['arrays_sha256']
            with np.load(directory/'arrays.npz', allow_pickle=False) as z: a = {n: z[n] for n in z.files}
            deterministic(seed, cfg, a, row['metadata']); del a
            row['record_file'] = str((directory/'record.json').relative_to(out)); row['record_sha256'] = suite.sha(directory/'record.json')
            memory.append(row)
            if len(memory) % 4 == 0:
                print(dict(memory_calls=len(memory), total=2*len(memory_configs), seconds=time.perf_counter()-begin), flush=True)
    methods = []
    for cfg in configs:
        rr = [r for r in timings if r['method'] == cfg['name']]; mm = [r for r in memory if r['method'] == cfg['name']]
        times = np.array([r['seconds'] for r in rr]); assert len(rr) == 16 and len(mm) in [0, 2]
        methods.append(dict(method=cfg['name'], family=cfg['family'], k=cfg.get('k'), mean_seconds=means[cfg['name']],
            median_seconds=float(np.median(times)), p90_seconds=float(np.quantile(times, .9)), maximum_seconds=float(times.max()),
            failures=failures[cfg['name']], maximum_traced_peak_bytes=max((r['traced_peak_bytes'] for r in mm), default=None),
            maximum_absolute_lifetime_peak_wset=max((r['process_after'].get('peak_wset', 0) for r in mm), default=None),
            maximum_dp_entries=max(((r['metadata'].get('proposal') or {}).get('meta', {}).get('max_retained_dp_entries', 0) for r in rr), default=0),
            memory_probe_count=len(mm)))
    environment()
    for name, value in [('methods.json', methods), ('memory.json', memory), ('files.json', files), ('environments.json', environments)]: suite.save(out/name, value)
    assert suite.gate(root) == hashes
    result = dict(passed=True, methods=88, timing_runs=1408, warmup_runs=88, memory_runs=len(memory),
        saved_predictors=len(files), deterministic_array_checks=checks, numerical_failures=sum(failures.values()),
        selected_common_k=selection['common_k'], reference_seconds=selection['budget_seconds'],
        seconds=time.perf_counter()-begin, query_targets_accessed=False, core_research_goal_complete=False,
        outputs_sha256={n: suite.sha(out/n) for n in ['protocol.json', 'warmup.json', 'timings.json', 'selection.json',
            'methods.json', 'memory.json', 'files.json', 'environments.json']})
    suite.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path); parser.add_argument('--worker')
    parser.add_argument('--seed', type=int); parser.add_argument('--parent', type=int); args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]; out = args.out if args.out else root/suite.BASE/'calibration_v1'
    out.mkdir(parents=True, exist_ok=False)
    try:
        if args.worker: worker(root, out, args.worker, args.seed, args.parent)
        else: run(root, out)
    except Exception:
        suite.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
