"""305: fixed 38 x 8 x 3 complete untraced calls; no query-target access."""
import argparse
import gc
from pathlib import Path
import time

import numpy as np
import torch
import counterfactual_resource_suite_v1 as suite
import conditioned_mode_geometry as conditioned
import online_stasis_memory_v2 as online
from posterior_confirmation_pipeline import discovery_box
from diagnose_gradient_flat_split_states_v1 import read, save, sha

PRIMARY = 'online_stasis_alm_keep64'


def run(root, out):
    begin = time.perf_counter()
    hashes = suite.gate(root)
    old = root/'results/counterfactual_resources/calibration_v1'
    functional = root/'results/online_stasis_memory/functional_v2'
    old_summary = suite.complete(old)
    online_summary = suite.complete(functional)
    assert old_summary['timing_runs'] == 888 and online_summary['tasks'] == 64
    old_protocol = read(old/'protocol.json')
    online_protocol = read(functional/'protocol.json')
    for name, digest in online_protocol['source_sha256'].items():
        assert sha(root/'work/experiments'/name) == digest, name
        hashes[name] = digest
    hashes[Path(__file__).name] = sha(Path(__file__))
    cfgs = suite.catalogue(root)
    assert cfgs == old_protocol['configs']
    cfgs.append(dict(name=PRIMARY, family='online_stasis', group='online_stasis', rule='forward_stasis'))
    assert len(cfgs) == len({c['name'] for c in cfgs}) == 38
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    inputs = suite.observed_inputs(root, suite.frozen_inputs(root), suite.SEEDS)
    input_loading = time.perf_counter()-start
    start = time.perf_counter()
    loaded, manifest = suite.resources.legacy.oldfit.meta.load(root)
    model_loading = time.perf_counter()-start
    assert manifest == old_protocol['checkpoint_manifest']
    references = {}
    for r in read(old/'timings.json'):
        references.setdefault((r['seed'], r['method']), (old, r))
    for r in read(functional/'rows.json'):
        references[r['seed'], PRIMARY] = (functional, r)
    assert all((s, c['name']) in references for s in suite.SEEDS for c in cfgs)
    jobs = [(s, c, r) for s in suite.SEEDS for c in cfgs for r in range(3)]
    order = np.random.default_rng(305929).permutation(len(jobs))
    assert len(jobs) == 912
    save(out/'schedule.json', [dict(seed=jobs[int(i)][0], method=jobs[int(i)][1]['name'],
                                  repeat=jobs[int(i)][2]) for i in order])
    protocol = dict(source_sha256=hashes,
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/305_online_stasis_resource_protocol.md'),
        original_summary_sha256=sha(old/'summary.json'), functional_summary_sha256=sha(functional/'summary.json'),
        configs=cfgs, seeds=suite.SEEDS, repeats=3, primary=PRIMARY, order_seed=305929,
        warmup_seed=5910001, checkpoint_manifest=manifest,
        observed_loading_seconds=input_loading, model_loading_seconds=model_loading,
        shared_observed_array_bytes=sum(a.nbytes for tup in inputs.values() for a in tup),
        all_preloaded_model_bytes=sum(m['shared_model_bytes'] for m in manifest.values()),
        threads=dict(blas=1, omp=1, torch=torch.get_num_threads()), trace=False,
        query_targets_accessed=False, scope='Complete calls including range projection and validation; I/O, GC and equivalence checking excluded',
        memory_scope='Only reported named-state subtotals and shared checkpoints, not per-method peak memory',
        schedule_sha256=sha(out/'schedule.json'))
    save(out/'protocol.json', protocol)
    (out/'calls').mkdir()
    (out/'attempts').mkdir()
    environments = []; warmup = []; timings = []; files = {}; checks = []

    def check_environment():
        # Interference pauses future calls. No foreign process is terminated.
        while True:
            row = suite.environment_snapshot(root)
            row['paused'] = bool(row['other_research_or_git_pack_processes'])
            environments.append(row)
            save(out/'calls'/f'environment_{len(environments):04d}.json', row)
            if not row['paused']:
                return
            print(dict(phase='interference_pause', environment=row), flush=True)
            time.sleep(10)

    def invoke(cfg, seed):
        x, v, q = inputs[seed]
        repairs = []
        start = time.perf_counter()
        def runner(c, xx, vv, qq, ss, ll):
            if c['family'] != 'online_stasis':
                return suite.fit(c, xx, vv, qq, ss, ll, trace=False)
            a, m = online.fit(xx, vv, qq, ss, rule=c['rule'], trace=False)
            tick = time.perf_counter()
            for field in suite.FIELDS:
                a[field] = suite.project(a[field])
            m.update(range_projection_seconds=time.perf_counter()-tick,
                projected_output_array_bytes=sum(a[f].nbytes for f in suite.FIELDS))
            return a, m
        if cfg['family'] == 'online_stasis':
            a, m = suite.resources.legacy.guarded_fit(cfg, x, v, q, seed, loaded, runner=runner)
        else:
            with conditioned.geometry_scope(repairs):
                a, m = suite.resources.legacy.guarded_fit(cfg, x, v, q, seed, loaded, runner=runner)
            m.update(geometry_repair_log=repairs, geometry_repair_count=len(repairs))
        if m['execution_failed']:
            for field in suite.FIELDS:
                a[field] = suite.project(a[field])
        for field in suite.FIELDS:
            assert a[field].shape == q.shape and np.isfinite(a[field]).all()
            assert np.all((a[field] >= 0) & (a[field] <= 1))
        seconds = time.perf_counter()-start
        m['charged_complete_seconds'] = seconds
        return a, m, seconds

    def record(kind, cfg, seed, repeat, number):
        gc.collect()
        a, m, seconds = invoke(cfg, seed)
        # Preserve the attempted call before any out-of-timer consistency assertion.
        row = dict(seed=seed, method=cfg['name'], group=cfg['group'], repeat=repeat,
                   kind=kind, order=number, seconds=seconds, metadata=m)
        callname = f'{kind}_{number:04d}.json'
        save(out/'attempts'/callname, row)
        name = f'{seed}_{cfg["name"]}.npz'
        duplicate_checks = 0
        if name not in files:
            with (out/name).open('xb') as stream:
                np.savez_compressed(stream, **a)
            files[name] = sha(out/name)
        else:
            assert sha(out/name) == files[name]
            with np.load(out/name, allow_pickle=False) as z:
                assert set(z.files) == set(a)
                for k, value in a.items():
                    assert value.shape == z[k].shape and value.dtype == z[k].dtype
                    assert value.tobytes() == z[k].tobytes(), (seed, cfg['name'], k, 'repeat')
                    duplicate_checks += 1
        folder, ref = references[seed, cfg['name']]
        ref_path = folder/ref['file']
        assert sha(ref_path) == ref['sha256']
        common = []
        if not m['execution_failed']:
            assert not ref['metadata'].get('execution_failed', False)
            with np.load(ref_path, allow_pickle=False) as z:
                # All returned fields, not just predictions, must match frozen output.
                assert set(a).issubset(z.files), (seed, cfg['name'], set(a)-set(z.files))
                for k, value in a.items():
                    expected = suite.project(z[k]) if k in suite.FIELDS else z[k]
                    assert value.shape == expected.shape and value.dtype == expected.dtype
                    assert value.tobytes() == expected.tobytes(), (seed, cfg['name'], k, 'frozen')
                    common.append(k)
            if 'positive_modes' in m:
                assert m['positive_modes'] == ref['metadata']['positive_modes']
        checks.append(dict(kind=kind, order=number, seed=seed, method=cfg['name'],
            duplicate_array_checks=duplicate_checks, frozen_fields=common,
            reference=str(ref_path.relative_to(root)), reference_sha256=ref['sha256'],
            execution_failed=m['execution_failed']))
        row.update(file=name, sha256=files[name])
        save(out/'calls'/callname, row)
        return row

    check_environment()
    with discovery_box(.12):
        for cfg in cfgs:
            warmup.append(record('warmup', cfg, 5910001, 0, len(warmup)))
        save(out/'warmup.json', warmup)
        check_environment()
        for idx in order:
            seed, cfg, repeat = jobs[int(idx)]
            timings.append(record('timing', cfg, seed, repeat, len(timings)))
            if len(timings) % 38 == 0:
                check_environment()
                print(dict(phase='timing', completed=len(timings), total=912,
                           elapsed_seconds=time.perf_counter()-begin), flush=True)
    check_environment()
    save(out/'timings.json', timings)
    save(out/'files.json', files)
    save(out/'checks.json', checks)
    save(out/'environments.json', environments)
    primary = {(r['seed'], r['repeat']): r['seconds'] for r in timings if r['method'] == PRIMARY}
    cap = float(np.mean(list(primary.values())))
    methods = []
    for cfg in cfgs:
        rr = sorted((r for r in timings if r['method'] == cfg['name']), key=lambda r: (r['seed'], r['repeat']))
        assert len(rr) == 24
        times = np.array([r['seconds'] for r in rr])
        pairs = [dict(seed=r['seed'], repeat=r['repeat'], method_seconds=r['seconds'],
            candidate_seconds=primary[r['seed'], r['repeat']],
            method_minus_candidate=r['seconds']-primary[r['seed'], r['repeat']],
            method_over_candidate=r['seconds']/primary[r['seed'], r['repeat']]) for r in rr]
        methods.append(dict(method=cfg['name'], group=cfg['group'], mean_seconds=float(times.mean()),
            median_seconds=float(np.median(times)), p90_seconds=float(np.quantile(times, .9)),
            maximum_seconds=float(times.max()), mean_over_candidate=float(times.mean()/cap),
            paired_mean_difference=float(np.mean([r['method_minus_candidate'] for r in pairs])),
            paired_mean_ratio=float(np.mean([r['method_over_candidate'] for r in pairs])),
            failures=sum(r['metadata']['execution_failed'] for r in rr), pairs=pairs))
    selection = dict(primary=PRIMARY, budget_seconds=cap, budget_factor=1., sensitivity_factor=1.10,
        within_budget=[r['method'] for r in methods if r['mean_seconds'] <= cap],
        over_budget=[r['method'] for r in methods if r['mean_seconds'] > cap],
        sensitivity_110_percent=[r['method'] for r in methods if r['mean_seconds'] <= 1.10*cap],
        all_methods_retained=True, query_quality_used=False)
    save(out/'methods.json', methods)
    save(out/'selection.json', selection)
    for n, d in hashes.items():
        assert sha(root/'work/experiments'/n) == d, n
    result = dict(passed=True, timing_runs=len(timings), warmup_runs=len(warmup), methods=len(cfgs),
        saved_predictors=len(files), failed_timing_calls=sum(r['metadata']['execution_failed'] for r in timings),
        frozen_array_checks=sum(len(r['frozen_fields']) for r in checks),
        repeated_array_checks=sum(r['duplicate_array_checks'] for r in checks),
        interference_snapshots=sum(r['paused'] for r in environments), primary_seconds=cap,
        query_targets_accessed=False, new_task_quality_test=False, core_research_goal_complete=False,
        seconds=time.perf_counter()-begin, independent_audit_pending=True,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json', 'schedule.json', 'warmup.json', 'timings.json',
                                              'files.json', 'checks.json', 'environments.json', 'methods.json', 'selection.json']})
    save(out/'summary.json', result)
    print({k:v for k,v in result.items() if k != 'outputs_sha256'}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', required=True, type=Path)
    root = ap.parse_args().project.resolve()
    out = root/'results/online_stasis_resources/calibration_v1'
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(root, out)
    except BaseException as exc:
        save(out/'failure.json', dict(error_type=type(exc).__name__, message=str(exc), no_automatic_retry=True))
        raise
