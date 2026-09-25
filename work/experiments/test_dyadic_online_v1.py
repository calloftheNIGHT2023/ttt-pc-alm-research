"""335 real online fits, exact saved-array equivalence, matched timing ablations."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback
import numpy as np
import online_credit_dyadic_v1 as online
import support_consistency_trigger_v1 as oldtrigger
import support_consistency_trigger_dyadic_v1 as newtrigger
from posterior_confirmation_pipeline import discovery_box

BASE = 'results/online_credit_fresh_pilot'
DESIGN = 'outputs/ttt-pc-alm-research/335_dyadic_online_equivalence_protocol_v1.md'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def selftest():
    rng = np.random.default_rng(335193); count = 0
    for i in range(1032):
        b = rng.uniform(-.12, .12, 4); x = rng.uniform(0, 1, 4); v = rng.uniform(-.001, 1.001, 4)
        if i == 1024: b[:] = 0; x[:] = 0; v[:] = 0
        if i == 1025: b[:] = 0; x[:] = .5; v[:] = 0
        if i == 1026: b[:] = 0; x[:] = 1; v[:] = 0
        if i == 1027: b[:] = float.fromhex('0x0.0000000000001p-1022'); x[:] = 0
        if i == 1028: b[:] = 0; x[:] = 0; v[:] = .001
        if i == 1029: b[:] = 0; x[:] = 0; v[:] = np.nextafter(.001, 1.)
        if i == 1030: b[:] = .12; x[:] = 0; v[:] = -.001
        if i == 1031: b[:] = -.12; x[:] = 1; v[:] = 1.001
        assert oldtrigger.exact_forward(b, x) == newtrigger.exact_forward(b, x)
        assert oldtrigger.support_loss(b, x, v) == newtrigger.support_loss(b, x, v)
        count += 1
    return dict(exact_forward_pattern_loss_cases=count)


def run(root, out):
    start = time.perf_counter(); pred = root/BASE/'pilot_predictions_v2'; kernel = root/BASE/'dyadic_branch_kernel_v1'
    ks = read(kernel/'summary.json'); assert ks['passed'] and ks['saved_state_equivalences'] == 6144
    for path, digest in read(kernel/'protocol.json')['source_sha256'].items(): assert sha(root/path) == digest
    for name, digest in ks['outputs_sha256'].items(): assert sha(kernel/name) == digest
    ps = read(pred/'summary.json'); assert ps['passed'] and sha(pred/'rows.json') == ps['outputs_sha256']['rows.json']
    source = read(pred/'protocol.json')['source_sha256']
    for name, digest in source.items(): assert sha(root/'work/experiments'/name) == digest
    newnames = ['support_consistency_trigger_dyadic_v1.py', 'online_credit_dyadic_v1.py',
                'test_dyadic_online_v1.py', 'branch_image_chain_dyadic_v1.py']
    newhash = {'work/experiments/'+name: sha(root/'work/experiments'/name) for name in newnames}
    newhash[DESIGN] = sha(root/DESIGN)
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    seeds = list(range(328000000, 328000008))
    rows = [r for r in read(pred/'rows.json') if r['seed'] in seeds and r['method'].startswith('online_')]
    rows.sort(key=lambda r: (r['seed'], r['method'])); assert len(rows) == 96
    write(out/'protocol.json', dict(source_sha256=newhash, frozen_source_sha256=source, seeds=seeds,
        implementations=online.IMPLEMENTATIONS, repetitions=2, order_seed=335929,
        kernel_summary_sha256=sha(kernel/'summary.json'), prediction_summary_sha256=sha(pred/'summary.json'),
        query_targets_accessed=False, threads=dict(blas=1, omp=1), arithmetic_only=True))
    test = selftest(); write(out/'selftest.json', test); print(test, flush=True)
    lookup = {(r['seed'], r['method']): r for r in rows}; names = sorted({r['method'] for r in rows})
    def load(row):
        assert sha(pred/row['file']) == row['sha256'] and sha(pred/row['metadata_file']) == row['metadata_sha256']
        with np.load(pred/row['file'], allow_pickle=False) as z: arrays = {n: z[n] for n in z.files}
        meta = read(pred/row['metadata_file'])['metadata']
        assert not meta['execution_failed'] and not meta['query_targets_accessed']
        return arrays, meta
    records = []; array_checks = 0; rng = np.random.default_rng(335929)
    with discovery_box(.12):
        a, m = load(rows[0])
        for impl in online.IMPLEMENTATIONS:
            online.fit(a['x_observed'], a['v_observed'], a['q_observed'], rows[0]['seed'],
                       policy=m['policy'], channel=m['channel'], implementation=impl)
        for seed in seeds:
            directory = out/str(seed); directory.mkdir()
            cached = {name: load(lookup[seed, name]) for name in names}
            jobs = [(name, impl, rep) for name in names for impl in online.IMPLEMENTATIONS for rep in range(2)]
            for order in rng.permutation(len(jobs)):
                name, impl, rep = jobs[int(order)]; expected, emeta = cached[name]
                a, meta, seconds = online.fit(expected['x_observed'], expected['v_observed'], expected['q_observed'],
                    seed, policy=emeta['policy'], channel=emeta['channel'], implementation=impl)
                assert set(a) == set(expected)-{'x_observed', 'v_observed', 'q_observed'}
                for field, value in a.items():
                    assert value.dtype == expected[field].dtype and value.shape == expected[field].shape
                    assert value.tobytes() == expected[field].tobytes(), (seed, name, impl, field)
                    array_checks += 1
                for field in ['positive_modes', 'original_positive_modes', 'selected_state']:
                    assert meta[field] == emeta[field], (seed, name, impl, field)
                assert {k: v for k, v in meta['proposal'].items() if k != 'meta'} == {
                    k: v for k, v in emeta['proposal'].items() if k != 'meta'}
                assert meta['no_global_bp_guard_enabled'] == (meta['channel'] != 'bp')
                assert not meta['query_targets_accessed'] and not meta['uses_complete_posterior_reference']
                record = dict(seed=seed, method=name, implementation=impl, repetition=rep,
                    seconds=seconds, search_seconds=meta['search_seconds'],
                    continuation_seconds=meta['continuation_collection_seconds'],
                    trigger_seconds=meta['trigger_selection_seconds'],
                    geometry_seconds=meta['original_geometry_seconds']+meta['extra_geometry_seconds'],
                    bitwise_arrays=len(a), positive_modes=len(meta['positive_modes']),
                    no_global_bp_guard=meta['no_global_bp_guard_enabled'])
                write(directory/f'{name}_{impl}_{rep}.json', record); records.append(record)
            print(dict(tasks=len(records)//72, total_tasks=8, calls=len(records),
                       bitwise_array_checks=array_checks, seconds=time.perf_counter()-start), flush=True)
    assert len(records) == 576
    methods = []
    for name in names:
        times = {impl: statistics.fmean(r['seconds'] for r in records if r['method'] == name and r['implementation'] == impl)
                 for impl in online.IMPLEMENTATIONS}
        faster = sum(statistics.fmean(r['seconds'] for r in records if r['method'] == name and r['seed'] == seed and r['implementation'] == 'integer_dp_trigger') <
                     statistics.fmean(r['seconds'] for r in records if r['method'] == name and r['seed'] == seed and r['implementation'] == 'fraction') for seed in seeds)
        methods.append(dict(method=name, mean_seconds=times,
            fractional_reduction=1-times['integer_dp_trigger']/times['fraction'], faster_tasks=faster, tasks=8))
    for path, digest in newhash.items(): assert sha(root/path) == digest
    for name, digest in source.items(): assert sha(root/'work/experiments'/name) == digest
    write(out/'calls.json', records); write(out/'methods.json', methods)
    summary = dict(passed=True, tasks=8, methods=12, implementations=3, calls=576,
        bitwise_array_checks=array_checks, trigger_selftest=test, query_targets_accessed=False,
        timing_scope='eight development tasks, two repetitions; not 576 independent tasks',
        retained_prediction_state_unchanged=True, peak_memory_measured=False, core_research_goal_complete=False,
        seconds=time.perf_counter()-start,
        outputs_sha256={n: sha(out/n) for n in ['protocol.json', 'selftest.json', 'calls.json', 'methods.json']})
    write(out/'summary.json', summary); print(summary, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/BASE/'dyadic_online_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        write(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
