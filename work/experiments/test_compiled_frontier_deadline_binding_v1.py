"""427 actual binding and baseline-route check; historical sources hash-only."""
from collections import Counter
from pathlib import Path
import ast
import os
import time
import traceback
import numpy as np
import compiled_frontier_deadline_io_v1 as io
import frontier_deadline_worker_v3 as worker
import prefix_deadline_worker_v2 as legacy
import audit_continuous_frontier_v1 as audit
import frontier_search_receipt_v1 as receipt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/io.BASE/'binding_preflight_v1'


def main():
    begin = time.perf_counter(); dependencies = io.dependencies(ROOT); configs = io.configs(ROOT)
    counts = Counter(); sessions = []
    assert configs[:42] == io.previous.configs(ROOT)
    syntax_names = {'continuous_frontier_prediction_v1.py', 'frontier_deadline_worker_v1.py',
        'frontier_deadline_worker_v2.py', 'frontier_search_receipt_v1.py', 'audit_continuous_frontier_v1.py',
        'compiled_continuous_frontier_v1.py', 'piecewise_frontier_enclosure_v1.py', 'frontier_deadline_worker_v3.py',
        'test_compiled_continuous_frontier_v1.py', 'compiled_frontier_deadline_io_v1.py',
        'run_compiled_frontier_deadline_v1.py', 'audit_compiled_frontier_deadline_v1.py',
        'evaluate_compiled_frontier_deadline_v1.py', 'test_continuous_frontier_v1.py', 'test_frontier_receipt_v1.py', 'frontier_deadline_io_v1.py',
        'run_frontier_deadline_v1.py', 'audit_frontier_deadline_v1.py', 'evaluate_frontier_deadline_v1.py',
        Path(__file__).name}
    for path in dependencies['source_sha256']:
        if Path(path).name in syntax_names:
            ast.parse((ROOT/path).read_text(encoding='utf-8-sig'), filename=path)
            counts['new_interface_python_sources_parsed'] += 1
    jobs = [(ci, c['name'], seed, n, budget) for ci, c, jj in
            io.jobs(configs, io.SEEDS, io.STAGES, io.BUDGETS) for seed, n, budget in jj]
    assert len(jobs) == len(set(jobs)) == 1792
    assert sum(ci < 42 for ci, *_ in jobs) == 1344
    assert sum(ci >= 42 for ci, *_ in jobs) == 448
    counts['frozen_unique_planned_calls'] = len(jobs)
    io.save(OUT/'protocol.json', dict(**dependencies, configs=configs, planned_calls=jobs,
        query_targets_accessed=False, performance_comparison=False))
    path = ROOT/'results/runtime_matched_prefix/development_v1/inputs/5920000.npz'
    manifest = io.read(path.parent.parent/'inputs_manifest.json'); assert io.sha(path) == manifest[path.name]
    d = io.load_arrays(path); assert set(d) == {'x', 'v', 'q'}
    for cfg in configs[42:]:
        session = worker.Session(cfg, ROOT)
        try:
            result = session.run(d['x'][:4], d['v'][:4], d['q'], seed=5920000, budget=20.)
            io.old.save_call(ROOT, OUT/cfg['name'], *result)
            p, meta, events, archive = result
            assert meta['selected'] == 'final' and meta['error'] is None and archive is not None
            counts.update(audit.audit_call(archive['arrays'], archive['metadata'], events))
            if any('frontier_receipt' in e for e in events):
                counts.update(receipt.audit_events(d['x'][:4], d['v'][:4], d['q'], events))
            counts['new_bound_worker_calls'] += 1
        finally:
            sessions.append(dict(method=cfg['name'], **session.close()))
        print(dict(stage='new_binding_checked', method=cfg['name']), flush=True)
    for name in ['prior4096_ridge', 'official_ttt_native_prior256_32_p4', 'adam_gn64_256_anytime']:
        cfg = next(c for c in configs if c['name'] == name); results = []
        for label, cls in [('old402', legacy.Session), ('new425', worker.Session)]:
            session = cls(cfg, ROOT)
            try:
                result = session.run(d['x'][:4], d['v'][:4], d['q'], seed=5920000, budget=20.)
                io.old.save_call(ROOT, OUT/f'{name}_{label}', *result)
                assert result[1]['selected'] == 'final' and result[1]['error'] is None
                results.append(result)
            finally:
                sessions.append(dict(method=name, route=label, **session.close()))
        left, right = results
        assert left[3]['arrays'].keys() == right[3]['arrays'].keys()
        for key in left[3]['arrays']:
            np.testing.assert_array_equal(left[3]['arrays'][key], right[3]['arrays'][key])
            counts['legacy_route_identical_arrays'] += 1
        assert len(left[2]) == len(right[2])
        for a, b in zip(left[2], right[2]):
            assert a['kind'] == b['kind']; np.testing.assert_array_equal(a['prediction'], b['prediction'])
            counts['legacy_route_identical_packets'] += 1
        counts['legacy_route_pairs'] += 1
        print(dict(stage='old_binding_checked', method=name), flush=True)
    assert worker.Session.run is legacy.Session.run
    io.old.verify_hashes(ROOT, dependencies['source_sha256']); io.old.verify_hashes(ROOT, dependencies['pretrained_sha256'])
    io.save(OUT/'sessions.json', sessions)
    files = {p.relative_to(OUT).as_posix(): io.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    summary = dict(passed=True, checks=dict(counts), source_sha256=dependencies['source_sha256'],
        outputs_sha256=files, seconds=time.perf_counter()-begin, query_targets_accessed=False,
        performance_comparison=False, core_research_goal_complete=False)
    io.save(OUT/'summary.json', summary)
    print({k: v for k, v in summary.items() if k not in ['source_sha256', 'outputs_sha256']}, flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        main()
    except Exception:
        io.save(OUT/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
