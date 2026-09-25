"""427 new run -> independent all-method audit -> evaluator-only query risk."""
from pathlib import Path
import argparse
import os
import time
import traceback
import numpy as np
import compiled_frontier_deadline_io_v1 as io
import frontier_deadline_worker_v3 as model


def run(root, out):
    begin = time.perf_counter()
    preflight = root/io.BASE/'binding_preflight_v1'
    gate = io.read(preflight/'summary.json'); assert gate['passed'] and not (preflight/'failure.json').exists()
    io.old.verify_hashes(root, gate['source_sha256'])
    for path, digest in gate['outputs_sha256'].items():
        assert io.sha(preflight/path) == digest
    dependencies = io.dependencies(root); configurations = io.configs(root)
    dependencies['binding_preflight_sha256'] = io.sha(preflight/'summary.json')
    expected = len(configurations)*len(io.SEEDS)*len(io.STAGES)*len(io.BUDGETS)
    assert expected == 1792
    protocol = dict(**dependencies, configs=configurations, seeds=io.SEEDS, stages=io.STAGES,
        budgets=io.BUDGETS, primary=io.PRIMARY, primary_budget=io.PRIMARY_BUDGET, expected_calls=expected,
        stage='run', old_development_tasks=True, independent_confirmation=False, query_targets_accessed=False,
        matched_official_ttt_included=True, strict_equal_peak_memory_enforced=False,
        setup_receives_support=False, task_state_reused_across_prefixes=False)
    io.save(out/'protocol.json', protocol); (out/'inputs').mkdir(); (out/'sessions').mkdir()
    inputs = {}; q = np.linspace(0., 1., 257)
    for seed in io.SEEDS:
        x, v = io.observations(seed); inputs[seed] = (x, v)
        np.savez_compressed(out/'inputs'/f'{seed}.npz', x=x, v=v, q=q)
    io.save(out/'inputs_manifest.json', {p.name: io.sha(p) for p in (out/'inputs').iterdir()})
    rows = []; sessions = []
    for ci, cfg, jobs in io.jobs(configurations, io.SEEDS, io.STAGES, io.BUDGETS):
        session = model.Session(cfg, root); method_rows = []
        try:
            for seed, n, budget in jobs:
                x, v = inputs[seed]
                job = dict(config_index=ci, method=cfg['name'], seed=seed, n=n, budget=budget)
                io.save(out/'current_job.json', job)
                result = session.run(x[:n], v[:n], q, seed=seed, budget=budget)
                directory = out/'calls'/f"{seed}_n{n}_{cfg['name']}_b{int(budget*1000)}"
                row = dict(**job, **io.old.save_call(root, directory, *result))
                rows.append(row); method_rows.append(row)
                assert row['error'] is None, ('Preserved worker error', job, row['error'])
                if len(rows) % 16 == 0:
                    print(dict(stage='compiled_frontier_deadline_run', calls=len(rows), expected=expected,
                        final=sum(r['selected'] == 'final' for r in rows), seconds=time.perf_counter()-begin), flush=True)
        finally:
            lifecycle = dict(method=cfg['name'], **session.close()); sessions.append(lifecycle)
            io.save(out/'sessions'/f"{cfg['name']}.json", lifecycle)
        io.save(out/f"sealed_{cfg['name']}.json", method_rows)
        io.old.verify_hashes(root, dependencies['source_sha256'])
        io.old.verify_hashes(root, dependencies['pretrained_sha256'])
    assert len(rows) == len({(r['seed'], r['n'], r['method'], r['budget']) for r in rows}) == expected
    io.save(out/'rows.json', rows); io.save(out/'sessions.json', sessions)
    summary = dict(passed=True, calls=expected, old_tasks=4, stage='run', all_predictions_sealed=True,
        final=sum(r['selected'] == 'final' for r in rows), fallback=sum(r['selected'] == 'fallback' for r in rows),
        constant=sum(r['selected'] == 'constant' for r in rows), query_targets_accessed=False,
        goal_complete=False, seconds=time.perf_counter()-begin,
        outputs_sha256={f: io.sha(out/f) for f in ['protocol.json', 'inputs_manifest.json', 'rows.json', 'sessions.json']})
    io.save(out/'summary.json', summary); print(summary, flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['run', 'audit', 'evaluate', 'all'])
    args = parser.parse_args(); root = Path(__file__).resolve().parents[2]
    for action in (['run', 'audit', 'evaluate'] if args.action == 'all' else [args.action]):
        folder = {'run': 'development_v1', 'audit': 'audit_v1', 'evaluate': 'evaluation_v1'}[action]
        out = root/io.BASE/folder; out.mkdir(parents=True, exist_ok=False)
        try:
            if action == 'run':
                run(root, out)
            elif action == 'audit':
                from audit_compiled_frontier_deadline_v1 import audit
                audit(root, out)
            else:
                from evaluate_compiled_frontier_deadline_v1 import evaluate
                evaluate(root, out)
        except Exception:
            io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
            raise
