"""387 frozen fresh-task predictions, all outputs sealed before any query truth."""
from pathlib import Path
import os
import platform
import sys
import time
import traceback
import numpy as np
import deadline_prediction_worker_v2 as model
import deadline_risk_registry_v1 as registry
import deadline_risk_io_v1 as io


def run(root, out):
    begin = time.perf_counter(); pre = root/io.BASE/'preflight_v1'
    assert io.read(pre/'summary.json')['passed'] and not (pre/'failure.json').exists()
    pp = io.read(pre/'protocol.json'); io.verify_hashes(root, pp['source_sha256']); io.verify_hashes(root, pp['pretrained_sha256'])
    audit_pre = root/io.BASE/'audit_preflight_v1'; ap = io.read(audit_pre/'summary.json')
    assert ap['passed'] and ap['preflight_summary_sha256'] == io.sha(pre/'summary.json')
    io.verify_hashes(root, ap['source_sha256'])
    h = io.prerequisites(root); history = io.explicit_seed_history(root, registry.SEEDS)
    configurations = registry.configs(); q = np.linspace(0., 1., 257)
    protocol = dict(**h, configs=configurations, seeds=registry.SEEDS, budgets=registry.BUDGETS,
        primary=registry.PRIMARY, primary_budget=registry.PRIMARY_BUDGET, main_controls=registry.MAIN_CONTROLS,
        block_size=registry.BLOCK_SIZE, expected_calls=2304, bootstrap_replicates=100000,
        preflight_summary_sha256=io.sha(pre/'summary.json'), explicit_seed_history=history,
        audit_preflight_summary_sha256=io.sha(audit_pre/'summary.json'),
        environment=dict(python=sys.version, platform=platform.platform(), blas_threads=1, omp_threads=1),
        query_grid=dict(points=257, lower=0., upper=1.), query_targets_accessed=False,
        new_development_not_independent_confirmation=True, official_matched_ttt_included=False,
        strict_equal_peak_memory_enforced=False, core_research_goal_complete=False)
    io.save(out/'protocol.json', protocol)
    inputs = {}
    for seed in registry.SEEDS:
        x, v = io.observations(seed); inputs[seed] = x, v
    (out/'inputs').mkdir(); (out/'sessions').mkdir()
    for seed, (x, v) in inputs.items(): np.savez_compressed(out/'inputs'/f'{seed}.npz', x=x, v=v, q=q)
    io.save(out/'inputs_manifest.json', {p.name:io.sha(p) for p in (out/'inputs').iterdir()})
    rows = []; session_rows = []
    for block, first in enumerate(range(0, len(registry.SEEDS), registry.BLOCK_SIZE)):
        seeds = registry.SEEDS[first:first+registry.BLOCK_SIZE]
        for ci in np.random.default_rng(np.random.SeedSequence([387101, block])).permutation(len(configurations)):
            ci = int(ci); config = configurations[ci]; session = model.Session(config, root); method_rows = []
            jobs = [(seed, budget) for seed in seeds for budget in registry.BUDGETS]
            try:
                for ji in np.random.default_rng(np.random.SeedSequence([387103, block, ci])).permutation(len(jobs)):
                    seed, budget = jobs[int(ji)]; x, v = inputs[seed]
                    job = dict(block=block, config_index=ci, method=config['name'], seed=seed, budget=budget)
                    io.save(out/'current_job.json', job)
                    result = session.run(x, v, q, seed=seed, budget=budget)
                    directory = out/'calls'/f"{seed}_{config['name']}_b{int(budget*1000)}"
                    row = dict(**job, **io.save_call(root, directory, *result))
                    rows.append(row); method_rows.append(row)
                    if row['error'] is not None: raise RuntimeError(('Preserved worker exception', job, row['error']))
                    if len(rows) % 16 == 0:
                        print(dict(stage='new_task_deadline', calls=len(rows), total=2304,
                            final=sum(r['selected'] == 'final' for r in rows), seconds=time.perf_counter()-begin), flush=True)
            finally:
                session_record = dict(block=block, method=config['name'], **session.close())
                io.save(out/'sessions'/f'{block}_{config["name"]}.json', session_record); session_rows.append(session_record)
            io.save(out/f'sealed_{block}_{config["name"]}.json', method_rows)
        io.verify_hashes(root, h['source_sha256']); io.verify_hashes(root, h['pretrained_sha256'])
    assert len(rows) == len({(r['seed'], r['method'], r['budget']) for r in rows}) == 2304
    io.save(out/'rows.json', rows); io.save(out/'sessions.json', session_rows)
    io.verify_hashes(root, h['source_sha256']); io.verify_hashes(root, h['pretrained_sha256'])
    result = dict(passed=True, calls=len(rows), independent_tasks=32, seconds=time.perf_counter()-begin,
        final=sum(r['selected'] == 'final' for r in rows), fallback=sum(r['selected'] == 'fallback' for r in rows),
        constant=sum(r['selected'] == 'constant' for r in rows), all_predictions_sealed=True,
        query_targets_accessed=False, core_research_goal_complete=False,
        outputs_sha256={p:io.sha(out/p) for p in ['protocol.json', 'inputs_manifest.json', 'rows.json', 'sessions.json']})
    io.save(out/'summary.json', result); print(dict(stage='new_task_deadline_sealed', **result), flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    root = Path(__file__).resolve().parents[2]; out = root/io.BASE/'development_v1'; out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
