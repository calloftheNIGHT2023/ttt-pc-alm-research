"""419 all 42 existing controls plus seven common-frontier readout versions."""
import numpy as np
import runtime_matched_prefix_io_v1 as previous
import continuous_frontier_prediction_v1 as frontier

BASE = 'results/frontier_deadline'
DESIGN = 'outputs/ttt-pc-alm-research/419_frontier_deadline_protocol_v1.md'
PRIMARY = 'frontier_regional_active512'
PRIMARY_BUDGET = .5
SEEDS = list(previous.SEEDS)
STAGES = list(previous.STAGES)
BUDGETS = list(previous.BUDGETS)
old = previous.old
read, save, sha, load_arrays, observations = previous.read, previous.save, previous.sha, previous.load_arrays, previous.observations


def configs(root):
    original = previous.configs(root)
    extra = [dict(name='frontier_'+method, kind='frontier_receipt', method=method,
                  search_seconds=.25, max_expanded=65536, batch_queries=32) for method in frontier.METHODS]
    result = original+extra
    assert len(original) == 42 and len(result) == len({c['name'] for c in result}) == 49
    return result


def dependencies(root):
    result = previous.dependencies(root)
    checks = {}
    for stage in ['preflight_v1', 'receipt_preflight_v1']:
        folder = root/'results/continuous_frontier'/stage
        summary = read(folder/'summary.json')
        assert summary['passed'] and not (folder/'failure.json').exists()
        old.verify_hashes(root, summary['source_sha256'])
        for name, digest in summary['outputs_sha256'].items():
            assert sha(folder/name) == digest, name
        checks[stage] = sha(folder/'summary.json')
    result['source_sha256'][DESIGN] = sha(root/DESIGN)
    result.update(frontier_preflight_sha256=checks, unchanged_old_configurations=42,
                  frontier_receipts_charged_online=True)
    return result


def jobs(configurations, seeds, stages, budgets):
    for ci in np.random.default_rng(419101).permutation(len(configurations)):
        ci = int(ci); cfg = configurations[ci]
        all_jobs = [(seed, n, budget) for seed in seeds for n in stages for budget in budgets]
        order = np.random.default_rng(np.random.SeedSequence([419103, ci])).permutation(len(all_jobs))
        yield ci, cfg, [all_jobs[int(j)] for j in order]
