"""427 retain all 49 controls and add seven shared compiled-reader interfaces."""
import numpy as np
import frontier_deadline_io_v1 as uncompiled

BASE = 'results/compiled_frontier_deadline'
DESIGN = 'outputs/ttt-pc-alm-research/427_compiled_frontier_deadline_protocol_v1.md'
PRIMARY = 'compiled_frontier_regional_active512'
PRIMARY_BUDGET = .5
SEEDS, STAGES, BUDGETS = list(uncompiled.SEEDS), list(uncompiled.STAGES), list(uncompiled.BUDGETS)
previous, old = uncompiled.previous, uncompiled.old
read, save, sha = uncompiled.read, uncompiled.save, uncompiled.sha
load_arrays, observations = uncompiled.load_arrays, uncompiled.observations


def configs(root):
    original = uncompiled.configs(root)
    extra = [dict(c, name='compiled_'+c['name'], kind='compiled_frontier_receipt') for c in original[42:]]
    result = original+extra
    assert len(original) == 49 and len(result) == len({c['name'] for c in result}) == 56
    return result


def dependencies(root):
    result = uncompiled.dependencies(root)
    folder = root/'results/compiled_continuous_frontier/preflight_v1'
    summary = read(folder/'summary.json')
    assert summary['passed'] and not (folder/'failure.json').exists()
    old.verify_hashes(root, summary['source_sha256'])
    for name, digest in summary['outputs_sha256'].items():
        assert sha(folder/name) == digest, name
    result['source_sha256'].update({p.relative_to(root).as_posix(): sha(p) for p in sorted((root/'work/experiments').glob('*.py'))})
    for name in [DESIGN, 'outputs/ttt-pc-alm-research/425_compiled_continuous_frontier_protocol_v1.md']:
        result['source_sha256'][name] = sha(root/name)
    result.update(compiled_interface_preflight_sha256=sha(folder/'summary.json'),
        unchanged_previous_configurations=49, online_compilation_charged=True,
        frozen_reference_audits_compiled_packets=True)
    return result


def jobs(configurations, seeds, stages, budgets):
    for ci in np.random.default_rng(427101).permutation(len(configurations)):
        ci = int(ci); cfg = configurations[ci]
        all_jobs = [(seed, n, budget) for seed in seeds for n in stages for budget in budgets]
        order = np.random.default_rng(np.random.SeedSequence([427103, ci])).permutation(len(all_jobs))
        yield ci, cfg, [all_jobs[int(j)] for j in order]
