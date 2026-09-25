"""338 support-only census of strong feasible modes unreachable by any wider K."""
from collections import Counter
from pathlib import Path
import time
import traceback
import numpy as np
import budget_reinvestment_suite_v1 as suite
import complete_credit_mode_geometry_v1 as geometry
from audit_post_escape_geometry_v1 import certificates
from audit_stasis_escape_geometry_v1 import inequalities
from audit_online_credit_fresh_pools_v1 import exact_trigger

DESIGN = 'outputs/ttt-pc-alm-research/338_search_radius_reachability_protocol_v1.md'
STRONG = ['probe_all_alm64', 'probe_alm64_33', 'probe_then_adam1920_33', 'probe_then_adam3840_33']
PRIMARY = 'online_first_fit_dual'
PROBE = 'credit_control_probe33'


def distance(first, second):
    a, b = bytes.fromhex(first), bytes.fromhex(second)
    assert len(a) == len(b) == 16 and all(0 <= x <= 3 for x in a+b)
    return sum(x != y for x, y in zip(a, b))


def category(hamming, minimum):
    if minimum is None: return 'no_nonexcluded_shell'
    if hamming < minimum: return 'below_window'
    if hamming > min(16, minimum+2): return 'above_window'
    return 'within_window'


def main(root, out):
    begin = time.perf_counter(); resource = root/'results/budget_reinvestment/calibration_v1'
    rs = suite.complete(resource); audit = suite.complete(root/'results/budget_reinvestment/audit_v1')
    assert audit['calibration_summary_sha256'] == suite.sha(resource/'summary.json')
    assert suite.gate(root) == suite.read(resource/'protocol.json')['source_sha256']
    env = suite.resources.old.environment_snapshot(root)
    assert not env['other_research_or_git_pack_processes'], env
    pred = root/'results/online_credit_fresh_pilot/pilot_predictions_v2'; sealed = suite.complete(pred)
    assert sealed['tasks'] == 512
    rows = suite.read(pred/'rows.json'); lookup = {(r['seed'], r['method']): r for r in rows}
    hashes = suite.read(resource/'protocol.json')['source_sha256'].copy()
    for name in ['audit_search_radius_reachability_v1.py', 'audit_post_escape_geometry_v1.py',
                 'audit_stasis_escape_geometry_v1.py', 'audit_online_credit_fresh_pools_v1.py']:
        path = 'work/experiments/'+name
        if path in hashes: assert suite.sha(root/path) == hashes[path]
        hashes[path] = suite.sha(root/path)
    hashes[DESIGN] = suite.sha(root/DESIGN)
    suite.save(out/'protocol.json', dict(source_sha256=hashes, prediction_summary_sha256=suite.sha(pred/'summary.json'),
        resource_summary_sha256=suite.sha(resource/'summary.json'), strong_methods=STRONG, primary=PRIMARY,
        seeds=list(range(328000000, 328000512)), query_targets_accessed=False, posterior_reference_accessed=False,
        inference='Post-seal development reachability, not risk, posterior mass, or algorithm superiority', environment=env))
    tasks = []; counts = Counter(); allfiles = {}
    for seed in range(328000000, 328000512):
        logs = {}; sources = {}
        for name in [PRIMARY, PROBE]+STRONG:
            row = lookup[seed, name]; path = pred/row['metadata_file']
            assert suite.sha(path) == row['metadata_sha256']
            m = suite.read(path)['metadata']; assert not m['execution_failed'] and not m['query_targets_accessed']
            logs[name] = m; sources[name] = dict(file=row['metadata_file'], sha256=row['metadata_sha256'])
        row = lookup[seed, PRIMARY]; path = pred/row['file']; assert suite.sha(path) == row['sha256']
        with np.load(path, allow_pickle=False) as z: x, v, b = z['x_observed'], z['v_observed'], z['trigger_b']
        loss, original = exact_trigger(b, x, v); main = logs[PRIMARY]
        assert original == main['selected_state']['original_mode'] and str(loss) == main['selected_state']['exact_support_loss']
        assert main['original_positive_modes'] == logs[PROBE]['positive_modes']
        minimum = main['proposal']['minimum_nonexcluded_hamming']; pool = set(main['positive_modes'])
        missing = {name: set(logs[name]['positive_modes'])-pool for name in STRONG}
        union = set().union(*missing.values()); missing['strong_union'] = union
        classified = {}
        for key in sorted(union):
            note = geometry.classify_mode(x, v, key)
            a, rhs = inequalities(x, v, key); counts['independent_certificates'] += certificates(a, rhs, note)
            classified[key] = note; counts['unique_classifications'] += 1
            counts['classification_'+note['classification']] += 1
            if note['classification'] in ['infeasible', 'zero_volume_or_empty']:
                suite.save(out/f'contradiction_{seed}_{key}.json', dict(seed=seed, mode=key, classification=note, source_metadata=sources))
                raise AssertionError('A reported positive strong-pool mode conflicts with an exact new certificate')
        comparisons = []
        for name, modes in missing.items():
            entries = []
            for key in sorted(modes):
                h = distance(original, key); note = classified[key]
                entries.append(dict(mode=key, hamming=h, category=category(h, minimum),
                    positive_volume_certified=note['positive_volume_certified'],
                    numerical_volume_available=note['numerical_volume_available'], classification=note['classification']))
            comparisons.append(dict(method=name, missing_modes=len(entries), entries=entries))
        task = dict(seed=seed, support_fit=loss == 0, original_mode=original, minimum_nonexcluded_hamming=minimum,
            maximum_proposed_hamming=None if minimum is None else min(16, minimum+2),
            primary_positive_modes=len(pool), source_metadata=sources, source_arrays=dict(file=row['file'], sha256=row['sha256']),
            comparisons=comparisons, certificates=classified)
        file = f'task_{seed}.json'; suite.save(out/file, task); allfiles[file] = suite.sha(out/file)
        tasks.append({k: val for k, val in task.items() if k != 'certificates'}); counts['tasks'] += 1
        if counts['tasks'] % 32 == 0: print(dict(tasks=counts['tasks'], total=512, classifications=counts['unique_classifications'], seconds=time.perf_counter()-begin), flush=True)
    aggregate = []
    for name in STRONG+['strong_union']:
        rows = [(t, next(r for r in t['comparisons'] if r['method'] == name)) for t in tasks]
        positive = [(t, e) for t, r in rows for e in r['entries'] if e['positive_volume_certified']]
        aggregate.append(dict(method=name, tasks_with_missing_reported_modes=sum(bool(r['entries']) for _, r in rows),
            reported_missing_mode_pairs=sum(r['missing_modes'] for _, r in rows),
            certified_positive_mode_pairs=len(positive), unresolved_mode_pairs=sum(not e['positive_volume_certified'] for _, r in rows for e in r['entries']),
            certified_categories=dict(Counter(e['category'] for _, e in positive)),
            certified_hamming_histogram={str(h): sum(e['hamming'] == h for _, e in positive) for h in range(17)},
            tasks_with_certified_above_window=len({t['seed'] for t, e in positive if e['category'] == 'above_window'}),
            support_fit_tasks_with_certified_above_window=len({t['seed'] for t, e in positive if e['category'] == 'above_window' and t['support_fit']})))
    for path, digest in hashes.items(): assert suite.sha(root/path) == digest
    for name, value in [('tasks.json', tasks), ('methods.json', aggregate), ('files.json', allfiles)]: suite.save(out/name, value)
    result = dict(passed=True, counts=dict(counts), seconds=time.perf_counter()-begin,
        query_targets_accessed=False, posterior_reference_accessed=False, new_risk_or_significance_claim=False,
        core_research_goal_complete=False,
        outputs_sha256={n: suite.sha(out/n) for n in ['protocol.json', 'tasks.json', 'methods.json', 'files.json']})
    suite.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/'results/search_radius_reachability/audit_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: main(root, out)
    except Exception:
        suite.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
