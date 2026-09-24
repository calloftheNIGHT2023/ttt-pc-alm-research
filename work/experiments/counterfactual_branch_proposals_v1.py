"""291 OLD64 support-only proposal unions; no new fitting or query results."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import time
import numpy as np
from diagnose_gradient_flat_split_states_v1 import read, save, sha, mode_list
from audit_local_dual_jump_modes import modes

RULES = ['all_steps', 'changed_forward_mode', 'first_global_forward_mode']


def select_rule(rule, before, after, seen):
    if rule == 'all_steps':
        return list(range(len(after)))
    if rule == 'changed_forward_mode':
        return [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert rule == 'first_global_forward_mode'
    selected, batch_seen = [], set()
    for i, key in enumerate(after):
        if key not in seen and key not in batch_seen:
            selected.append(i)
            batch_seen.add(key)
    return selected


def selftests():
    before = ['a', 'a', 'a', 'b']
    after = ['b', 'c', 'c', 'a']
    known = {'a', 'b'}
    assert select_rule('all_steps', before, after, known) == [0, 1, 2, 3]
    assert select_rule('changed_forward_mode', before, after, known) == [0, 1, 2, 3]
    assert select_rule('first_global_forward_mode', before, after, known) == [1]
    assert known == {'a', 'b'}
    assert select_rule('changed_forward_mode', before, before, known) == []
    assert select_rule('first_global_forward_mode', before, after, known|{'c'}) == []
    return dict(passed=True, selection_cases=5, no_reference_mass_argument=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    root = parser.parse_args().project.resolve()
    out = root/'results/counterfactual_branch_proposals/development_v1'
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    save(out/'selftests.json', selftests())
    parent = root/'results/state_matched_dual_intervention/development_v2'
    ss, wrapper = read(parent/'summary.json'), read(parent/'wrapper_complete.json')
    assert ss['passed'] and ss['tasks'] == 64 and wrapper['passed']
    assert wrapper['summary_sha256'] == sha(parent/'summary.json')
    assert wrapper['scope_correction_sha256'] == sha(parent/'scope_correction.json')
    for name, digest in ss['outputs_sha256'].items():
        assert sha(parent/name) == digest
    diagnosis = root/'results/gradient_flat_split_states/development_v1'
    mass = root/'results/gradient_flat_split_states/branch_mass_v1'
    ds, ms = read(diagnosis/'summary.json'), read(mass/'summary.json')
    assert ds['passed'] and ms['passed']
    for folder, summary in [(diagnosis, ds), (mass, ms)]:
        for name, digest in summary['outputs_sha256'].items():
            assert sha(folder/name) == digest
    originals = {r['seed']: r for r in read(diagnosis/'tasks.json')}
    inputs = read(diagnosis/'input_hashes.json')
    refhash = read(mass/'reference_hashes.json')
    ref = root/'results/confirmation_conditional_risk/reference'
    refrows = {r['seed']: r for r in read(ref/'coverage.json')}
    raw = root/'results/certificate_activity_attribution/development'
    oldrows = {r['seed']: r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    save(out/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(Path(__file__)), rules=RULES, fixed_seeds=list(range(5910000, 5910064)),
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/291_counterfactual_branch_proposal_protocol.md'),
        parent_summary_sha256=sha(parent/'summary.json'), parent_wrapper_sha256=sha(parent/'wrapper_complete.json'),
        scope='OLD64 candidate design; shadow proposals remain chargeable; no query or timing comparison',
        query_arrays_decoded=False, query_targets_accessed=False, core_research_goal_complete=False))
    records = []
    for seed in range(5910000, 5910064):
        old = originals[seed]
        path = raw/oldrows[seed]['file']
        assert sha(path) == oldrows[seed]['sha256'] == inputs[str(path.relative_to(root)).replace('\\','/')]
        with np.load(path, allow_pickle=False) as z:
            x = z['x_observed']
            trial = z['atomic_trial_b']
            prefix, anchor = z['prefix_b'], z['anchor_b']
        with np.load(parent/f'{seed}.npz', allow_pickle=False) as z:
            shadows = {k: z[k+'_shadow_b'] for k in ['prefix', 'anchor']}
        seen = set(mode_list(np.concatenate([trial, prefix[0]]), x))
        proposal_modes = {rule: set() for rule in RULES}
        locations = {rule: [] for rule in RULES}
        shadow_union = set()
        for phase, b in [('prefix', prefix), ('anchor', anchor)]:
            for t in range(32):
                before, after = mode_list(b[t], x), mode_list(b[t+1], x)
                zero = mode_list(shadows[phase][t], x)
                assert set(zero) == set(modes(x, shadows[phase][t]))
                shadow_union.update(zero)
                for rule in RULES:
                    indices = select_rule(rule, before, after, seen)
                    proposal_modes[rule].update(zero[i] for i in indices)
                    locations[rule].extend([[0 if phase=='prefix' else 1, t+1, i] for i in indices])
                seen.update(after)
        assert seen == set(oldrows[seed]['metadata']['visited_modes'])
        assert proposal_modes['all_steps'] == shadow_union
        # Reference is opened only AFTER all three support-observable selections.
        rp = ref/refrows[seed]['file']
        assert sha(rp) == refrows[seed]['sha256'] == refhash[str(rp.relative_to(root)).replace('\\','/')]
        volumes = {r['pattern']: r['volume'] for r in read(rp)['reference']['positive_regions']}
        total = math.fsum(volumes.values())
        actual = set(old['positive_modes']['alm'])
        bp = set(old['positive_modes']['bp'])
        assert actual == seen & volumes.keys()
        def mass_of(keys):
            return math.fsum(volumes[k] for k in sorted(keys))/total
        actual_mass = mass_of(actual)
        row = dict(seed=seed, old_alm_coverage=actual_mass, old_bp_coverage=mass_of(bp),
                   original_sha256=oldrows[seed]['sha256'], shadow_sha256=ss['outputs_sha256'][f'{seed}.npz'],
                   reference_sha256=refrows[seed]['sha256'], rules={})
        for rule in RULES:
            positive = proposal_modes[rule] & volumes.keys()
            extra = positive-actual
            union = actual|positive
            union_mass = mass_of(union)
            assert union_mass >= actual_mass-1e-15
            row['rules'][rule] = dict(proposals=len(locations[rule]), locations=locations[rule],
                new_positive_modes=sorted(extra), extra_numerical_mass=mass_of(extra), union_coverage=union_mass,
                original_ideal_tv=1-actual_mass, union_ideal_tv=1-union_mass,
                proposal_unique_modes=len(proposal_modes[rule]), new_lp_modes=len(proposal_modes[rule]-seen),
                union_only_vs_bp_mass=mass_of(union-bp), bp_only_vs_union_mass=mass_of(bp-union))
        records.append(row)
    save(out/'tasks.json', records)
    summary = dict(passed=True, tasks=64, old_alm_coverage=float(np.mean([r['old_alm_coverage'] for r in records])),
        old_bp_coverage=float(np.mean([r['old_bp_coverage'] for r in records])), rules={}, query_targets_accessed=False,
        query_arrays_decoded=False, online_algorithm_executed=False, online_cost_not_measured=True,
        core_research_goal_complete=False, seconds=time.perf_counter()-start)
    for rule in RULES:
        rows = [r['rules'][rule] for r in records]
        summary['rules'][rule] = dict(proposals=sum(r['proposals'] for r in rows),
            mean_proposals=float(np.mean([r['proposals'] for r in rows])), new_positive_modes=sum(len(r['new_positive_modes']) for r in rows),
            tasks_with_new_positive_modes=sum(bool(r['new_positive_modes']) for r in rows),
            mean_extra_numerical_mass=float(np.mean([r['extra_numerical_mass'] for r in rows])),
            mean_union_coverage=float(np.mean([r['union_coverage'] for r in rows])),
            mean_new_lp_modes=float(np.mean([r['new_lp_modes'] for r in rows])))
    summary['outputs_sha256'] = {n:sha(out/n) for n in ['protocol.json','selftests.json','tasks.json']}
    save(out/'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
