"""290: OLD64 state-matched one-step dual erasure, never query evaluation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
import math
from pathlib import Path
import time

import numpy as np
import cold_stagnation_switch as cold
import local_dual_jump as jump
from diagnose_gradient_flat_split_states_v1 import read, save, sha, mode_list


def selftests():
    rng = np.random.default_rng(290193)
    strict = 0
    witness = None
    for _ in range(256):
        s = [F(int(rng.integers(0, 21)), 20), F(int(rng.integers(-2, 3)), 20),
             F(int(rng.integers(-2, 3)), 20), F(int(rng.integers(0, 21)), 20),
             F(int(rng.integers(0, 21)), 20)]
        u = [F(int(rng.integers(-10, 11)), 10) for _ in range(2)]
        zero = jump.values(s, F(0), True, [F(0), F(0)])
        dual = jump.values(s, F(0), True, u)
        a = min(zero['branches'], key=lambda r: (r['energy'], r['k']))
        b = min(dual['branches'], key=lambda r: (r['energy'], r['k']))
        def residual(z):
            return [z-jump.exact.g(s[0]+s[1]), s[4]-jump.exact.g(z+s[2])]
        def energy(z, uu):
            return sum((r+t)**2 for r, t in zip(residual(z), uu))+jump.exact.TRUST*(z-s[3])**2
        for z in [F(0), F(1, 2), F(1), a['z'], b['z']]:
            assert energy(z, u)-energy(z, [F(0), F(0)]) == 2*sum(t*r for t, r in zip(u, residual(z)))+sum(t*t for t in u)
        gap = energy(b['z'], [F(0), F(0)])-energy(a['z'], [F(0), F(0)])
        work = -2*sum(t*(r2-r1) for t, r1, r2 in zip(u, residual(a['z']), residual(b['z'])))
        assert 0 <= gap <= work
        if a['k'] != b['k'] and gap > 0:
            strict += 1
            if witness is None:
                witness = dict(s=[str(t) for t in s], u=[str(t) for t in u],
                               zero_branch=a['k'], dual_branch=b['k'], zero_energy_gap=str(gap), dual_work=str(work))
    assert strict > 0
    x = np.array([.13, .31, .61, .89])
    v = np.array([.17, .38, .74, .83])
    b = np.array([[.03, -.02, .04, -.05], [-.08, .07, -.01, .02]])
    a, n = cold.Local(b, x, v, 'alm'), cold.Local(b, x, v, 'nodual')
    a.step()
    n.step()
    assert a.b.tobytes() == n.b.tobytes() and a.h.tobytes() == n.h.tobytes()
    return dict(passed=True, rational_cases=256, exact_energy_identities=1280,
                strict_conditional_branch_changes=strict, rational_witness=witness,
                zero_dual_one_step_identity=True, query_targets_accessed=False)


def one_step(b, h, u, best, x, v, method='alm'):
    state = cold.Local(b, x, v, method)
    state.h = h.copy()
    state.u = u.copy()
    state.best = best.copy()
    state.errors, state.moves = cold.base.score(best, x, v, np.zeros(4))
    state.step()
    return state


def run(root, out):
    start = time.perf_counter()
    save(out/'selftests.json', selftests())
    diag = root/'results/gradient_flat_split_states/development_v1'
    mass = root/'results/gradient_flat_split_states/branch_mass_v1'
    for folder in [diag, mass]:
        summary = read(folder/'summary.json')
        assert summary['passed'] and summary['tasks'] == 64
        assert summary['protocol_sha256'] == sha(folder/'protocol.json')
        for name, digest in summary['outputs_sha256'].items():
            assert sha(folder/name) == digest
    parents = [diag/'summary.json', mass/'summary.json', root/'results/round_287_audit_v6.json',
               root/'outputs/ttt-pc-alm-research/290_state_matched_dual_intervention_protocol.md']
    seal = read(parents[2])
    assert seal['passed']
    for name, digest in seal['source_sha256'].items():
        assert sha(root/'work/experiments'/name) == digest, name
    input_hashes = read(diag/'input_hashes.json')
    reference_hashes = read(mass/'reference_hashes.json')
    tasks = read(diag/'tasks.json')
    assert [r['seed'] for r in tasks] == list(range(5910000, 5910064))
    ref = root/'results/confirmation_conditional_risk/reference'
    refrows = {r['seed']: r for r in read(ref/'coverage.json')}
    raw = root/'results/certificate_activity_attribution/development'
    rows = {r['seed']: r for r in read(raw/'rows.json') if r['method'] == 'credit_control_probe33'}
    save(out/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), seeds=[r['seed'] for r in tasks],
         source_sha256=sha(Path(__file__)), frozen_dependency_seal_sha256=sha(parents[2]),
         parent_sha256={str(p.relative_to(root)).replace('\\', '/'): sha(p) for p in parents},
         observed_information='Saved ALM b/h/u/best and support x/v; reference only labels outputs after a step',
         query_arrays_decoded=False, query_targets_accessed=False, state_matched_shadow_not_standalone_solver=True,
         stage='old64_exploratory', core_research_goal_complete=False))
    global_counts = Counter()
    results, output_hashes = [], {}
    labels = ['actual_new', 'absent_same_origin_zero', 'absent_entire_round_zero',
              'absent_zero_up_to_discovery', 'absent_zero_entire_shadow_union',
              'actual_only_vs_bp', 'only_vs_bp_absent_entire_round_zero',
              'only_vs_bp_absent_entire_shadow_union']
    for task in tasks:
        seed = task['seed']
        path = raw/rows[seed]['file']
        relative = str(path.relative_to(root)).replace('\\', '/')
        assert sha(path) == rows[seed]['sha256'] == input_hashes[relative]
        rp = ref/refrows[seed]['file']
        assert sha(rp) == refrows[seed]['sha256'] == reference_hashes[str(rp.relative_to(root)).replace('\\', '/')]
        reference = read(rp)['reference']
        volumes = {r['pattern']: r['volume'] for r in reference['positive_regions']}
        assert reference['enumeration_completed'] and all(w > 0 for w in volumes.values())
        total = math.fsum(volumes.values())
        with np.load(path, allow_pickle=False) as z:
            x, v = z['x_observed'], z['v_observed']
            arrays = {name: z[name].copy() for name in z.files if name.startswith(('prefix_', 'anchor_'))}
            assert not z['effective_initial_u'].any()
        seen = set(task['positive_modes']['archive'])
        shadow_seen = seen.copy()
        novel_events, bundles = [], {}
        counts = Counter()
        for phase, rounds, origins in [('prefix', 32, list(range(33))), ('anchor', 32, [0])]:
            b, h, u, best = [arrays[phase+'_'+key] for key in ['b', 'h', 'u', 'best']]
            assert b.shape == (33, len(origins), 4) and h.shape == u.shape == (33, 4, len(origins), 4)
            shadow_b, shadow_h, db, dh, du = [], [], [], [], []
            for t in range(rounds):
                actual = one_step(b[t], h[t], u[t], best[t], x, v)
                for key, old in [('b', b[t+1]), ('h', h[t+1]), ('u', u[t+1])]:
                    assert getattr(actual, key).tobytes() == old.tobytes(), (seed, phase, t, key)
                shadow = one_step(b[t], h[t], np.zeros_like(u[t]), best[t], x, v)
                nodual = one_step(b[t], h[t], np.zeros_like(u[t]), best[t], x, v, 'nodual')
                assert shadow.b.tobytes() == nodual.b.tobytes() and shadow.h.tobytes() == nodual.h.tobytes()
                if not u[t].any():
                    assert actual.b.tobytes() == shadow.b.tobytes() and actual.h.tobytes() == shadow.h.tobytes()
                    counts['zero_dual_equal_transitions'] += len(origins)
                actual_modes, zero_modes = mode_list(actual.b, x), mode_list(shadow.b, x)
                pos = set(actual_modes) & volumes.keys()
                zero_pos = set(zero_modes) & volumes.keys()
                shadow_before = shadow_seen.copy()
                shadow_seen |= zero_pos
                for key in sorted(pos-seen):
                    who = [j for j, mode in enumerate(actual_modes) if mode == key]
                    novel_events.append(dict(phase=phase, update=t+1, absolute_update=t+1+(32 if phase=='anchor' else 0),
                        pattern=key, origins=[origins[j] for j in who], mass=volumes[key]/total,
                        absent_same_origin_zero=all(zero_modes[j] != key for j in who),
                        absent_entire_round_zero=key not in zero_pos,
                        absent_zero_up_to_discovery=key not in shadow_seen,
                        already_in_earlier_shadow=key in shadow_before,
                        original_bp_did_not_find=key not in task['positive_modes']['bp']))
                seen |= pos
                bd = np.max(abs(actual.b-shadow.b), axis=1)
                hd = np.max(abs(actual.h-shadow.h), axis=(0, 2))
                un = np.max(abs(u[t]), axis=(0, 2))
                counts['transitions'] += len(origins)
                counts['b_bitwise_changed'] += sum(a.tobytes() != s.tobytes() for a, s in zip(actual.b, shadow.b))
                counts['mode_changed'] += sum(a != s for a, s in zip(actual_modes, zero_modes))
                counts['actual_positive_mode'] += sum(a in volumes for a in actual_modes)
                counts['shadow_positive_mode'] += sum(a in volumes for a in zero_modes)
                shadow_b.append(shadow.b.copy())
                shadow_h.append(shadow.h.copy())
                db.append(bd); dh.append(hd); du.append(un)
            for key, value in [('shadow_b', shadow_b), ('shadow_h', shadow_h), ('b_max_difference', db),
                               ('h_max_difference', dh), ('input_u_max', du)]:
                bundles[phase+'_'+key] = np.array(value)
        assert seen == set(task['positive_modes']['alm'])
        assert {e['pattern'] for e in novel_events} == set(task['sets']['alm_new_vs_archive'])
        sets = {name: set() for name in labels}
        for e in novel_events:
            key = e['pattern']
            sets['actual_new'].add(key)
            for label in ['absent_same_origin_zero', 'absent_entire_round_zero', 'absent_zero_up_to_discovery']:
                if e[label]: sets[label].add(key)
            e['absent_zero_entire_shadow_union'] = key not in shadow_seen
            if key not in shadow_seen: sets['absent_zero_entire_shadow_union'].add(key)
            if e['original_bp_did_not_find']:
                sets['actual_only_vs_bp'].add(key)
                if e['absent_entire_round_zero']: sets['only_vs_bp_absent_entire_round_zero'].add(key)
                if key not in shadow_seen: sets['only_vs_bp_absent_entire_shadow_union'].add(key)
        assert sets['actual_only_vs_bp'] == set(task['sets']['alm_only_vs_bp'])
        counts['novel_positive_modes'] = len(novel_events)
        global_counts.update(counts)
        name = f'{seed}.npz'
        with (out/name).open('xb') as stream:
            np.savez_compressed(stream, **bundles)
        output_hashes[name] = sha(out/name)
        results.append(dict(seed=seed, counts=dict(counts), source_sha256=rows[seed]['sha256'],
            reference_sha256=refrows[seed]['sha256'], shadow_arrays_file=name, shadow_arrays_sha256=output_hashes[name],
            novel_events=novel_events, sets={k: sorted(vv) for k, vv in sets.items()},
            numerical_posterior_mass={k: math.fsum(volumes[s] for s in sorted(vv))/total for k, vv in sets.items()}))
        print(dict(seed=seed, tasks=len(results), transitions=global_counts['transitions'], seconds=time.perf_counter()-start), flush=True)
    assert global_counts['transitions'] == 69632
    save(out/'tasks.json', results)
    for name in ['selftests.json', 'protocol.json', 'tasks.json']:
        output_hashes[name] = sha(out/name)
    aggregate = {name: dict(task_modes=sum(len(r['sets'][name]) for r in results),
                   tasks=sum(bool(r['sets'][name]) for r in results),
                   mean_mass=float(np.mean([r['numerical_posterior_mass'][name] for r in results]))) for name in labels}
    summary = dict(passed=True, tasks=64, counts=dict(global_counts), mechanism_sets=aggregate,
                   query_arrays_decoded=False, query_targets_accessed=False, actual_steps_replayed=True,
                   scientific_scope='Exploratory OLD64 one-step interventions; not standalone solver or new query benefit',
                   counterfactual_cost_is_offline_not_free_online=True, core_research_goal_complete=False,
                   seconds=time.perf_counter()-start, outputs_sha256=output_hashes)
    save(out/'summary.json', summary)
    print({k: v for k, v in summary.items() if k != 'outputs_sha256'}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    out = root/'results/state_matched_dual_intervention/development_v1'
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(root, out)
    except BaseException as exc:
        save(out/'failure.json', dict(error_type=type(exc).__name__, message=str(exc), no_automatic_retry=True))
        raise


if __name__ == '__main__':
    main()
