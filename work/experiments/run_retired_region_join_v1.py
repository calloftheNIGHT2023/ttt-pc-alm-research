"""380 frozen shared-retirement preflight and 544 full candidate searches."""
import argparse
from collections import Counter
from pathlib import Path
import os
import time
import traceback
import numpy as np
import retired_region_join_v1 as model
import run_regional_prefix_join_v1 as previous
from run_prefix_obstruction_v3 import read, sha, save, complete

BASE = 'results/retired_region_join'
DESIGN = 'outputs/ttt-pc-alm-research/380_certified_retirement_protocol_v1.md'
FILES = ['certified_row_retirement_v1.py', 'retired_region_join_v1.py',
         'run_retired_region_join_v1.py', 'audit_retired_region_join_v1.py']
load, same, cases = previous.load, previous.same, previous.cases
LEGACY_NAMES = [name for name, _, _ in model.original.CONFIGS]


def hashes(root):
    h = dict(complete(root/previous.BASE/'preflight_v1')['source_sha256'])
    for path, value in h.items(): assert sha(root/path) == value, path
    for path in [DESIGN]+['work/experiments/'+f for f in FILES]: h[path] = sha(root/path)
    return h


def solver_pair(x, v, regs, family, steps, counts):
    from audit_retired_region_join_v1 import audit_solver
    old, om = model.retirement.ORIGINAL_SOLVE(x, v, regs, family=family, steps=steps)
    new, nm = model.retirement.solve(x, v, regs, family=family, steps=steps)
    keys = ['initial_b', 'initial_z', 'initial_h', 'first_step', 'proof_p', 'proof_a', 'proof_lower', 'proof_kind']
    counts['first_proof_and_initial_arrays'] += same({k: old[k] for k in keys}, {k: new[k] for k in keys})
    unresolved = new['first_step'] < 0
    for key in ['b', 'z', 'h', 'p', 'a']:
        counts['unresolved_terminal_arrays'] += same({key: old['final_'+key][unresolved]}, {key: new['stopped_'+key][unresolved]})
    for key, value in new.items(): assert np.all(np.isfinite(value)), key
    assert om['rounded_checks'] == nm['rounded_checks'] and nm['proposal_rows'] <= om['proposal_rows']
    counts.update(audit_solver(new['first_step'], nm, regs.shape[1], regs.shape[2]))
    counts['exact_solver_cuts'] += previous.previous.prior.verify_proofs(x, v, regs, new)
    permutation = np.random.default_rng(380071).permutation(len(regs)); undo = np.argsort(permutation)
    permuted, pm = model.retirement.solve(x, v, regs[permutation], family=family, steps=steps)
    counts['permutation_arrays'] += same(new, {k: value[undo] for k, value in permuted.items()})
    assert pm['row_steps'] == nm['row_steps'] and pm['proposal_rows'] == nm['proposal_rows']
    return old, om, new, nm


def preflight(root, out):
    from audit_retired_region_join_v1 import audit_case
    start = time.perf_counter(); h = hashes(root); counts = Counter(); records = []
    fixed = previous.previous.prior.cases(root)[:2]
    for ci, case in enumerate(fixed):
        x, v, regs = previous.previous.prior.data(root, case)
        for name, family, steps in model.CONFIGS[1:]:
            old, om, new, nm = solver_pair(x, v, regs, family, steps, counts)
            directory = out/f'frontier_{ci}_{name}'; directory.mkdir()
            np.savez_compressed(directory/'original.npz', **old)
            np.savez_compressed(directory/'retired.npz', **new)
            save(directory/'metadata.json', dict(original=om, retired=nm))
            records.append(dict(case=ci, method=name, certified=int((new['first_step'] >= 0).sum()),
                                row_steps=nm['row_steps'], upper=nm['full_row_steps_upper_bound']))
    # Exercise the empty live-batch path with a test-only certified subset.
    x, v, regs = previous.previous.prior.data(root, fixed[0])
    a, m = model.retirement.ORIGINAL_SOLVE(x, v, regs, family='regional_active', steps=1024)
    subset = regs[a['first_step'] >= 0]; assert len(subset)
    _, _, aa, mm = solver_pair(x, v, subset, 'regional_active', 1024, counts)
    assert np.all(aa['first_step'] >= 0) and mm['checkpoints'][-1]['active_after'] == 0
    counts['all_retired_tests'] += 1
    # No labels are supplied to the solver; this known feasible toy is only an invariant test.
    rng = np.random.default_rng(380171); xx = rng.uniform(.05, .95, 8); bias = rng.uniform(-.1, .1, 4)
    vv = previous.old_solver.screen.base.forward(xx, bias)
    regs = previous.old_solver.screen.base.pattern(xx, bias)[None].astype(np.uint8)
    for family in ['regional_active', 'regional_passive', 'pdhg_cold', 'pdhg_box']:
        _, _, aa, mm = solver_pair(xx, vv, regs, family, 128, counts)
        assert aa['first_step'][0] == -1 and mm['row_steps'] == 128; counts['none_retired_tests'] += 1
    small = previous.old_join.load_cases(root, 'preflight')[0]
    for name, _, _ in model.CONFIGS:
        for schedule, order in model.SETTINGS:
            old, om = model.join(small['x'], small['v'], name=name, schedule=schedule, order_name=order, retire=False, max_seconds=60.)
            new, nm = model.join(small['x'], small['v'], name=name, schedule=schedule, order_name=order, max_seconds=60.)
            assert om['completed'] and nm['completed']; counts['small_search_array_replays'] += same(old, new)
            counts.update(audit_case(new, nm, [], exact=True)); counts['small_search_calls'] += 1
    # One fixed real n24 case, every legacy config, both predeclared settings.
    complete(root/previous.BASE/'development_v1')
    legacy = {(r['seed'], r['method'], r['schedule'], r['ordering']): r for r in read(root/previous.BASE/'development_v1/rows.json')}
    c = cases(root)[0]
    for name in LEGACY_NAMES:
        for schedule, order in model.SETTINGS:
            row = legacy[c['seed'], name, schedule, order]; directory = root/row['directory']
            for f, value in row['files'].items(): assert sha(directory/f) == value
            new, nm = model.join(c['x'], c['v'], name=name, schedule=schedule, order_name=order, max_seconds=60.)
            counts['real_search_array_replays'] += same(load(directory/'arrays.npz'), new)
            counts.update(audit_case(new, nm, [], exact=True)); counts['real_search_calls'] += 1
    save(out/'records.json', records); assert hashes(root) == h
    files = [str(p.relative_to(out)).replace('\\', '/') for p in out.rglob('*') if p.is_file()]
    result = dict(passed=True, checks=dict(counts), source_sha256=h, seconds=time.perf_counter()-start,
                  outputs_sha256={f: sha(out/f) for f in files})
    save(out/'summary.json', result)
    print(dict(stage='preflight_complete', passed=True, checks=dict(counts), seconds=result['seconds']), flush=True)


def run(root, out):
    start = time.perf_counter(); h = hashes(root)
    assert complete(root/BASE/'preflight_v1')['source_sha256'] == h
    cc = cases(root); assert len(cc) == 16
    save(out/'protocol.json', dict(source_sha256=h, cases=[{k: v for k, v in c.items() if k not in ['x', 'v']} for c in cc],
        configs=model.CONFIGS, primary=model.PRIMARY, settings=model.SETTINGS, legacy_names=LEGACY_NAMES,
        query_targets_accessed=False, new_blind_tasks=False, all_audits_after_search_seal=True,
        nominal_coordinate_steps_not_flops=True, all_controls_share_retirement=True))
    rows = []
    for c in cc:
        jobs = [(name, schedule, order) for name, _, _ in model.CONFIGS for schedule, order in model.SETTINGS]
        for j in np.random.default_rng(np.random.SeedSequence([380929, c['seed']])).permutation(len(jobs)):
            name, schedule, order = jobs[int(j)]; x, v = c['x'], c['v']; before = x.tobytes(), v.tobytes()
            a, m = model.join(x, v, name=name, schedule=schedule, order_name=order)
            assert before == (x.tobytes(), v.tobytes())
            directory = out/f"{c['seed']}_{schedule}_{order}_{name}"; directory.mkdir()
            tick = time.perf_counter(); np.savez_compressed(directory/'arrays.npz', **a); disk_seconds = time.perf_counter()-tick
            save(directory/'metadata.json', m)
            row = dict(seed=c['seed'], method=name, schedule=schedule, ordering=order, completed=m['completed'], stop_reason=m['stop_reason'],
                       seconds=m['total_seconds'], expanded=m['expanded'], remaining=len(a['regions']), local_rejected=m['rejected_local'],
                       solver_coordinate_steps=m['solver_coordinate_steps'], solver_coordinate_steps_upper_bound=m['solver_coordinate_steps_upper_bound'],
                       returned_array_bytes=m['returned_array_bytes'], named_bytes=m['live_named_array_bytes_lower_bound'],
                       solver_seconds=m['solver_seconds'], proposal_rows=m['proposal_rows'], rounded_checks=m['rounded_checks'],
                       compression_seconds=disk_seconds, directory=str(directory.relative_to(root)),
                       files={f: sha(directory/f) for f in ['arrays.npz', 'metadata.json']})
            rows.append(row)
            if len(rows) % 17 == 0:
                print(dict(stage='search', calls=len(rows), completed=sum(r['completed'] for r in rows), seconds=time.perf_counter()-start), flush=True)
        save(out/f"sealed_seed_{c['seed']}.json", rows[-34:])
    assert len(rows) == 544 and hashes(root) == h; save(out/'rows.json', rows)
    result = dict(passed=True, calls=len(rows), completed=sum(r['completed'] for r in rows), seconds=time.perf_counter()-start,
                  query_targets_accessed=False, all_methods_sealed_before_external_labels=True, core_research_goal_complete=False,
                  outputs_sha256={f: sha(out/f) for f in ['protocol.json', 'rows.json']})
    save(out/'summary.json', result); print(dict(stage='search_complete', **result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['preflight', 'run'], required=True); args = p.parse_args()
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    root = Path(__file__).resolve().parents[2]; out = root/BASE/('preflight_v1' if args.stage == 'preflight' else 'development_v1')
    out.mkdir(parents=True, exist_ok=False)
    try: globals()[args.stage](root, out)
    except Exception:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
