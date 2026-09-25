"""378 preflight and frozen 576-call full-search development experiment."""
import argparse
from collections import Counter
from pathlib import Path
import os
import time
import traceback
import numpy as np
import regional_prefix_join_v1 as model
import regional_alm_v1 as old_solver
import run_regional_alm_v1 as previous
import run_prefix_language_join_v1 as old_join
from run_prefix_obstruction_v3 import read, sha, save, complete

BASE = 'results/regional_prefix_join'
DESIGN = 'outputs/ttt-pc-alm-research/378_regional_join_protocol_v1.md'
FILES = ['regional_alm_explicit_v1.py', 'regional_prefix_join_v1.py',
         'run_regional_prefix_join_v1.py', 'audit_regional_prefix_join_v1.py']


def hashes(root):
    h = dict(read(root/'results/regional_alm/preflight_v1/summary.json')['source_sha256'])
    for p, value in h.items(): assert sha(root/p) == value, p
    for p in [DESIGN]+['work/experiments/'+f for f in FILES]: h[p] = sha(root/p)
    return h


def load(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def same(a, b):
    assert set(a) == set(b)
    for k in a:
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and a[k].tobytes() == b[k].tobytes(), k
    return len(a)


def cases(root):
    return [c for c in old_join.load_cases(root, 'development') if c['group'] == 'context_scaling' and c['n'] == 24]


def preflight(root, out):
    from audit_regional_prefix_join_v1 import audit_case
    start = time.perf_counter(); h = hashes(root); counts = Counter()
    c = previous.prior.cases(root)[0]; x, v, regs = previous.prior.data(root, c)
    for name in old_solver.METHODS:
        steps = 1024 if name.endswith('1024') else 128; family = name[:-len(str(steps))]
        a, am = old_solver.solve(x, v, regs, name)
        b, bm = model.regional.solve(x, v, regs, family=family, steps=steps)
        counts['original_solver_array_replays'] += same(a, b)
        assert am['proposal_rows'] == bm['proposal_rows'] and am['rounded_checks'] == bm['rounded_checks']
    for family in ['pdhg_cold', 'pdhg_box']:
        full, fm = model.regional.solve(x, v, regs, family=family, steps=1024)
        for steps in [256, 512]:
            a, m = model.regional.solve(x, v, regs, family=family, steps=steps)
            assert m['steps'] == steps and m['checkpoints'][-1]['step'] == steps
            expected = np.where((full['first_step'] >= 0) & (full['first_step'] <= steps), full['first_step'], -1)
            assert np.array_equal(expected, a['first_step']); counts['explicit_step_prefix_matches'] += 1
    import scipy.optimize as opt
    import batched_bp_discovery as bp
    with model.regional.guarded():
        with model.regional.guarded():
            for fn in [opt.linprog, opt.minimize, bp.evaluate, bp.refine]:
                try: fn()
                except AssertionError as e:
                    assert 'Global LP/BP' in str(e); counts['nested_guard_rejections'] += 1
                else: raise AssertionError('Guard failed')
    assert model.regional._depth.get() == 0
    small = old_join.load_cases(root, 'preflight')[0]; xx, vv = small['x'], small['v']
    with model.regional.guarded(): legacy, lm = old_join.model.join(xx, vv, ordering='observed', max_seconds=60.)
    states = {}; records = []
    for name, family, steps in model.CONFIGS:
        for schedule in ['bfs', 'dfs']:
            a, m = model.join(xx, vv, name=name, schedule=schedule, max_seconds=60.)
            assert m['completed']
            if family is None: assert np.array_equal(a['regions'], legacy['regions'])
            result = audit_case(a, m, [], exact=True); counts.update(result)
            again, mm = model.join(xx, vv, name=name, schedule=schedule, max_seconds=60.)
            counts['repeat_join_arrays'] += same(a, again)
            states[name, schedule] = {r.tobytes() for r in a['regions']}
            assert m['expanded'] == mm['expanded']
            records.append(dict(method=name, schedule=schedule, completed=m['completed'], expanded=m['expanded'], seconds=m['total_seconds']))
    for name, _, _ in model.CONFIGS:
        # Same per-batch gate can cause different false-positive survivors under DFS;
        # equality is required for the no-extra control, not stronger relaxation filters.
        if name == 'none': assert states[name, 'bfs'] == states[name, 'dfs']
    rng = np.random.default_rng(378071)
    for n in [4, 8]:
        xx = rng.uniform(.05, .95, n); b = rng.uniform(-.1, .1, 4)
        vv = old_solver.screen.base.forward(xx, b)
        key = old_solver.screen.base.pattern(xx, b).astype(np.uint8).tobytes().hex()
        for schedule in ['bfs', 'dfs']:
            for name in ['none', model.PRIMARY]:
                a, m = model.join(xx, vv, name=name, schedule=schedule, max_expanded=256, max_seconds=60.)
                counts.update(audit_case(a, m, [key], exact=True))
                counts['budgeted_known_positive_checks'] += 1
    save(out/'records.json', records)
    assert hashes(root) == h
    result = dict(passed=True, checks=dict(counts), source_sha256=h, seconds=time.perf_counter()-start,
                  outputs_sha256={'records.json': sha(out/'records.json')})
    save(out/'summary.json', result); print(dict(stage='preflight', **result), flush=True)


def run(root, out):
    start = time.perf_counter(); h = hashes(root)
    assert complete(root/BASE/'preflight_v1')['source_sha256'] == h
    cc = cases(root); assert len(cc) == 16
    save(out/'protocol.json', dict(source_sha256=h, cases=[{k: v for k, v in c.items() if k not in ['x', 'v']} for c in cc],
        configs=model.CONFIGS, primary=model.PRIMARY, schedules=['bfs', 'dfs'], orderings=['observed', 'farthest_x'],
        query_targets_accessed=False, new_blind_tasks=False, all_audits_after_search_seal=True))
    rows = []
    for c in cc:
        jobs = [(name, schedule, order) for name, _, _ in model.CONFIGS for schedule in ['bfs', 'dfs'] for order in ['observed', 'farthest_x']]
        for j in np.random.default_rng(np.random.SeedSequence([378929, c['seed']])).permutation(len(jobs)):
            name, schedule, order = jobs[int(j)]; x, v = c['x'], c['v']; before = x.tobytes(), v.tobytes()
            a, m = model.join(x, v, name=name, schedule=schedule, order_name=order)
            assert before == (x.tobytes(), v.tobytes())
            directory = out/f"{c['seed']}_{schedule}_{order}_{name}"; directory.mkdir()
            tick = time.perf_counter(); np.savez_compressed(directory/'arrays.npz', **a); disk_seconds = time.perf_counter()-tick
            save(directory/'metadata.json', m)
            row = dict(seed=c['seed'], method=name, schedule=schedule, ordering=order, completed=m['completed'], stop_reason=m['stop_reason'],
                       seconds=m['total_seconds'], expanded=m['expanded'], remaining=len(a['regions']),
                       local_rejected=m['rejected_local'], solver_coordinate_steps=m['solver_coordinate_steps'],
                       returned_array_bytes=m['returned_array_bytes'], compression_seconds=disk_seconds,
                       directory=str(directory.relative_to(root)), files={f: sha(directory/f) for f in ['arrays.npz', 'metadata.json']})
            rows.append(row)
            if len(rows) % 9 == 0:
                print(dict(stage='search', calls=len(rows), tasks=len({r['seed'] for r in rows}),
                           completed=sum(r['completed'] for r in rows), seconds=time.perf_counter()-start), flush=True)
        save(out/f"sealed_seed_{c['seed']}.json", rows[-36:])
    assert len(rows) == 576 and hashes(root) == h
    save(out/'rows.json', rows)
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
