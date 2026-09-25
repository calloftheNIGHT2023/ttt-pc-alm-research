"""360 query-free post-seal frontier diagnostic and exact lazy component run."""
from collections import Counter
from pathlib import Path
import time
import traceback
import numpy as np
import budget_reinvestment_suite_v1 as io
import budget_frontier_v1 as model
from test_region_conditioned_credit_v1 import guarded

BASE = 'results/budget_frontier'
DESIGN = 'outputs/ttt-pc-alm-research/360_budget_frontier_protocol_v1.md'
SOURCES = ['budget_frontier_v1.py', 'run_budget_frontier_v1.py', 'audit_budget_frontier_v1.py',
           'test_budget_frontier_v1.py', 'region_conditioned_credit_v1.py', 'test_region_conditioned_credit_v1.py']


def hashes(root):
    return {p: io.sha(root/p) for p in [DESIGN]+['work/experiments/'+n for n in SOURCES]}


def run(root, out):
    begin = time.perf_counter(); frozen = hashes(root)
    pre = io.complete(root/BASE/'preflight_v1')
    assert pre['source_sha256'] == frozen
    source = root/'results/cross_region_credit/development_v1'
    audit = root/'results/cross_region_credit/audit_v1'
    io.complete(source); io.complete(audit)
    protocol = io.read(source/'protocol.json'); tasks = io.read(source/'tasks.json')
    configs = [c for c in protocol['configs'] if not c['reuse']]
    assert len(configs) == 7 and len(tasks) == 32
    io.save(out/'protocol.json', dict(source_sha256=frozen, source_protocol_sha256=io.sha(source/'protocol.json'),
        source_tasks_sha256=io.sha(source/'tasks.json'), geometry_summary_sha256=io.sha(audit/'summary.json'),
        configs=configs, primary='dual_independent', budget=8, posthoc_development=True,
        query_targets_accessed=False, archived_geometry_diagnostic_only=True))
    rows = []; frontiers = []; positive_rows = []; seals = []; totals = {c['name']: Counter() for c in configs}
    for task in tasks:
        seed = task['seed']; src = source/str(seed); dest = out/str(seed); dest.mkdir()
        for name, digest in task['files'].items():
            assert io.sha(src/name) == digest
        inp = io.read(src/'input.json'); x = np.array(inp['x_observed']); v = np.array(inp['v_observed'])
        with np.load(src/'pool.npz', allow_pickle=False) as z:
            regs = z['regions']
        geometry = io.read(audit/(str(seed)+'_geometry.json'))
        labels = [geometry[r.tobytes().hex()]['classification'] for r in regs]
        feasible = np.array([s == 'positive_volume' for s in labels], bool)
        impossible = np.array([s == 'infeasible' for s in labels], bool)
        # The labels never enter model.solve; they are only diagnostic assertions.
        for cfg in protocol['configs']:
            name = cfg['name']; meta = io.read(src/(name+'.json'))
            with np.load(src/(name+'.npz'), allow_pickle=False) as z:
                first = z['first_step']; disabled = z['disabled']
            assert not disabled.any()
            cuts = first > 0
            assert not np.any(cuts & ~impossible)
            for budget in sorted({1, 4, 8, 16, max(1, len(regs))}):
                selected = model.selected(first, budget)
                end = int(selected[-1])+1 if len(selected) == budget else len(regs)
                ideal = np.flatnonzero(~impossible)[:budget]
                counts = np.cumsum(cuts)
                predicted = [j for j in range(len(regs)) if not cuts[j] and j+1-int(counts[j]) <= budget]
                assert selected.tolist() == predicted
                effective = [int(j) for j in selected if feasible[j]]
                frontier = dict(seed=seed, method=name, budget=budget, candidates=len(regs), cutoff=end,
                    selected=selected.tolist(), selected_positive=effective,
                    positive_total=int(feasible.sum()), ideal_positive=[int(j) for j in ideal if feasible[j]],
                    cuts_total=int(cuts.sum()), cuts_prefix=int(cuts[:end].sum()), cuts_tail=int(cuts[end:].sum()))
                if not cfg['reuse']:
                    work = np.where(cuts, first, cfg['steps'])
                    assert int(work.sum()) == meta['local_response_pairs']
                    frontier.update(full_response_pairs=int(work.sum()), prefix_response_pairs=int(work[:end].sum()))
                frontiers.append(frontier)
                if budget == 8:
                    for j in np.flatnonzero(feasible):
                        required = max(0, int(j)+1-budget); obtained = int(counts[j])
                        positive_rows.append(dict(seed=seed, method=name, index=int(j), rank=int(j)+1,
                            required_prefix_cuts=required, actual_prefix_cuts=obtained,
                            deficit=max(0, required-obtained), selected=int(j) in effective))
        taskrows = []
        for ci in np.random.default_rng(np.random.SeedSequence([360929, seed])).permutation(len(configs)):
            cfg = configs[int(ci)]; name = cfg['name']; credit = np.array(inp['credits'][cfg['channel']])
            with guarded():
                arrays, meta = model.solve(x, v, regs, credit, steps=cfg['steps'], budget=8)
            with np.load(src/(name+'.npz'), allow_pickle=False) as reference:
                expected = model.selected(reference['first_step'], 8)
                assert np.array_equal(arrays['selected_indices'], expected)
                end = int(expected[-1])+1 if len(expected) == 8 else len(regs)
                assert meta['processed'] == end
                for key in ['first_step', 'proof_credit', 'final_credit', 'disabled']:
                    assert arrays[key].dtype == reference[key].dtype
                    assert arrays[key].shape == reference[key][:end].shape
                    assert arrays[key].tobytes() == reference[key][:end].tobytes()
                work = np.where(reference['first_step'] > 0, reference['first_step'], cfg['steps'])
                assert meta['response_pairs'] == int(work[:end].sum())
            np.savez_compressed(dest/(name+'.npz'), **arrays); io.save(dest/(name+'.json'), meta)
            row = dict(seed=seed, method=name, order=len(taskrows), processed=end, candidates=len(regs),
                response_pairs=meta['response_pairs'], full_response_pairs=int(work.sum()),
                batches=len(meta['calls']), maximum_batch=meta['maximum_batch'], seconds=meta['total_seconds'],
                selected_indices=expected.tolist(), files={ext:io.sha(dest/(name+ext)) for ext in ['.npz','.json']})
            taskrows.append(row); rows.append(row)
            for k in ['processed', 'candidates', 'response_pairs', 'full_response_pairs', 'batches', 'seconds']:
                totals[name][k] += row[k]
        io.save(dest/'commit.json', dict(seed=seed, rows=taskrows))
        seals.append(dict(seed=seed, source_files=task['files'], geometry_sha256=io.sha(audit/(str(seed)+'_geometry.json')),
                          commit_sha256=io.sha(dest/'commit.json')))
        print(dict(tasks=len(seals), calls=len(rows), seconds=time.perf_counter()-begin), flush=True)
    for name, value in [('rows.json', rows), ('frontiers.json', frontiers), ('positive_frontiers.json', positive_rows), ('input_seals.json', seals)]:
        io.save(out/name, value)
    assert hashes(root) == frozen
    result = dict(passed=True, calls=len(rows), tasks=len(tasks), frontier_rows=len(frontiers),
        positive_frontier_rows=len(positive_rows), seconds=time.perf_counter()-begin,
        totals={k:dict(v) for k,v in totals.items()}, query_targets_accessed=False, query_risk_evaluated=False,
        complete_online_fits=False, core_research_goal_complete=False,
        outputs_sha256={n:io.sha(out/n) for n in ['protocol.json','rows.json','frontiers.json','positive_frontiers.json','input_seals.json']})
    io.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/BASE/'development_v1'; out.mkdir(parents=True, exist_ok=False)
    try:
        run(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
