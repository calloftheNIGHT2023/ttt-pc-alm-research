"""355 support-only common-pool development; seal before geometry audit."""
from pathlib import Path
import time
import traceback
import numpy as np
import region_conditioned_credit_v1 as model
import local_region_screen as baseline
import budget_reinvestment_suite_v1 as io
from test_region_conditioned_credit_v1 import guarded, hashes, DESIGN
from posterior_confirmation_pipeline import discovery_box

CHANNELS = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']


def run(root, out):
    begin = time.perf_counter(); pre = root/'results/region_conditioned_credit/preflight_v1/summary.json'
    passed = io.read(pre); frozen = hashes(root)
    assert passed['passed'] and frozen == passed['source_sha256']
    source = root/'results/unvisited_language/census_v1'
    tasks = io.read(source/'tasks.json'); assert len(tasks) == 32
    protocol = dict(source_sha256=frozen, preflight_sha256=io.sha(pre), delta=model.DELTA,
        prefixes=list(model.PREFIXES), seeds=[t['seed'] for t in tasks], channels=CHANNELS,
        common_pool='Union of six frozen 350 proposal sets', primary='dual',
        source_tasks_sha256=io.sha(source/'tasks.json'), query_targets_accessed=False,
        geometry_accessed=False, saved_credit_cost_excluded=True, end_to_end_benchmark=False)
    io.save(out/'protocol.json', protocol); records = []; totals = {ch: dict(rejected=0, pairs=0, seconds=0.) for ch in CHANNELS}
    with discovery_box(.12):
        for task in tasks:
            seed = task['seed']; directory = out/str(seed); directory.mkdir()
            path = source/str(seed)/'calls.json'; assert io.sha(path) == task['files']['calls.json']
            # This old file also has development labels. Only proposals and source
            # references are extracted; no label is used or passed to the solver.
            calls = io.read(path)
            modes = sorted({p['mode'] for ch in CHANNELS for p in calls[ch]['new']['proposals']})
            regs = np.array([list(bytes.fromhex(m)) for m in modes], dtype=np.uint8).reshape(-1, 4, 4)
            inputs = {}; credits = {}; shared = None
            for ch in CHANNELS:
                record = calls[ch]; source_file = root/record['source_file']
                assert io.sha(source_file) == record['source_sha256']
                with np.load(source_file, allow_pickle=False) as z:
                    x, v = z['x_observed'], z['v_observed']; credit = z['trigger_credit']; b = z['trigger_b']
                    state = (z['trigger_b'].tobytes(), z['trigger_h'].tobytes(), z['trigger_u'].tobytes())
                if shared is None:
                    shared = (x.tobytes(), v.tobytes(), state)
                assert shared == (x.tobytes(), v.tobytes(), state)
                credits[ch] = credit
                inputs[ch] = dict(file=record['source_file'], sha256=record['source_sha256'])
            io.save(directory/'input.json', dict(seed=seed, modes=modes, x_observed=x.tolist(),
                v_observed=v.tolist(), trigger_b=b.tolist(), credits={ch:a.tolist() for ch,a in credits.items()},
                source_files=inputs, calls_sha256=io.sha(path), geometry_labels_extracted=False))
            files = {'input.json': io.sha(directory/'input.json')}
            for ch in CHANNELS:
                with guarded():
                    arrays, meta = model.solve(x, v, regs, credits[ch])
                np.savez_compressed(directory/(ch+'.npz'), **arrays)
                io.save(directory/(ch+'.json'), meta)
                files[ch+'.npz'] = io.sha(directory/(ch+'.npz')); files[ch+'.json'] = io.sha(directory/(ch+'.json'))
                totals[ch]['rejected'] += len(meta['proofs']); totals[ch]['pairs'] += meta['response_pairs']
                totals[ch]['seconds'] += meta['total_seconds']
            controls = {}; masks = {}
            with guarded():
                for rounds in [5, 20]:
                    tick = time.perf_counter(); masks['c'+str(rounds)] = baseline.contract(x, v, regs, rounds)
                    controls['c'+str(rounds)] = dict(seconds=time.perf_counter()-tick, rejected=int(masks['c'+str(rounds)].sum()))
                tick = time.perf_counter()
                masks['pdhg60'], info = baseline.pdhg(x, v, np.broadcast_to(b, (len(regs), 4)).copy(), regs, 60, 20)
                controls['pdhg60'] = dict(seconds=time.perf_counter()-tick, rejected=int(masks['pdhg60'].sum()), metadata=info)
            np.savez_compressed(directory/'controls.npz', **masks)
            io.save(directory/'controls.json', controls)
            files['controls.npz'] = io.sha(directory/'controls.npz'); files['controls.json'] = io.sha(directory/'controls.json')
            record = dict(seed=seed, regions=len(regs), files=files)
            io.save(directory/'commit.json', record); records.append(record)
            print(dict(tasks=len(records), regions=sum(r['regions'] for r in records),
                       seconds=time.perf_counter()-begin, query_targets_accessed=False), flush=True)
    assert hashes(root) == frozen
    io.save(out/'tasks.json', records)
    summary = dict(passed=True, tasks=len(records), regions=sum(r['regions'] for r in records), totals=totals,
        seconds=time.perf_counter()-begin, query_targets_accessed=False, geometry_accessed=False,
        old_label_file_read_for_proposal_extraction=True, outputs_sealed_before_new_audit=True,
        core_research_goal_complete=False, outputs_sha256={n:io.sha(out/n) for n in ['tasks.json','protocol.json']})
    io.save(out/'summary.json', summary); print(summary, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    out = root/'results/region_conditioned_credit/development_v1'; out.mkdir(parents=True, exist_ok=False)
    try:
        run(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
