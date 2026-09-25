"""355 independent Fraction proof and pre-existing exact geometry audit."""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time
import numpy as np
import branch_image_chain_v1 as independent
import budget_reinvestment_suite_v1 as io
from certified_branch_solver import exact_certificate
from audit_stasis_escape_geometry_v1 import inequalities
from audit_post_escape_geometry_v1 import certificates
from posterior_confirmation_pipeline import discovery_box
from run_region_conditioned_credit_v1 import CHANNELS


def run(root, out):
    begin = time.perf_counter(); source = root/'results/region_conditioned_credit/development_v1'
    summary = io.complete(source); tasks = io.read(source/'tasks.json')
    counts = Counter(); labels = Counter(); rows = []; combined = {}
    totals = {ch:Counter() for ch in CHANNELS+['c5','c20','pdhg60']}
    old = root/'results/unvisited_language/census_v1'
    previous_tasks = {r['seed']:r for r in io.read(old/'tasks.json')}
    with discovery_box(.12):
        for task in tasks:
            seed = task['seed']; directory = source/str(seed)
            for name, digest in task['files'].items(): assert io.sha(directory/name) == digest
            inp = io.read(directory/'input.json'); x = np.array(inp['x_observed']); v = np.array(inp['v_observed'])
            regs = np.array([list(bytes.fromhex(m)) for m in inp['modes']], np.uint8).reshape(-1,4,4)
            geometry = old/str(seed)/'geometry.json'
            assert io.sha(geometry) == previous_tasks[seed]['files']['geometry.json']
            geo = io.read(geometry); infeasible = np.array([geo[m]['classification']=='infeasible' for m in inp['modes']])
            for mode in inp['modes']:
                aa, rhs = inequalities(x, v, mode); counts['geometry_certificates'] += certificates(aa, rhs, geo[mode])
                labels[geo[mode]['classification']] += 1
            masks = {}; step_arrays = {}
            for ch in CHANNELS:
                meta = io.read(directory/(ch+'.json'))
                with np.load(directory/(ch+'.npz'),allow_pickle=False) as z:
                    first = z['first_step']; a = z['proof_credit']
                mask = first > 0; assert int(mask.sum()) == len(meta['proofs'])
                seen = set()
                for proof in meta['proofs']:
                    i = proof['index']; assert i not in seen; seen.add(i)
                    assert proof['mode'] == inp['modes'][i] and first[i] == proof['step']
                    value = independent.fixed_value(x, v, a[i], regs[i])
                    if proof['structural_empty']:
                        assert value is None and proof['lower'] is None
                    else:
                        assert value == F(proof['lower']) > 0
                    assert infeasible[i]; counts['independent_fraction_proofs'] += 1
                assert seen == set(np.flatnonzero(mask))
                assert not np.any(mask & ~infeasible)
                masks[ch] = mask; step_arrays[ch] = first
                totals[ch]['rejected'] += int(mask.sum()); totals[ch]['seconds'] += meta['total_seconds']
                totals[ch]['response_pairs'] += meta['response_pairs']; totals[ch]['exact_calls'] += meta['exact_calls']
                totals[ch]['maximum_named_array_bytes'] = max(totals[ch]['maximum_named_array_bytes'], meta['named_array_bytes_subtotal_max'])
                for p in [1,4,16,64,128]: totals[ch]['prefix_'+str(p)] += int(((first>0)&(first<=p)).sum())
            controlmeta = io.read(directory/'controls.json')
            with np.load(directory/'controls.npz', allow_pickle=False) as z:
                for name in ['c5','c20','pdhg60']:
                    masks[name] = z[name]; assert not np.any(masks[name] & ~infeasible)
                    assert int(masks[name].sum()) == controlmeta[name]['rejected']
                    totals[name]['rejected'] += int(masks[name].sum()); totals[name]['seconds'] += controlmeta[name]['seconds']
            for index, proof in controlmeta['pdhg60']['metadata']['certificates'].items():
                i = int(index); check = exact_certificate(x, v, regs[i], np.array(proof['p']), np.array(proof['a']))
                assert check['positive']; counts['independent_pdhg_proofs'] += 1
            for ch in CHANNELS:
                totals[ch]['beyond_c20'] += int((masks[ch]&~masks['c20']).sum())
                totals[ch]['beyond_zero'] += int((masks[ch]&~masks['zero']).sum())
                totals[ch]['missed_zero'] += int((~masks[ch]&masks['zero']).sum())
            row = dict(seed=seed, regions=len(regs), classifications=dict(Counter(geo[m]['classification'] for m in inp['modes'])),
                rejected={ch:int(mask.sum()) for ch,mask in masks.items()},
                contrasts={a:{b:int((masks[a]&~masks[b]).sum()) for b in masks} for a in masks})
            rows.append(row)
            for ch, mask in masks.items(): combined.setdefault(ch,[]).extend(mask.tolist())
    combined = {ch:np.array(mask) for ch,mask in combined.items()}
    contrasts = {a:{b:int((combined[a]&~combined[b]).sum()) for b in combined} for a in combined}
    io.save(out/'rows.json', rows)
    result = dict(passed=True, tasks=32, regions=summary['regions'], counts=dict(counts), classifications=dict(labels),
        totals={ch:dict(t) for ch,t in totals.items()}, contrasts=contrasts, seconds=time.perf_counter()-begin,
        source_summary_sha256=io.sha(source/'summary.json'), outputs_sha256={'rows.json':io.sha(out/'rows.json')},
        query_targets_accessed=False, query_risk_evaluated=False, independent_task_gain_established=False)
    io.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root=Path(__file__).resolve().parents[2]
    out=root/'results/region_conditioned_credit/audit_v1';out.mkdir(parents=True,exist_ok=False)
    run(root,out)
