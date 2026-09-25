"""Frozen evaluator of paired risks: analytical q integration, stratified b MC."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import paired_spline_risk as spline
import enumerate_support_modes as reference
model = reference.memory

PAIRS = [
    ('alm_local_k24', 'alm_none_k24'),
    ('alm_local_k24', 'adam16_bp_k24'),
    ('alm_local_k24', 'adam60_bp_k24'),
    ('alm_local_k24', 'alm_local_full'),
    ('alm_both_c20_k24', 'alm_bp_c20_k24'),
    ('alm_local_full', 'adam16_bp_full'),
    ('alm_local_full', 'adam60_h2_full'),
    ('alm_local_full', 'direct4096_h2_full'),
]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values):
    values = np.asarray(values)
    ids = np.random.default_rng(192823).integers(0, len(values), (20000, len(values)))
    return np.quantile(values[ids].mean(1), [.025, .975]).tolist()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); args = p.parse_args()
    root = args.project; online = root / 'results/budgeted_credit_memory/development'
    refdir = root / 'results/posterior_state_reuse/first_write_reference'
    curvepath = root / 'results/posterior_state_reuse/conditional_risk'
    out = root / 'results/paired_conditional_risk/diagnostic'; out.mkdir(parents=True, exist_ok=True)
    old = json.loads((online / 'protocol.json').read_text()); hashes = old['source_sha256'].copy()
    refprotocol = json.loads((refdir / 'protocol.json').read_text())
    for name, h in refprotocol['source_sha256'].items():
        if name in hashes: assert hashes[name] == h
        hashes[name] = h
    for name, h in hashes.items(): assert sha(Path(__file__).with_name(name)) == h, name
    for name in [Path(__file__).name, 'paired_spline_risk.py', 'compiled_piecewise_readout.py']:
        hashes[name] = sha(Path(__file__).with_name(name))
    assert not (out / 'protocol.json').exists()
    protocol = dict(source_sha256=hashes, online_protocol_sha256=sha(online/'protocol.json'), reference_protocol_sha256=sha(refdir/'protocol.json'),
        pairs=PAIRS, seeds=list(range(old['seed0'], old['seed0']+old['count'])), context=4, batches=8, samples_per_region_per_batch=1024,
        rng_root=428731, verification=spline.verify(), scope='Evaluator only; analytical input integral in floating point, numerical complete-volume posterior and stratified Monte Carlo; 16 old contexts')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    episodes = {(r['seed'], r['method']): r for r in json.loads((online/'episodes.json').read_text()) if r['repetition'] == 0 and r['n_context'] == 4}
    coverage = {r['seed']: r for r in json.loads((refdir/'coverage.json').read_text())}
    names = sorted(set(sum([list(p) for p in PAIRS], []))); rows=[]; audits=[]
    for seed in protocol['seeds']:
        begin = time.perf_counter(); item = coverage[seed]; path = refdir/item['reference_file']; assert sha(path) == item['reference_sha256']
        ref = json.loads(path.read_text()); assert ref['numerical_volume_reference_complete']
        records = sorted(ref['positive_regions'], key=lambda r:r['pattern']); volumes=np.array([r['volume'] for r in records]); weights=volumes/volumes.sum()
        x=np.array(ref['x']); v=np.array(ref['v']); funcs={}; states={}; statehash={}; replay=[]
        rng=np.random.default_rng(seed); truth=rng.uniform(-.12, .12, 4); rng.uniform(0, 1, 24); q=rng.uniform(0, 1, old['queries']); target=spline.base.forward(q, truth)
        for name in names:
            row=episodes[seed, name]; path=online/row['state_file']; assert sha(path)==row['state_sha256']; statehash[name]=row['state_sha256']
            with np.load(path) as z: bank=z['samples'].copy()
            states[name]=bank; funcs[name]=spline.from_bank(bank)
            direct=model.posterior.make_predict(bank)(q); err=float(np.max(np.abs(funcs[name](q)-direct))); assert err<1e-10
            assert float(np.mean((direct-target)**2))==row['raw_query_mse']; replay.append(err)
        diffs=[spline.pair(funcs[a], funcs[b]) for a,b in PAIRS]
        teacherseg=spline.batch_segments(truth[None]); actual=np.array([c-2*spline.integrate_bank(d, teacherseg, 1)[0] for d,c in diffs])
        all_rep=np.zeros((len(PAIRS), protocol['batches'])); mean=np.zeros(len(PAIRS)); var=np.zeros(len(PAIRS)); probability=np.zeros(len(PAIRS)); cells=[]
        for k, record in enumerate(records):
            _,_,g,rhs=model.base.branch_polytope(x, v, np.array(record['representative']))
            poly,note=model.posterior.polytope(g, rhs); assert poly is not None and np.isclose(poly['volume'], record['volume'], rtol=1e-9, atol=1e-30)
            blocks=[]
            for batch in range(protocol['batches']):
                rng=np.random.default_rng(np.random.SeedSequence([protocol['rng_root'], seed, k, batch]))
                bank=model.posterior.sample(poly, protocol['samples_per_region_per_batch'], rng); segments=spline.batch_segments(bank)
                vals=np.array([np.zeros(len(bank)) if not np.any(d.values) and c==0 else c-2*spline.integrate_bank(d, segments, len(bank)) for d,c in diffs])
                all_rep[:, batch]+=weights[k]*vals.mean(1); blocks.append(vals)
            values=np.concatenate(blocks, axis=1); cm=values.mean(1); cv=values.var(1, ddof=1)
            mean+=weights[k]*cm; var+=weights[k]**2*cv/values.shape[1]; probability+=weights[k]*(values<-1e-14).mean(1)
            cells.append(dict(pattern=record['pattern'], weight=float(weights[k]), paired_means=cm.tolist(), paired_sample_variances=cv.tolist()))
        assert np.max(np.abs(mean-all_rep.mean(1)))<1e-13
        audit=json.loads((curvepath/f'audit_{seed}.json').read_text()); cp=curvepath/audit['curve_file']; assert sha(cp)==audit['curve_sha256']
        with np.load(cp) as z:
            grid=z['q']; full=np.einsum('k,rkq->rq', z['weights'], z['region_means'])
            oldgrid=np.array([np.mean([np.trapezoid(d(grid)*(funcs[a](grid)+funcs[b](grid)-2*m), x=grid) for m in full]) for (a,b),(d,c) in zip(PAIRS,diffs)])
        for j,(a,b) in enumerate(PAIRS):
            empirical=episodes[seed,a]['raw_query_mse']-episodes[seed,b]['raw_query_mse']
            rows.append(dict(seed=seed, candidate=a, comparator=b, conditional_delta=float(mean[j]), conditional_mc_se=float(np.sqrt(var[j])),
                mc_normal_ci95=[float(mean[j]-1.96*np.sqrt(var[j])),float(mean[j]+1.96*np.sqrt(var[j]))], replica_conditional_deltas=all_rep[j].tolist(),
                conditional_probability_candidate_better=float(probability[j]), actual_teacher_integrated_delta=float(actual[j]), empirical_first_query_delta=empirical,
                teacher_fluctuation_delta=float(actual[j]-mean[j]), query_sampling_delta=float(empirical-actual[j]), old_grid_conditional_delta=float(oldgrid[j]),
                paired_predictors_bitwise_identical=bool(np.array_equal(states[a], states[b]))))
        audit=dict(seed=seed, source_reference_sha256=item['reference_sha256'], curve_sha256=audit['curve_sha256'], state_sha256=statehash,
            max_compiler_query_error=max(replay), posterior_teachers=len(records)*protocol['batches']*protocol['samples_per_region_per_batch'], cells=cells, elapsed_seconds=time.perf_counter()-begin)
        audits.append(audit); (out/f'audit_{seed}.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        (out/'paired_risks.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(dict(seed=seed, completed=len(audits), total=len(protocol['seeds']), elapsed=audit['elapsed_seconds'])), flush=True)
    summary=[]
    for a,b in PAIRS:
        rr=[r for r in rows if r['candidate']==a and r['comparator']==b]; delta=np.array([r['conditional_delta'] for r in rr])
        summary.append(dict(candidate=a, comparator=b, conditional_mean_delta=float(delta.mean()), descriptive_task_bootstrap_ci95=ci(delta),
            conditional_mean_mc_se=float(np.sqrt(sum(r['conditional_mc_se']**2 for r in rr))/len(rr)),
            actual_teacher_mean_delta=float(np.mean([r['actual_teacher_integrated_delta'] for r in rr])),
            empirical_first_query_mean_delta=float(np.mean([r['empirical_first_query_delta'] for r in rr])),
            mean_teacher_fluctuation=float(np.mean([r['teacher_fluctuation_delta'] for r in rr])), mean_query_sampling_error=float(np.mean([r['query_sampling_delta'] for r in rr])),
            negative_conditional_tasks=int((delta<-1e-14).sum()), zero_conditional_tasks=int((np.abs(delta)<=1e-14).sum()),
            changed_state_seeds=[r['seed'] for r in rr if not r['paired_predictors_bitwise_identical']],
            old_grid_delta=float(np.mean([r['old_grid_conditional_delta'] for r in rr]))))
    result=dict(audit=dict(source_hashes=len(hashes), tasks=len(audits), states=sum(len(a['state_sha256']) for a in audits), posterior_teachers=sum(a['posterior_teachers'] for a in audits),
        max_compiler_query_error=max(a['max_compiler_query_error'] for a in audits)), summary=summary, protocol_sha256=sha(out/'protocol.json'))
    (out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); print(json.dumps(result, indent=2), flush=True)


if __name__=='__main__': main()
