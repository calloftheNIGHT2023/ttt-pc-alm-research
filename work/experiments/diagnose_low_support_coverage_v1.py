"""393 sealed finite-start banks, then support-only complete-coverage diagnosis.

This is an old-task numerical diagnostic, not a timed method comparison or
unseen-query risk experiment. No realized query targets enter either phase.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import time
import traceback
import numpy as np
import n24_optimizer_controls_v1 as optimizer
import retired_region_join_v1 as search
import audit_retired_region_join_v1 as auditor
import candidate_set_readout_v1 as readout
from deadline_risk_io_v1 import observations

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/low_support_coverage/development_v1'
DESIGN = ROOT/'outputs/ttt-pc-alm-research/393_low_support_coverage_hypothesis_v1.md'
SEEDS = [5920000, 5920001]
STAGES = [4, 8, 16, 24]
CONFIGS = [(family, steps, restarts) for family, steps in [('adam',240),('gauss_newton',40)] for restarts in [64,256]]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def archive(folder, arrays, metadata):
    tick = time.perf_counter()
    np.savez_compressed(folder/'arrays.npz', **arrays)
    metadata['array_archive_seconds'] = time.perf_counter()-tick
    save(folder/'metadata.json', readout.serializable(metadata))
    return {p.name:sha(p) for p in folder.iterdir() if p.is_file()}


def main():
    start = time.perf_counter()
    checked = ROOT/'results/low_support_coverage/math_preflight_v1/summary.json'
    math_check = read(checked)
    assert math_check['passed'] and math_check['design_sha256'] == sha(DESIGN)
    assert math_check['source_sha256'] == sha(Path(__file__).with_name('test_posterior_coverage_math_v1.py'))
    assert read(ROOT/'results/new_task_deadline_risk/audit_v1/summary.json')['passed']
    source_files = [*sorted((ROOT/'work/experiments').glob('*.py')), DESIGN]
    hashes = {p.relative_to(ROOT).as_posix():sha(p) for p in source_files}
    protocol = dict(seeds=SEEDS, stages=STAGES, configurations=CONFIGS,
        source_sha256=hashes, math_preflight_sha256=sha(checked),
        old_development_tasks=True, query_targets_accessed=False,
        reference='active1024 DFS farthest_x; offline after all optimizer banks sealed',
        max_expanded=65536, max_states=20000, max_seconds=8.,
        particles_per_positive_region=4096, query_points=257,
        volume_exact=False, per_region_integral_exact=False,
        controlled_runtime_comparison=False, may_overlap_gpu_meta_training=True,
        no_method_superiority_claim=True)
    save(OUT/'protocol.json', protocol)
    q = np.linspace(0., 1., 257)
    contexts = {}; banks = []; checks = Counter()
    for seed in SEEDS:
        xx, vv = observations(seed)
        for n in STAGES:
            contexts[seed,n] = (xx[:n].copy(), vv[:n].copy())
            x,v = contexts[seed,n]
            before = (x.tobytes(),v.tobytes(),q.tobytes())
            for family,steps,restarts in CONFIGS:
                name = f'{family}{steps}_r{restarts}'
                arrays, meta = optimizer.fit(x,v,q,seed=seed,family=family,steps=steps,restarts=restarts,readout='point')
                assert before == (x.tobytes(),v.tobytes(),q.tobytes())
                assert np.isfinite(arrays['best_bank']).all()
                assert arrays['best_bank'].shape == (restarts,4)
                # These keys are obtained only from this bank, not a reference.
                keys = sorted({optimizer.old.base.pattern(x,b).astype(np.uint8).tobytes().hex() for b in arrays['best_bank']})
                folder = OUT/f'bank_{seed}_{n}_{name}'; folder.mkdir()
                files = archive(folder, arrays, meta)
                banks.append(dict(seed=seed,n=n,method=name,mode_keys=keys,
                    directory=folder.relative_to(ROOT).as_posix(),files=files))
            print(json.dumps(dict(stage='banks',seed=seed,n=n,sealed_banks=len(banks))),flush=True)
    assert len(banks) == 32
    save(OUT/'sealed_banks.json', banks)
    seal = sha(OUT/'sealed_banks.json')
    save(OUT/'before_reference.json', dict(all_32_banks_sealed=True, sealed_banks_sha256=seal,
        query_targets_accessed=False, reference_results_supplied_to_optimizer=False))
    rows = []; references = []
    for (seed,n),(x,v) in contexts.items():
        folder = OUT/f'reference_{seed}_{n}'; folder.mkdir()
        aa, mm = search.join(x,v,name='regional_active1024',schedule='dfs',order_name='farthest_x',
                            max_expanded=65536,max_states=20000,max_seconds=8.)
        audit_begin = time.perf_counter()
        cc = auditor.audit_case(aa,mm,[],exact=True); checks.update(cc)
        audit_seconds = time.perf_counter()-audit_begin
        files = archive(folder, aa, mm)
        reference = dict(seed=seed,n=n,completed=mm['completed'],search_seconds=mm['total_seconds'],
            audit_seconds=audit_seconds, audit_checks=dict(cc), candidate_count=len(aa['regions']),
            directory=folder.relative_to(ROOT).as_posix(),files=files)
        polys = {}; notes = {}; tick = time.perf_counter()
        if mm['completed']:
            with readout.local.no_bp(True):
                for region in aa['regions']:
                    key = region.tobytes().hex()
                    poly,note = readout.local.explicit_geometry(x,v,key)
                    notes[key] = note
                    if poly is not None:
                        assert note['positive_volume_certified'] and note['numerical_volume_available']
                        polys[key] = poly
        geometry_seconds = time.perf_counter()-tick
        unresolved = [k for k,note in notes.items() if note['classification'] == 'unresolved'
                      or (note['classification'] == 'positive_volume' and k not in polys)]
        resolved = bool(mm['completed'] and not unresolved and polys)
        reference.update(geometry_seconds=geometry_seconds,geometry_notes=readout.serializable(notes),
                         unresolved_modes=unresolved,positive_modes=sorted(polys),fully_resolved=resolved)
        if resolved:
            keys = sorted(polys); means = []; variances = []; points_archive = {}
            tick = time.perf_counter()
            for i,key in enumerate(keys):
                rng = np.random.default_rng(np.random.SeedSequence([393731,seed,n,i,4096]))
                points = readout.shared.geometry.sample(polys[key],4096,rng)
                # Explicit per-particle forward, independently of mixture formula.
                h = np.broadcast_to(q,(len(points),len(q))).copy()
                support = np.broadcast_to(x,(len(points),len(x))).copy()
                for j in range(4):
                    h = optimizer.old.base.g(h+points[:,j,None])
                    support = optimizer.old.base.g(support+points[:,j,None])
                assert np.max(abs(support-v)) <= .001+1e-8
                assert np.all((h>=0)&(h<=1))
                means.append(h.mean(0)); variances.append(h.var(0,ddof=1))
                points_archive[f'points_{i}'] = points
                checks['support_particle_checks'] += len(points)
            means = np.array(means); variances = np.array(variances)
            volumes = np.array([polys[k]['volume'] for k in keys]); weights = volumes/volumes.sum()
            full = weights @ means
            points_archive.update(region_means=means,region_sample_variances=variances,
                volumes=volumes,weights=weights,q=q,full_prediction=full)
            np.savez_compressed(folder/'integration.npz',**points_archive)
            reference.update(integration_seconds=time.perf_counter()-tick,
                integration_sha256=sha(folder/'integration.npz'),numeric_total_volume=float(volumes.sum()),
                numerical_geometry_not_exact_posterior=True)
            key_set = set(keys)
            for bank in [b for b in banks if b['seed']==seed and b['n']==n]:
                for f,digest in bank['files'].items():
                    assert sha(ROOT/bank['directory']/f)==digest
                kept = np.array([key in set(bank['mode_keys']) for key in keys])
                mass = float(weights[kept].sum()); delta = float(weights[~kept].sum())
                row = dict(seed=seed,n=n,method=bank['method'],reference_resolved=True,
                    positive_reference_modes=len(keys), discovered_bank_modes=len(bank['mode_keys']),
                    covered_positive_modes=int(kept.sum()), covered_posterior_mass=mass,
                    omitted_posterior_mass=delta,empty_positive_union=not bool(kept.any()),
                    conditional_bias_proxy=None,missing_mass_disagreement_term=None,
                    finite_integration_identity_error=None,mean_disagreement_squared=None)
                if kept.any():
                    conditional = weights[kept] @ means[kept] / mass
                    bias = float(np.mean((conditional-full)**2))
                    outside = weights[~kept] @ means[~kept] / delta if (~kept).any() else conditional
                    disagreement = float(np.mean((conditional-outside)**2))
                    term = delta**2 * disagreement
                    error = abs(term-bias)
                    assert error <= 2e-14
                    row.update(conditional_bias_proxy=bias,missing_mass_disagreement_term=term,
                        finite_integration_identity_error=error,mean_disagreement_squared=disagreement)
                    checks['numeric_bias_identities'] += 1
                assert set(bank['mode_keys']) & key_set == {keys[i] for i in np.flatnonzero(kept)}
                rows.append(row)
        else:
            rows.extend(dict(seed=seed,n=n,method=b['method'],reference_resolved=False,
                reason='Incomplete search, unresolved geometry, or no positive-volume region')
                for b in banks if b['seed']==seed and b['n']==n)
        references.append(reference)
        save(folder/'reference.json',reference)
        save(OUT/'coverage_rows.json',rows)
        save(OUT/'references.json',references)
        print(json.dumps(dict(stage='reference',seed=seed,n=n,completed=mm['completed'],
            resolved=resolved,positive_modes=len(polys),checked_contexts=len(references))),flush=True)
    assert sha(OUT/'sealed_banks.json') == seal
    for name,digest in hashes.items():
        assert sha(ROOT/name)==digest,name
    assert len(rows)==32 and len(references)==8
    save(OUT/'summary.json',dict(passed=True,old_contexts=8,sealed_optimizer_banks=32,
        resolved_contexts=sum(r['fully_resolved'] for r in references),checks=dict(checks),
        query_targets_accessed=False,independent_confirmation=False,
        numeric_posterior_proxy_not_realized_query_risk=True,goal_complete=False,
        seconds=time.perf_counter()-start,
        outputs_sha256={f:sha(OUT/f) for f in ['protocol.json','sealed_banks.json','before_reference.json','references.json','coverage_rows.json']}))
    print(json.dumps(read(OUT/'summary.json')),flush=True)


if __name__=='__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        main()
    except Exception:
        save(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
        raise
