"""413 real support only; old baseline predictions are analysis references."""
from pathlib import Path
from fractions import Fraction as F
import argparse
import time
import traceback
import numpy as np
import partial_support_certificate_v1 as model
import retired_region_join_v1 as search
import candidate_set_readout_v1 as serialize
import deadline_risk_io_v1 as io

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/partial_support_certificate'
DESIGN='outputs/ttt-pc-alm-research/413_partial_support_certificate_protocol_v1.md'
BASELINES=['prior4096_ridge','cohort_meta_ridge128','official_ttt_native_prior256_32_p4','adam_gn64_256_anytime']


def fraction_save(path,value):io.save(path,serialize.serializable(value))


def interval_metrics(intervals,cheap,baseline):
    projected=[model.mathcore.project(F(float(a)),bb) for a,bb in zip(baseline,intervals)]
    cheap_projected=[model.mathcore.project(F(float(a)),bb) for a,bb in zip(baseline,cheap)]
    changed=sum(a!=F(float(b)) for a,b in zip(projected,baseline))
    return dict(excluded=changed,queries=len(baseline),
        additional_changes_beyond_cheap=sum(a!=b for a,b in zip(projected,cheap_projected)),
        squared_projection_distance=float(sum((F(float(a))-b)**2 for a,b in zip(baseline,projected))/len(baseline)))


def run(out):
    begin=time.perf_counter();gate=io.read(BASE/'preflight_v1/summary.json')
    assert gate['passed'];io.verify_hashes(ROOT,gate['source_sha256'])
    hashes={p.relative_to(ROOT).as_posix():io.sha(p) for p in sorted((ROOT/'work/experiments').glob('*.py'))+[ROOT/DESIGN]}
    old=ROOT/'results/runtime_matched_prefix/development_v1'
    old_summary=io.read(old/'summary.json');assert old_summary['passed'] and old_summary['all_predictions_sealed']
    inputs=io.read(old/'inputs_manifest.json');old_rows=io.read(old/'rows.json')
    assert io.sha(old/'inputs_manifest.json')==old_summary['outputs_sha256']['inputs_manifest.json']
    assert io.sha(old/'rows.json')==old_summary['outputs_sha256']['rows.json']
    lookup={(r['method'],r['n'],r['budget'],r['seed']):r for r in old_rows}
    protocol=dict(source_sha256=hashes,preflight_summary_sha256=io.sha(BASE/'preflight_v1/summary.json'),
        old_prediction_summary_sha256=io.sha(old/'summary.json'),seeds=list(range(5920000,5920004)),
        stages=[4,8,16,24],expected_contexts=16,search_cooperative_seconds=.25,max_modes=4,max_cover_cells=256,
        queries=33,baseline_prediction_archives=BASELINES,baseline_archives_are_online_inputs=False,
        query_targets_accessed=False,hard_end_to_end_deadline=False,matched_budget_comparison=False)
    io.save(out/'protocol.json',protocol);rows=[]
    for seed in protocol['seeds']:
        source=old/'inputs'/f'{seed}.npz';assert io.sha(source)==inputs[source.name]
        data=io.load_arrays(source)
        for n in protocol['stages']:
            x,v,q=data['x'][:n].copy(),data['v'][:n].copy(),data['q'][::8].copy()
            assert len(q)==33
            directory=out/f'{seed}_n{n}';directory.mkdir()
            io.save(out/'current_job.json',dict(seed=seed,n=n))
            call_start=time.perf_counter()
            sa,sm=search.join(x,v,name='regional_active512',schedule='dfs',order_name='farthest_x',max_seconds=.25,max_expanded=65536)
            ra,rm=model.fit(x,v,q,sa,sm,max_modes=4,max_cover_cells=256)
            online_component_seconds=time.perf_counter()-call_start
            np.savez_compressed(directory/'search_arrays.npz',**sa);io.save(directory/'search_metadata.json',sm)
            np.savez_compressed(directory/'certificate_arrays.npz',**ra);fraction_save(directory/'certificate_metadata.json',rm)
            baseline_refs={};metrics={}
            for name in BASELINES:
                old_row=lookup[name,n,.5,seed];path=ROOT/old_row['directory']/'outputs.npz'
                assert io.sha(path)==old_row['files']['outputs.npz']
                bp=io.load_arrays(path)['prediction'][::8]
                assert bp.shape==q.shape
                baseline_refs[path.relative_to(ROOT).as_posix()]=old_row['files']['outputs.npz']
                metrics[name]=dict(posterior=interval_metrics(rm['posterior_intervals'],rm['cheap_intervals'],bp),
                    intersection=interval_metrics(rm['intersected_intervals'],rm['cheap_intervals'],bp),
                    cheap=interval_metrics(rm['cheap_intervals'],rm['cheap_intervals'],bp))
            io.save(directory/'baseline_references.json',baseline_refs);io.save(directory/'baseline_metrics.json',metrics)
            row=dict(seed=seed,n=n,directory=directory.relative_to(ROOT).as_posix(),search_completed=sm['completed'],
                full_modes=rm['full_candidate_modes'],processed_modes=rm['processed_modes'],
                upper_over_prior=float(rm['total_upper']/model.PRIOR),covered_fraction=float(rm['covered_fraction']),
                mean_posterior_width=float(sum(b-a for a,b in rm['posterior_intervals'])/len(q)),
                mean_cheap_width=float(sum(b-a for a,b in rm['cheap_intervals'])/len(q)),
                mean_intersection_width=float(sum(b-a for a,b in rm['intersected_intervals'])/len(q)),
                search_seconds=sm['total_seconds'],certificate_seconds=rm['total_seconds'],
                online_component_seconds=online_component_seconds,baseline_metrics=metrics,
                files={p.name:io.sha(p) for p in directory.iterdir() if p.is_file()})
            rows.append(row);io.save(out/'rows_partial.json',rows)
            print({k:row[k] for k in ['seed','n','search_completed','full_modes','processed_modes','covered_fraction','mean_posterior_width','mean_cheap_width','mean_intersection_width','certificate_seconds']},flush=True)
    io.verify_hashes(ROOT,hashes);io.save(out/'rows.json',rows)
    summary=dict(passed=True,contexts=len(rows),query_targets_accessed=False,baseline_generation_cost_included=False,
        matched_budget_comparison=False,independent_confirmation=False,seconds=time.perf_counter()-begin,
        outputs_sha256={f:io.sha(out/f) for f in ['protocol.json','rows.json']})
    io.save(out/'summary.json',summary);print(summary,flush=True)


def audit(out):
    import audit_retired_region_join_v1 as checker
    source=BASE/'development_v1';ss=io.read(source/'summary.json');assert ss['passed']
    for file,digest in ss['outputs_sha256'].items():assert io.sha(source/file)==digest
    protocol=io.read(source/'protocol.json');io.verify_hashes(ROOT,protocol['source_sha256'])
    counters=dict(searches=0,cover_bounds=0,geometry_certificates=0,cones=0,moment_intervals=0,baseline_metrics=0)
    begin=time.perf_counter()
    for row in io.read(source/'rows.json'):
        directory=ROOT/row['directory']
        for name,digest in row['files'].items():assert io.sha(directory/name)==digest
        sa=io.load_arrays(directory/'search_arrays.npz');sm=io.read(directory/'search_metadata.json')
        checker.audit_case(sa,sm,[],exact=True);counters['searches']+=1
        ra=io.load_arrays(directory/'certificate_arrays.npz');rm=io.read(directory/'certificate_metadata.json')
        upper=model.forest_upper(sa,sm,256)
        expected=serialize.serializable(upper);saved=dict(rm['upper']);expected.pop('seconds');saved.pop('seconds')
        assert expected==saved;counters['cover_bounds']+=1
        mass=F(0);lo=[F(0)]*len(ra['q']);hi=lo.copy()
        for index,record in enumerate(rm['records']):
            _,a,r,_,_=model.classifier.matrices(ra['x'],ra['v'],record['key'])
            counters['geometry_certificates']+=model.classifier.verify_certificates(a,r,record['classification'])
            if 'cone' in record:
                cone=model.mathcore.certified_cones(ra[f'mode_{index}_vertices'],ra[f'mode_{index}_faces'],ra[f'mode_{index}_anchor'],a,r)
                assert serialize.serializable(cone)==record['cone']
                mass+=cone['mass'];counters['cones']+=len(cone['simplices'])
                for j,q in enumerate(ra['q']):
                    for simplex in cone['simplices']:
                        m=model.mathcore.moment_bounds(simplex,F(float(q)),splits=0)
                        lo[j]+=m['lower'];hi[j]+=m['upper']
            expected_bounds=[model.mathcore.posterior_interval(mass,l,h,upper['volume']) for l,h in zip(lo,hi)]
            assert serialize.serializable(expected_bounds)==rm['checkpoints'][index]['posterior_intervals']
            counters['moment_intervals']+=len(ra['q'])
        assert str(mass)==rm['total_inner_mass']
        cheap=model.lipschitz_intervals(ra['x'],ra['v'],ra['q'])
        assert serialize.serializable(cheap)==rm['cheap_intervals']
        pp=[model.mathcore.posterior_interval(mass,l,h,upper['volume']) for l,h in zip(lo,hi)]
        intersection=[model.mathcore.intersect_intervals(p,c) for p,c in zip(pp,cheap)]
        assert serialize.serializable(pp)==rm['posterior_intervals']
        assert serialize.serializable(intersection)==rm['intersected_intervals']
        metrics=io.read(directory/'baseline_metrics.json');refs=io.read(directory/'baseline_references.json')
        for name in BASELINES:
            # Exact directory suffix/prefix match; no query-answer archive.
            matches=[p for p in refs if f"_{name}_b500/" in p];assert len(matches)==1
            path=ROOT/matches[0];assert io.sha(path)==refs[matches[0]]
            bp=io.load_arrays(path)['prediction'][::8]
            actual=dict(posterior=interval_metrics(pp,cheap,bp),intersection=interval_metrics(intersection,cheap,bp),cheap=interval_metrics(cheap,cheap,bp))
            assert actual==metrics[name];counters['baseline_metrics']+=1
        print(dict(stage='certificate_audit',contexts=counters['searches'],seconds=time.perf_counter()-begin),flush=True)
    io.verify_hashes(ROOT,protocol['source_sha256'])
    summary=dict(passed=True,counts=counters,seconds=time.perf_counter()-begin,
        development_summary_sha256=io.sha(source/'summary.json'),query_targets_accessed=False,
        audit_source_sha256=io.sha(Path(__file__)),scope='independent search replay; exact certificate/moment recomputation; not a risk or matched-runtime claim')
    io.save(out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','audit']);args=parser.parse_args()
    out=BASE/('development_v1' if args.action=='run' else 'audit_v1');out.mkdir(parents=True,exist_ok=False)
    try:globals()[args.action](out)
    except Exception:
        io.save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
