"""287 v2 path audit: exact true-H geometry replaces raw-float closure.

Default old-task preflight. No query answers or complete posterior oracle.
Offline audit work is not a free online input or a timing measurement.
"""
import argparse,json,os,time
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
import probe_credit_confirmation_suite as suite
import probe_confirmation_path_reference as independent
import shared_mode_readout as shared
import conditioned_mode_geometry as conditioned
from audit_confirmation_geometry_determinants import check as exact_volume
from probe_exact_polytope_certificate import observed_constraints
from probe_exact_geometry_gate import checked_certificate, READOUT_BUDGET
from audit_shared_mode_readout import check_cube
from posterior_confirmation_pipeline import discovery_box
from run_probe_credit_confirmation_v2 import verify_commit,exclusive_json,acquire_lock
from run_multiplier_fixed_point_screen import sha,dump


def pool_coupling(keys, cache, certificates):
    # A bound for the ideal continuous mixture, not an exact finite-RNG claim.
    raw=np.array([cache[k]['volume'] for k in keys]); raw/=raw.sum()
    stored=[F(float(w)) for w in raw]; total=sum(stored); stored=[w/total for w in stored]
    truth=[F(certificates[k]['exact_reference_volume']) for k in keys]; volume=sum(truth)
    truth=[v/volume for v in truth]
    tv=sum(abs(a-b) for a,b in zip(stored,truth))/2
    bound=tv+sum(w*F(certificates[k]['exact_nominal_readout_expectation_error_bound']) for k,w in zip(keys,stored))
    assert bound<=READOUT_BUDGET, ('Full discovered-pool nominal coupling exceeds existing forward budget',float(bound))
    return dict(mode_weight_tv=float(tv),exact_mode_weight_tv=str(tv),
        nominal_readout_error_bound=float(bound),exact_nominal_readout_error_bound=str(bound),
        squared_risk_of_mean_difference_bound=float(2*bound),budget=float(READOUT_BUDGET))


def certificate_gate(base,src):
    testdir=base/'exact_geometry_gate_selftests'
    tested=json.loads((testdir/'summary.json').read_text())
    assert tested['passed'] and tested['tests']==22 and not tested['query_targets_accessed']
    assert sha(testdir/'tests.json')==tested['tests_sha256']
    for name,digest in tested['source_sha256'].items(): assert sha(src/name)==digest
    directory=base/'exact_polytope_certificate_independent_audit'
    audited=json.loads((directory/'summary.json').read_text())
    assert audited['passed'] and not audited['query_targets_accessed']
    assert sha(src/'audit_probe_exact_polytope_certificate.py')==audited['source_sha256']
    cert=base/'exact_polytope_certificate_preflight'
    assert sha(cert/'summary.json')==audited['certificate_summary_sha256']
    assert sha(cert/'witness.json')==audited['witness_sha256']
    return {str(p.relative_to(base)):sha(p) for p in [
        testdir/'summary.json',directory/'summary.json',cert/'summary.json',cert/'witness.json']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--stage',choices=['preflight','confirmation'],default='preflight');args=ap.parse_args();root=args.project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';inp=base/('runner_preflight_v2' if args.stage=='preflight' else 'predictions')
    out=base/('path_audit_preflight_v3' if args.stage=='preflight' else 'path_audit_v3');prior=base/('prediction_audit_preflight_v2' if args.stage=='preflight' else 'prediction_audit_v2')
    assert not (base/'predictions/RUNNING.lock').exists(),'Wait for confirmation timing before even the old audit preflight'
    assert not (inp/'RUNNING.lock').exists(),'Timing must finish before heavy audits'
    previous=json.loads((prior/'summary.json').read_text());assert previous['passed'] and not previous['query_targets_accessed']
    ps=json.loads((inp/'summary.json').read_text());p=json.loads((inp/'protocol.json').read_text());assert ps['passed'] and not ps['query_targets_accessed']
    expected=2 if args.stage=='preflight' else 8192;assert ps['tasks']==expected and ps['predictors']==expected*27 and previous['counts']['predictors']==expected*27
    for n,h in ps['outputs_sha256'].items():assert sha(inp/n)==h
    for n,h in previous['outputs_sha256'].items():assert sha(prior/n)==h
    for n,h in json.loads((prior/'files.json').read_text()).items():assert sha(prior/n)==h
    hashes=dict(json.loads((prior/'protocol.json').read_text())['source_sha256'])
    for name in [Path(__file__).name,'probe_confirmation_path_reference.py','probe_exact_polytope_certificate.py','probe_exact_geometry_gate.py','test_probe_exact_geometry_gate.py','audit_probe_exact_polytope_certificate.py']:hashes[name]=sha(src/name)
    certificate_prerequisites=certificate_gate(base,src)
    for n,h in hashes.items():assert sha(src/n)==h,n
    if args.stage=='confirmation':
        old=base/'path_audit_preflight_v3';gate=json.loads((old/'summary.json').read_text());assert gate['passed'] and gate['counts']['predictors']==54
        assert json.loads((old/'protocol.json').read_text())['source_sha256']==hashes
        for n,h in gate['outputs_sha256'].items():assert sha(old/n)==h
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    loaded,checkpoints=suite.resources.legacy.oldfit.meta.load(root);assert checkpoints==p['checkpoint_manifest']
    seeds=p['seeds'][:64];cfgs=suite.catalogue(root);assert cfgs==p['configs'];names=p['methods']
    protocol=dict(stage=args.stage,source_sha256=hashes,prediction_summary_sha256=sha(inp/'summary.json'),all_prediction_audit_sha256=sha(prior/'summary.json'),
        seeds=seeds,query_targets_accessed=False,certificate_prerequisites=certificate_prerequisites,
        geometry_revision='v2: exact true-constraint vertex enumeration, boundary-chain fan and independent face-lattice volume; original float closure retained as diagnostic',
        tolerances=dict(original_volume_rtol=1e-8,original_volume_atol=1e-22,additional_true_volume_rtol=1e-8,additional_true_volume_atol=1e-22,nominal_pool_readout_budget=float(READOUT_BUDGET)),scope='All27 raw-input replays on fixed first64; original atomic and local/BP trajectory references; every discovered positive geometry independently checked, no full-posterior oracle')
    out.mkdir(parents=True,exist_ok=True);lock,identity=acquire_lock(out)
    try:
        if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
        else:exclusive_json(out/'protocol.json',protocol)
        protocol_hash=sha(out/'protocol.json');(out/'tasks').mkdir(exist_ok=True)
        if (out/'summary.json').exists():
            s=json.loads((out/'summary.json').read_text());assert s['passed']
            for n,h in s['outputs_sha256'].items():assert sha(out/n)==h
            print(json.dumps(dict(passed=True,already_complete=True,stage=args.stage)));return
        counts=Counter();files={};begin=time.perf_counter()
        with discovery_box(.12):
            for seed in seeds:
                committed=verify_commit(inp,seed,names,sha(inp/'protocol.json'));taskroot=out/'tasks'/str(seed);taskroot.mkdir(exist_ok=True)
                taskcommit=taskroot/'commit.json'
                if taskcommit.exists():
                    record=json.loads(taskcommit.read_text());assert record['protocol_sha256']==protocol_hash and record['prediction_commit_sha256']==committed['sha256']
                    for n,h in record['files'].items():assert sha(out/n)==h
                else:
                    attempt=taskroot/f'attempt_{time.time_ns()}';attempt.mkdir();cc=Counter();geometries=[];savedfiles={};cache={};proofs={};certificates={};pool_bounds=[]
                    c=json.loads((inp/committed['file']).read_text());rows={r['method']:r for r in json.loads((inp/c['rows_file']).read_text())}
                    for cfg in cfgs:
                        row=rows[cfg['name']];meta=row['metadata']
                        with np.load(inp/row['file']) as z:a={k:z[k].copy() for k in z.files}
                        x=a['x_observed'];v=a['v_observed'];q=a['q_observed'];repairs=[]
                        def traced(cfg,x,v,q,seed,loaded):return suite.resources.fit(cfg,x,v,q,seed,loaded,trace=True)
                        with conditioned.geometry_scope(repairs):fresh,mm=suite.resources.legacy.guarded_fit(cfg,x,v,q,seed,loaded,runner=traced)
                        assert mm['execution_failed']==meta['execution_failed'] and repairs==meta['geometry_repair_log']
                        for k,value in a.items():
                            if k not in ['x_observed','v_observed','q_observed']:assert value.tobytes()==fresh[k].tobytes(),(seed,cfg['name'],k);cc['fresh_input_replay_arrays']+=1
                        if meta['execution_failed']:
                            assert mm['failure_type']==meta['failure_type'];cc['retained_failed_predictors']+=1
                        elif meta['method_kind']=='mode_pool':
                            seen,checks=independent.reference(cfg,x,v,fresh,meta);cc.update(checks);feasible=[];positive=[]
                            for key,b in sorted(seen.items()):
                                if key not in proofs:
                                    feasibility=suite.resources.legacy.cold.base.branch_feasibility(x,v,b);poly=None;proof=None
                                    if feasibility['current_branch_feasible']:
                                        with conditioned.geometry_scope([]):poly,proof=shared.build(x,v,b)
                                    proofs[key]=(feasibility,poly,proof);cc['independent_mode_feasibility_checks']+=1
                                    if poly is not None:
                                        assert proof['proof']['accepted'];cube=check_cube(x,v,key,proof['proof'])
                                        ev=exact_volume(poly['facets'].reshape(-1,4),np.arange(len(poly['facets'])*4).reshape(-1,4),poly['interior'])
                                        assert np.isclose(ev['exact_volume']*np.prod(poly['scale']),poly['volume'],rtol=1e-8,atol=1e-22)
                                        rr,bb=observed_constraints(x,v,key)
                                        ec,witness=checked_certificate(poly,rr,bb)
                                        certificates[key]=ec
                                        if not ev['exact_boundary_closed']:cc['raw_float_normal_nonclosure_diagnostics']+=1
                                        cc['exact_closed_facet_boundaries']+=1
                                        path=attempt/(key+'.npz')
                                        with path.open('xb') as f:np.savez_compressed(f,**poly)
                                        rel=str(path.relative_to(out));savedfiles[rel]=sha(path);geometries.append(dict(pattern=key,file=rel,sha256=sha(path),cube=cube,raw_float_exact=ev,true_constraint_certificate=ec,exact_witness=witness,proof=proof['proof']))
                                        cc['exact_positive_geometries']+=1;cc['exact_cube_corners']+=16
                                feasibility,poly,proof=proofs[key]
                                if feasibility['current_branch_feasible']:feasible.append(key)
                                if poly is not None:positive.append(key);cache[key]=poly
                            assert feasible==meta['feasible_modes'] and positive==meta['positive_modes']
                            if positive:
                                pool_bounds.append(dict(method=cfg['name'],patterns=positive,**pool_coupling(positive,cache,certificates)))
                                points,allocation=shared.draw(cache,positive,2048,np.random.default_rng(np.random.SeedSequence([249911,seed,2048])))
                                assert points.tobytes()==a['points'].tobytes() and allocation.tobytes()==a['allocation'].tobytes();cc['original_particle_draws']+=1
                            else:assert a['points'][0].tobytes()==a['selected_b'].tobytes();cc['empty_pool_predictors']+=1
                            cc['original_path_predictors']+=1
                        else:cc['head_input_replays']+=1
                        cc['predictors']+=1
                    gp=attempt/'geometries.json';exclusive_json(gp,geometries);savedfiles[str(gp.relative_to(out))]=sha(gp)
                    bp=attempt/'pool_coupling.json';exclusive_json(bp,pool_bounds);savedfiles[str(bp.relative_to(out))]=sha(bp);cc['tasks']+=1
                    record=dict(seed=seed,protocol_sha256=protocol_hash,prediction_commit_sha256=committed['sha256'],counts=cc,files=savedfiles,query_targets_accessed=False)
                    exclusive_json(taskcommit,record)
                counts.update(record['counts']);files[str(taskcommit.relative_to(out))]=sha(taskcommit)
                print(json.dumps(dict(tasks=counts['tasks'],total=len(seeds),predictors=counts['predictors'],seconds=time.perf_counter()-begin,query_targets_accessed=False)),flush=True)
        assert counts['predictors']==27*len(seeds) and counts['tasks']==len(seeds)
        for n,h in hashes.items():assert sha(src/n)==h,n
        exclusive_json(out/'files.json',files)
        result=dict(passed=True,stage=args.stage,counts=counts,query_targets_accessed=False,may_evaluate_confirmation=args.stage=='confirmation',
            seconds_this_invocation=time.perf_counter()-begin,scope='Original code/block references plus independent mode/geometry arithmetic; not an independent reimplementation of every solver',
            outputs_sha256={n:sha(out/n) for n in ['protocol.json','files.json']})
        exclusive_json(out/'summary.json',result);print(json.dumps(result),flush=True)
    finally:
        if lock.exists() and json.loads(lock.read_text())==identity:lock.unlink()


if __name__=='__main__':main()
