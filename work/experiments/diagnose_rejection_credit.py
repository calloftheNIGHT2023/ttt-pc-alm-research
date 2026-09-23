"""Frozen same-trajectory credit-volume and rejection sampling diagnosis."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import rejection_credit_capture as capture
import parallelotope_sampler as sampler
import neighbor_mode_memory as geometry


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(x,v,cfg):
    old=capture.model.prepare(x,v,cfg);new,roles=capture.capture(x,v,cfg,True)
    assert np.array_equal(old[0],new[0]) and np.array_equal(old[1],new[1])
    for a,b in zip(old[3],new[3]):assert a.feedback_bank.proofs==b.feedback_bank.proofs and set(a.retained)==set(b.retained)
    # Watchers receive all actual parameter/activity events but never mutate.
    jac=capture.model.base.forward_jacobian
    bp_module=capture.model.old.old.old.old.old.core.batched;refine=bp_module.refine
    observer=capture.WatchCollector
    def forbidden(*args,**kwargs):raise AssertionError('Global BP in local-only capture')
    def no_bp_observer(x,v,kind,*args,**kwargs):
        if kind=='bp':raise AssertionError('BP observer in local-only capture')
        return observer(x,v,kind,*args,**kwargs)
    try:
        capture.model.base.forward_jacobian=forbidden;bp_module.refine=forbidden;capture.WatchCollector=no_bp_observer
        local,local_roles=capture.capture(x,v,cfg,False)
    finally:
        capture.model.base.forward_jacobian=jac;bp_module.refine=refine;capture.WatchCollector=observer
    assert np.array_equal(local[0],new[0]) and np.array_equal(local[1],new[1]) and 'bp' not in local_roles
    for a,b in zip(local_roles['local'],roles['local']):
        for key in a.retained:assert np.array_equal(a.retained[key]['a'],b.retained[key]['a'])
    return dict(passed=True,original_bank_and_regions_bitwise=True,original_proofs_and_local_credit_unchanged=True,
                bp_observer_optional_no_local_effect=True,local_only_forbids_global_bp_and_bp_observer=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project.resolve();results=root/'results'
    previous=results/'credit_fiber/diagnostic';online=results/'typed_routing_memory/development';curve=results/'posterior_state_reuse/conditional_risk'
    refroot=results/'posterior_state_reuse/first_write_reference';out=results/'rejection_credit/diagnostic'
    parent=json.loads((previous/'protocol.json').read_text());hashes=dict(parent['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for path in [Path(__file__),Path(capture.__file__),Path(sampler.__file__)]:hashes[path.name]=sha(path)
    cfg=parent['config'];states={r['seed']:r for r in json.loads((online/'episodes.json').read_text()) if r['method']==cfg['name'] and r['n_context']==4}
    refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())}
    ca=json.loads((curve/'audit_5900001.json').read_text());cp=curve/ca['curve_file'];assert sha(cp)==ca['curve_sha256']
    with np.load(cp) as z:x=z['x'];v=z['v']
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(previous/'protocol.json'),config=cfg,seeds=parent['seeds'],
                  methods=['c20','residual','bp','local','bp_residual'],cross_credit_cap=32,repetitions=4,proposals=16384,rng_seed=481739,method_order_seed=481741,
                  verification=dict(sampler=sampler.verify(),capture=preflight(x,v,cfg)),
                  scope='common actual ALM pool, credit isolation and sampler acceptance; no query prediction or end-to-end superiority')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];audits=[];order_rng=np.random.default_rng(protocol['method_order_seed'])
    for seed in protocol['seeds']:
        ca=json.loads((curve/f'audit_{seed}.json').read_text());cp=curve/ca['curve_file'];assert sha(cp)==ca['curve_sha256']
        with np.load(cp) as z:x=z['x'];v=z['v']
        begin=time.perf_counter();result,roles=capture.capture(x,v,cfg,True);capture_seconds=time.perf_counter()-begin
        original=states[seed];assert [r.tobytes().hex() for r in result[1]]==original['evaluated_pattern_keys']
        for a,b in zip(result[2]['feedback_details'],original['feedback_details']):assert a['proofs']==b['proofs']
        state_path=online/original['state_file'];assert sha(state_path)==original['state_sha256']
        _,state,detail=capture.model.feedback.old.old.old.old.materialize_retained(x,v,result[0],result[1],512)
        with np.load(state_path) as z:assert np.array_equal(z['anchor'],state.anchor) and np.array_equal(z['samples'],state.samples)
        assert detail['positive_mode_keys']==original['positive_mode_keys']
        begin=time.perf_counter();allregs=capture.pool(result);removed=capture.model.conflict.screen.contract(x,v,allregs,20);regs=allregs[~removed]
        c20_seconds=time.perf_counter()-begin;boxes=[];matrices=[];rhs=[];begin=time.perf_counter()
        for reg in regs:
            p,c,a,r=geometry.pattern_matrix(x,v,reg);boxes.append(sampler.make_box(p,c,v));matrices.append(a);rhs.append(r)
        box_seconds=time.perf_counter()-begin;keys=[r.tobytes().hex() for r in regs];volumes=np.array([b['volume'] for b in boxes])
        masks={'c20':np.zeros(len(regs),bool)};proofs={}
        for kind in ['residual','bp','local']:
            masks[kind],proofs[kind]=capture.screen(x,v,regs,volumes,roles[kind],protocol['cross_credit_cap'])
        masks['bp_residual']=masks['bp']|masks['residual']
        proposal_file=out/f'proposal_{seed}.npz'
        np.savez_compressed(proposal_file,patterns=np.array(keys),volumes=volumes,rejected=np.array([masks[m] for m in protocol['methods']]),
                            methods=np.array(protocol['methods']),transforms=np.array([b['transform'] for b in boxes]),
                            centers=np.array([b['center'] for b in boxes]),radii=np.array([b['radius'] for b in boxes]),x=x,v=v)
        proof_file=out/f'proofs_{seed}.json';proof_file.write_text(json.dumps(proofs,indent=2),encoding='utf-8')
        # Evaluator reference is first loaded after deployable rejection sets
        # and proposal volumes have been committed to files.
        refpath=refroot/refs[seed]['reference_file'];assert sha(refpath)==refs[seed]['reference_sha256']
        reference=json.loads(refpath.read_text());assert reference['numerical_volume_reference_complete']
        positive={r['pattern']:r['volume'] for r in reference['positive_regions']}
        good=np.array([key in positive for key in keys]);target_volume=sum(positive[key] for key in keys if key in positive)
        assert target_volume>0 and set(original['positive_mode_keys'])<=set(keys)
        for method in protocol['methods']:assert not np.any(masks[method]&good),(seed,method)
        exclusive=masks['local']&~masks['bp_residual'];audit=dict(seed=seed,raw_candidates=len(allregs),c20_survivors=len(regs),
            original_state_and_proofs=True,original_state_sha256=original['state_sha256'],known_positive_regions=int(good.sum()),target_volume=float(target_volume),
            reference_sha256=refs[seed]['reference_sha256'],proposal_sha256=sha(proposal_file),proofs_sha256=sha(proof_file),
            capture_seconds=capture_seconds,c20_seconds=c20_seconds,box_seconds=box_seconds,
            local_exclusive_modes=int(exclusive.sum()),local_exclusive_proposal_volume=float(volumes[exclusive].sum()),
            full_proposal_volume=float(volumes.sum()),methods={})
        method_order=order_rng.permutation(protocol['methods']).tolist();audit['method_order']=method_order
        for method in method_order:
            keep=~masks[method];indices=np.flatnonzero(keep);remaining=float(volumes[keep].sum());expected=target_volume/remaining
            assert expected<=1+1e-8
            audit['methods'][method]=dict(rejected=int(masks[method].sum()),remaining_proposal_volume=remaining,
                removed_proposal_volume=float(volumes[masks[method]].sum()),expected_acceptance=expected,
                expected_proposals_relative_to_c20=remaining/volumes.sum())
            for repeat in range(protocol['repetitions']):
                rng=np.random.default_rng(np.random.SeedSequence([protocol['rng_seed'],seed,repeat]));begin=time.perf_counter()
                bank,owners,note=sampler.attempts([boxes[i] for i in indices],[matrices[i] for i in indices],[rhs[i] for i in indices],protocol['proposals'],rng)
                seconds=time.perf_counter()-begin;owners=indices[owners]
                if len(bank):
                    codes,activities=capture.model.light.forward_many(x,bank)
                    assert np.array_equal(codes,regs[owners])
                    error=float(np.max(np.abs(activities[:,-1,:]-v)));assert error<=.001+1e-12
                else:error=None
                counts=np.bincount(owners,minlength=len(regs))
                assert not np.any(counts[~good])
                rows.append(dict(seed=seed,method=method,repetition=repeat,accepted=len(bank),proposals=protocol['proposals'],
                                 empirical_acceptance=len(bank)/protocol['proposals'],reference_expected_acceptance=expected,
                                 zero_samples=not len(bank),max_support_error=error,sampling_seconds=seconds,
                                 accepted_region_counts={keys[i]:int(n) for i,n in enumerate(counts) if n}))
        audits.append(audit)
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'audits.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=seed,completed=len(rows),total=320,pool=len(regs),positive=int(good.sum()),local_exclusive=int(exclusive.sum()))),flush=True)
    result=dict(complete=True,rows=len(rows),tasks=len(audits),sources=len(hashes),scope=protocol['scope']);assert len(rows)==320
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
