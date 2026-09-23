"""Frozen actual-cell direction admission; not a full online benchmark."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import batched_conditional_fiber as fiber
import credit_fiber_directions as direction
import neighbor_mode_memory as geometry
import region_posterior_memory as posterior


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(x,v,cfg):
    original=direction.model.prepare(x,v,cfg)
    captured,stats=direction.capture(x,v,cfg,True)
    assert np.array_equal(original[0],captured[0]) and np.array_equal(original[1],captured[1])
    for a,b in zip(original[3],captured[3]):
        assert a.feedback_bank.proofs==b.feedback_bank.proofs and list(a.route_kept)==list(b.route_kept)
    for key in ['route_extra_all_keys','route_extra_pattern_keys','route_extra_origins']:
        assert original[2][key]==captured[2][key]
    # Explicitly reject all three possible global-credit entry points.
    helper=direction.bp_forces;jac=direction.model.base.forward_jacobian
    bp_module=direction.model.old.old.old.old.old.core.batched;refine=bp_module.refine
    def forbidden(*a,**k):raise AssertionError('Global BP in local-only capture')
    try:
        direction.bp_forces=forbidden;direction.model.base.forward_jacobian=forbidden;bp_module.refine=forbidden
        local,local_stats=direction.capture(x,v,cfg,False)
    finally:direction.bp_forces=helper;direction.model.base.forward_jacobian=jac;bp_module.refine=refine
    assert np.array_equal(local[0],captured[0]) and np.array_equal(local[1],captured[1])
    for kind in ['raw','augmented','residual']:assert np.array_equal(local_stats.scatter[kind],stats.scatter[kind])
    return dict(passed=True,bank_and_region_sequence_bitwise=True,proofs_and_extra_origins_exact=True,
                local_only_no_global_bp=True,local_scatter_with_without_control_bitwise=True,callbacks=stats.callbacks)


def evaluate(q,samples,groups,polys,dirs,which):
    prediction=np.zeros(len(q));variance_sum=np.zeros(len(q));pieces=0;peak_pieces=0;minimum_variance=np.inf
    interval_seconds=0.;integral_seconds=0.;line_lengths=[];checks=0
    for key,indices in groups.items():
        poly=polys[key];d=dirs[key][which]
        begin=time.perf_counter();lo,hi=fiber.intervals(poly['a'],poly['rhs'],samples[indices],d)
        interval_seconds+=time.perf_counter()-begin;line_lengths.extend((hi-lo).tolist())
        for start in range(0,len(indices),32):
            batch=indices[start:start+32];left=lo[start:start+32];right=hi[start:start+32]
            begin=time.perf_counter();mu,second,meta=fiber.moments(q,samples[batch],d,left,right)
            integral_seconds+=time.perf_counter()-begin
            variance=second-mu*mu;minimum_variance=min(minimum_variance,float(variance.min()))
            assert np.min(variance)>-1e-12 and np.all(np.isfinite(mu)) and mu.min()>-1e-10 and mu.max()<1+1e-10
            # Retain tiny signed numerical error; do not clip gains to zero.
            prediction+=mu.sum(0)/len(samples);variance_sum+=variance.sum(0)/len(samples)
            pieces+=meta['final_pieces'];peak_pieces=max(peak_pieces,meta['max_pieces']);checks+=len(batch)
    assert checks==512
    gain=float(np.trapezoid(variance_sum,x=q)/512)
    coarse=float(np.trapezoid(variance_sum[::2],x=q[::2])/512)
    return prediction,dict(estimated_expected_excess_reduction=gain,coarse_reduction=coarse,
                            interval_seconds=interval_seconds,integration_seconds=integral_seconds,
                            final_pieces=pieces,maximum_block_pieces=peak_pieces,minimum_raw_conditional_variance=minimum_variance,
                            mean_conditional_line_length=float(np.mean(line_lengths)),particle_checks=checks)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project.resolve()
    results=root/'results';online=results/'typed_routing_memory/development';deferred=results/'deferred_typed_memory/development'
    curve=results/'posterior_state_reuse/conditional_risk';out=results/'credit_fiber/diagnostic'
    parent=json.loads((deferred/'protocol.json').read_text());hashes=dict(parent['source_sha256'])
    primitives=json.loads((results/'conditional_fiber/primitive/protocol.json').read_text());hashes.update(primitives['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for path in [Path(__file__),Path(fiber.__file__),Path(direction.__file__)]:hashes[path.name]=sha(path)
    old=json.loads((online/'protocol.json').read_text());name='alm_tied_forward_full'
    cfg=dict(next(c for c in old['configs'] if c['name']==name),archive=True,pool='posterior_mix',posterior_samples=512,proposal_budget=1024)
    states={r['seed']:r for r in json.loads((online/'episodes.json').read_text()) if r['n_context']==4 and r['method']==name}
    first_seed=5900001;ca=json.loads((curve/f'audit_{first_seed}.json').read_text());cp=curve/ca['curve_file'];assert sha(cp)==ca['curve_sha256']
    with np.load(cp) as z:x=z['x'];v=z['v']
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(deferred/'protocol.json'),original_protocol_sha256=sha(online/'protocol.json'),
                  config=cfg,seeds=list(range(5900000,5900016)),directions=list(direction.DIRECTIONS),primary='local_augmented_cov',
                  particles=512,query_points=257,coarse_query_points=129,particle_chunk=32,order_seed=472823,
                  verification=dict(batch=fiber.verify(),forces=direction.verify(),capture=preflight(x,v,cfg)),
                  scope='fixed original ALM discovery/particles; component admission only, no teacher answers, no online speedup claim')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];audits=[];order_rng=np.random.default_rng(protocol['order_seed'])
    for seed in protocol['seeds']:
        ca=json.loads((curve/f'audit_{seed}.json').read_text());cp=curve/ca['curve_file'];assert sha(cp)==ca['curve_sha256']
        with np.load(cp) as z:x=z['x'];v=z['v']
        start=time.perf_counter();captured,stats=direction.capture(x,v,cfg,True);capture_seconds=time.perf_counter()-start
        bank,regs,meta,collectors=captured;original=states[seed];path=online/original['state_file'];assert sha(path)==original['state_sha256']
        assert [r.tobytes().hex() for r in regs]==original['evaluated_pattern_keys']
        for a,b in zip(meta['feedback_details'],original['feedback_details']):assert a['proofs']==b['proofs']
        begin=time.perf_counter();_,new,detail=direction.model.feedback.old.old.old.old.materialize_retained(x,v,bank,regs,512)
        materialize_seconds=time.perf_counter()-begin
        with np.load(path) as z:assert np.array_equal(z['anchor'],new.anchor) and np.array_equal(z['samples'],new.samples)
        assert detail['positive_mode_keys']==original['positive_mode_keys']
        samples=new.samples;keys=original['positive_mode_keys'];polys={};directions={};covs=[];direction_details=[];preparation=[]
        begin=time.perf_counter()
        for key in keys:
            reg=np.frombuffer(bytes.fromhex(key),np.uint8).reshape(4,len(x));p,c,a,rhs=geometry.pattern_matrix(x,v,reg)
            poly,note=posterior.polytope(a,rhs);assert poly is not None
            mean,cov=direction.covariance(poly);start=time.perf_counter();dd,dm=direction.directions(cov,p,stats.scatter);direction_seconds=time.perf_counter()-start
            polys[key]=poly;directions[key]=dd;covs.append(cov);direction_details.append(dm);preparation.append(direction_seconds)
        rebuild_seconds=time.perf_counter()-begin
        labels=[geometry.base.pattern(x,b).astype(np.uint8).tobytes().hex() for b in samples]
        assert set(labels)<=set(keys);groups={key:np.flatnonzero(np.array(labels)==key) for key in dict.fromkeys(labels)}
        direction_file=out/f'directions_{seed}.npz'
        np.savez_compressed(direction_file,keys=np.array(keys),covariance=np.array(covs),directions=np.array([[directions[k][n] for n in protocol['directions']] for k in keys]),
                            scatter=np.array([stats.scatter[k] for k in direction.KINDS]),kinds=np.array(direction.KINDS),samples=samples,labels=np.array(labels),x=x,v=v)
        audit=dict(seed=seed,original_state_sha256=original['state_sha256'],curve_sha256=ca['curve_sha256'],state_and_proofs_exact=True,
                   directions_file=direction_file.name,directions_sha256=sha(direction_file),positive_regions=len(keys),
                   scatter_counts=stats.count,callbacks=stats.callbacks,scatter_numeric_bytes=sum(a.nbytes for a in stats.scatter.values()),
                   capture_seconds=capture_seconds,force_seconds=stats.seconds,materialize_seconds=materialize_seconds,
                   diagnostic_geometry_rebuild_seconds=rebuild_seconds,direction_seconds=sum(preparation),direction_details=direction_details)
        # Reference moments are read only after every deployable direction and
        # its state file is fixed. The direction module never receives them.
        with np.load(cp) as z:
            q=z['q'][::4];full=np.einsum('k,rkq->rq',z['weights'],z['region_means'])[:,::4]
        assert len(q)==257
        for which in order_rng.permutation(protocol['directions']).tolist():
            begin=time.perf_counter();prediction,measure=evaluate(q,samples,groups,polys,directions,which);elapsed=time.perf_counter()-begin
            excess=float(np.mean([np.trapezoid((prediction-full[a])*(prediction-full[b]),x=q) for a,b in [(0,1),(2,3)]]))
            pred_file=out/f'prediction_{seed}_{which}.npz';np.savez_compressed(pred_file,q=q,prediction=prediction)
            rows.append(dict(seed=seed,direction=which,conditional_excess=excess,complete_readout_component_seconds=elapsed,
                             prediction_file=pred_file.name,prediction_sha256=sha(pred_file),**measure))
        audits.append(audit)
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'audits.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=seed,completed=len(rows),total=176,positive_regions=len(keys),state_exact=True)),flush=True)
    result=dict(complete=True,rows=len(rows),state_replays=len(audits),sources=len(hashes),scope=protocol['scope'])
    assert len(rows)==176
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
