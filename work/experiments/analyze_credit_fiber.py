"""Independent direction reconstruction, changed-chunk readout and risk audit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import credit_fiber_directions as direction
import batched_conditional_fiber as batch
import conditional_fiber_readout as scalar
import neighbor_mode_memory as geometry
import region_posterior_memory as posterior


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(delta):
    ids=np.random.default_rng(680341).integers(0,len(delta),(20000,len(delta)))
    return np.quantile(np.asarray(delta)[ids].mean(1),[.025,.975]).tolist()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project.resolve()
    inp=root/'results/credit_fiber/diagnostic';out=root/'results/credit_fiber/analysis';curve=root/'results/posterior_state_reuse/conditional_risk'
    p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text())
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert run['complete'] and run['rows']==176
    rows=json.loads((inp/'rows.json').read_text());audits=json.loads((inp/'audits.json').read_text())
    lookup={(r['seed'],r['direction']):r for r in rows};assert len(lookup)==176
    oldrisk={(r['seed'],r['method']):r for r in json.loads((root/'results/sampling_averaged_risk/diagnostic/rows.json').read_text())}
    checks=dict(rebuilt_regions=0,rebuilt_directions=0,readout_replays=0,scalar_moment_checks=0,query_risk_replays=0)
    covmeta=[];original_rows=[];worst_scalar_error=0.;worst_replay_error=0.
    for audit in audits:
        seed=audit['seed'];path=inp/audit['directions_file'];assert sha(path)==audit['directions_sha256']
        with np.load(path) as z:
            keys=z['keys'];saved_cov=z['covariance'];saved_dirs=z['directions'];samples=z['samples'];labels=z['labels'];x=z['x'];v=z['v']
            scatter={name:matrix for name,matrix in zip(z['kinds'].tolist(),z['scatter'])}
        ca=json.loads((curve/f'audit_{seed}.json').read_text());cp=curve/ca['curve_file'];assert sha(cp)==audit['curve_sha256']==ca['curve_sha256']
        with np.load(cp) as z:
            q=z['q'][::4];full=np.einsum('k,rkq->rq',z['weights'],z['region_means'])[:,::4]
            weights=z['weights']*np.array([k in set(keys) for k in z['patterns']]);weights/=weights.sum()
            truncated=np.einsum('k,rkq->rq',weights,z['region_means'])[:,::4]
            second_truncated=np.einsum('k,rkq->rq',weights,z['region_second_moments'])[:,::4]
        expected_on_grid=float(np.mean([np.trapezoid((truncated[a]-full[a])*(truncated[b]-full[b])+
                                                     ((second_truncated[a]+second_truncated[b])/2-truncated[a]*truncated[b])/512,x=q)
                                        for a,b in [(0,1),(2,3)]]))
        polys={};ds={};groups={}
        for i,key in enumerate(keys.tolist()):
            regs=np.frombuffer(bytes.fromhex(key),np.uint8).reshape(4,len(x));output,_,a,rhs=geometry.pattern_matrix(x,v,regs)
            poly,_=posterior.polytope(a,rhs);assert poly is not None
            _,cov=direction.covariance(poly);assert np.array_equal(cov,saved_cov[i]);checks['rebuilt_regions']+=1
            dirs,meta=direction.directions(cov,output,scatter)
            for j,name in enumerate(p['directions']):assert np.array_equal(dirs[name],saved_dirs[i,j]);checks['rebuilt_directions']+=1
            eig=np.linalg.eigvalsh(cov);weights=poly['volume'];cosine=abs(dirs[p['primary']]@dirs['pca'])
            covmeta.append(dict(seed=seed,pattern=key,volume=weights,second_to_first_eigenvalue=float(eig[-2]/eig[-1]),
                                credit_pca_absolute_cosine=float(cosine),support_rank=meta['support_rank'],fallback=meta['pca_fallback']))
            polys[key]=poly;ds[key]=dirs;groups[key]=np.flatnonzero(labels==key)
        baseline=posterior.make_predict(samples)(q)
        baseline_ce=float(np.mean([np.trapezoid((baseline-full[a])*(baseline-full[b]),x=q) for a,b in [(0,1),(2,3)]]))
        original_rows.append(dict(seed=seed,conditional_excess_on_257_grid=baseline_ce,sampling_averaged_excess_on_257_grid=expected_on_grid,
                                  expected_257_minus_1025=expected_on_grid-oldrisk[seed,p['config']['name']]['expected_total_excess']))
        for name in p['directions']:
            row=lookup[seed,name];predpath=inp/row['prediction_file'];assert sha(predpath)==row['prediction_sha256']
            with np.load(predpath) as z:prediction=z['prediction'];assert np.array_equal(q,z['q'])
            replay=np.zeros(len(q));conditional_variance=np.zeros(len(q))
            # Reverse cell order and use 17 instead of 32 samples per block.
            for key in reversed(keys.tolist()):
                indices=groups[key]
                if not len(indices):continue
                poly=polys[key];d=ds[key][name];lo,hi=batch.intervals(poly['a'],poly['rhs'],samples[indices],d)
                for begin in range(0,len(indices),17):
                    ii=indices[begin:begin+17];l=lo[begin:begin+17];h=hi[begin:begin+17]
                    mu,second,_=batch.moments(q,samples[ii],d,l,h)
                    replay+=mu.sum(0)/512;conditional_variance+=(second-mu*mu).sum(0)/512
                # Independent scalar integrator on first/last sample and
                # endpoint/middle queries for every occupied cell/direction.
                for offset in sorted(set([0,len(indices)-1])):
                    point=samples[indices[offset]]
                    for qi in [0,len(q)//2,len(q)-1]:
                        m,s,_=scalar.line_moments(q[qi],point,d,lo[offset],hi[offset])
                        bm,bs,_=batch.moments(q[qi:qi+1],point[None,:],d,lo[offset:offset+1],hi[offset:offset+1])
                        error=max(abs(m-bm[0,0]),abs(s-bs[0,0]));assert error<1e-11
                        worst_scalar_error=max(worst_scalar_error,error);checks['scalar_moment_checks']+=1
            err=float(np.max(np.abs(replay-prediction)));assert err<1e-12;worst_replay_error=max(worst_replay_error,err)
            gain=float(np.trapezoid(conditional_variance,x=q)/512)
            assert abs(gain-row['estimated_expected_excess_reduction'])<1e-14;checks['readout_replays']+=1
            risk=float(np.mean([np.trapezoid((prediction-full[a])*(prediction-full[b]),x=q) for a,b in [(0,1),(2,3)]]))
            assert abs(risk-row['conditional_excess'])<1e-15;checks['query_risk_replays']+=1
            row['estimated_sampling_averaged_excess']=expected_on_grid-gain
        print(json.dumps(dict(seed=seed,**checks)),flush=True)
    summary=[];paired=[];primary=p['primary']
    for name in p['directions']:
        rr=[lookup[seed,name] for seed in p['seeds']]
        record=dict(direction=name,**{key:float(np.mean([r[key] for r in rr])) for key in ['conditional_excess','estimated_expected_excess_reduction','estimated_sampling_averaged_excess','complete_readout_component_seconds','integration_seconds','interval_seconds']},
                    maximum_grid_change=max(abs(r['estimated_expected_excess_reduction']-r['coarse_reduction']) for r in rr),
                    mean_final_pieces=float(np.mean([r['final_pieces'] for r in rr])),maximum_block_pieces=max(r['maximum_block_pieces'] for r in rr))
        summary.append(record)
        if name!=primary:
            gain=np.array([lookup[s,primary]['estimated_expected_excess_reduction']-lookup[s,name]['estimated_expected_excess_reduction'] for s in p['seeds']])
            ce=np.array([lookup[s,primary]['conditional_excess']-lookup[s,name]['conditional_excess'] for s in p['seeds']])
            seconds=np.array([lookup[s,primary]['complete_readout_component_seconds']-lookup[s,name]['complete_readout_component_seconds'] for s in p['seeds']])
            paired.append(dict(comparator=name,additional_reduction=float(gain.mean()),descriptive_gain_ci95=ci(gain),
                               conditional_excess_delta=float(ce.mean()),descriptive_fixed_ce_ci95=ci(ce),
                               readout_seconds_delta=float(seconds.mean()),descriptive_time_ci95=ci(seconds),
                               greater_reduction_tasks=int(np.sum(gain>1e-12)),equal_reduction_tasks=int(np.sum(np.abs(gain)<=1e-12)),lesser_reduction_tasks=int(np.sum(gain<-1e-12))))
    out.mkdir(parents=True,exist_ok=True)
    result=dict(complete=True,analysis_source_sha256=sha(Path(__file__)),input_sha256={name:sha(inp/name) for name in ['protocol.json','run_audit.json','rows.json','audits.json']},
                original_risk_rows_sha256=sha(root/'results/sampling_averaged_risk/diagnostic/rows.json'),checks=checks,
                maximum_scalar_moment_error=worst_scalar_error,maximum_changed_chunk_prediction_error=worst_replay_error,
                original_fixed_ce_257_mean=float(np.mean([r['conditional_excess_on_257_grid'] for r in original_rows])),
                summary=summary,paired=paired,scope='same old task/particle seed, descriptive bootstrap without multiplicity correction; numerical reference, not end-to-end superiority')
    for name,value in [('summary.json',result),('rows.json',rows),('region_geometry.json',covmeta),('original_grid_risk.json',original_rows)]:
        (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    names=[s['direction'] for s in summary];gain=np.array([s['estimated_expected_excess_reduction'] for s in summary]);y=np.arange(len(names))
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),sharey=True)
    colors=['#178589' if n==primary else '#547c9c' for n in names]
    axes[0].barh(y,gain,color=colors);axes[0].set_yticks(y,names);axes[0].invert_yaxis();axes[0].set_xlabel('Estimated sampling-averaged excess-risk reduction')
    axes[1].barh(y,[s['conditional_excess'] for s in summary],color=colors)
    axes[1].axvline(result['original_fixed_ce_257_mean'],color='#c65a44',ls='--',label='Original fixed-particle prediction')
    axes[1].set_xlabel('Fixed-particle first-write conditional excess risk');axes[1].legend(fontsize=8,loc='lower right')
    for ax in axes:ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Shared joint integration helps; compare credit against geometry and zero-dual controls')
    fig.tight_layout();fig.savefig(out/'credit_fiber_results.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fa=dict(source_sha256=sha(Path(__file__)),summary_sha256=sha(out/'summary.json'),figure_sha256=sha(out/'credit_fiber_results.png'))
    (out/'figure_audit.json').write_text(json.dumps(fa,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,checks=checks,summary=summary,paired=paired),indent=2),flush=True)


if __name__=='__main__':main()
