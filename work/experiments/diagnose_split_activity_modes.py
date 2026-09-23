"""Post-trajectory reference audit of the predeclared split-activity hypothesis."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import split_activity_mode_trace as model
from diagnose_missing_mode_graph import distances


def risk(mask,weights,means,grid):
    mass=float(weights[mask].sum())
    if mass==0:return dict(mass=0.,truncation_risk=None,positive_modes=0)
    w=weights*mask/mass;full=np.einsum('k,rkq->rq',weights,means);pred=np.einsum('k,rkq->rq',w,means)
    errors=pred-full
    r=float(np.mean([np.trapezoid(errors[i]*errors[j],x=grid) for i,j in [(0,1),(2,3)]]))
    return dict(mass=mass,truncation_risk=r,positive_modes=int(mask.sum()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);a=parser.parse_args()
    root=a.project/'results/posterior_state_reuse';out=a.project/'results/split_activity_modes/diagnostic'
    previous=json.loads((a.project/'results/neighbor_mode_completion/development/protocol.json').read_text())
    hashes=previous['source_sha256'].copy()
    for name,h in hashes.items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    for name in [Path(__file__).name,'split_activity_mode_trace.py']:
        hashes[name]=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    configs=[dict(name='alm16',method='alm',sweeps=16),dict(name='alm20',method='alm',sweeps=20),
             dict(name='nodual16',method='nodual',sweeps=16),dict(name='pc80',method='pc',sweeps=80)]
    protocol=dict(seeds=list(range(5900000,5900016)),configs=configs,features=256,restarts=64,
        n_context=4,verification=model.verify(),source_sha256=hashes,
        observations='only x/v from frozen support reference; no truth or query answers',
        acquisition='initial and after each full activity / parameter pass, before dual update',
        scope='posthoc support-only mechanism diagnostic; reference membership is NOT an online feasibility solver; no timing claims')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];audits=[]
    refs={r['seed']:r for r in json.loads((root/'first_write_reference/coverage.json').read_text())}
    for seed in protocol['seeds']:
        refpath=root/'first_write_reference'/refs[seed]['reference_file']
        assert hashlib.sha256(refpath.read_bytes()).hexdigest()==refs[seed]['reference_sha256']
        data=json.loads(refpath.read_text());assert data['numerical_volume_reference_complete']
        x=np.array(data['x']);v=np.array(data['v']);anchor=np.zeros(4)
        pool=model.old.interface.make_pool(4,'prior256');starts,_=model.old.interface.select_pool(x,v,anchor,pool,64)
        traces=[]
        # Finish every frozen trajectory for this context BEFORE evaluating any reference member.
        for c in configs:
            observer=model.Observer(x,v)
            with model.old.core.pipeline.discovery_box(.12):
                actual=model.refine(starts,x,v,anchor,c['method'],c['sweeps'],observer)
                expected=model.original(starts,x,v,anchor,c['method'],c['sweeps'])
            assert np.array_equal(actual,expected),(seed,c)
            filename=f'trace_{seed}_{c["name"]}.json'
            path=out/filename;path.write_text(json.dumps(dict(forward=observer.forward,split=observer.split),indent=2),encoding='utf-8')
            audits.append(dict(seed=seed,method=c['name'],original_final_best_bitwise=True,
                trace_file=filename,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),events=observer.events))
            traces.append((c,observer))
        cells=sorted(data['positive_regions'],key=lambda r:r['pattern']);keys=[r['pattern'] for r in cells]
        patterns=np.array([np.frombuffer(bytes.fromhex(k),np.uint8) for k in keys])
        adjacency=np.abs(patterns[:,None].astype(int)-patterns[None,:].astype(int)).sum(-1)==1
        curve_audit=json.loads((root/f'conditional_risk/audit_{seed}.json').read_text());path=root/'conditional_risk'/curve_audit['curve_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==curve_audit['curve_sha256']
        with np.load(path) as curves:
            assert np.array_equal(curves['patterns'],np.array(keys))
            weights=curves['weights'];means=curves['region_means'];grid=curves['q']
            for c,observer in traces:
                forward=np.array([k in observer.forward for k in keys]);split=np.array([k in observer.split for k in keys]);union=forward|split
                distance=distances(adjacency,forward)
                extra=split&~forward;off_keys=set(observer.split)-set(observer.forward)
                scores={}
                for label,mask in [('forward',forward),('split',split),('union',union)]:
                    steps=distances(adjacency,mask)
                    for hop in [0,1,2]:
                        scores[f'{label}_h{hop}']=risk((steps>=0)&(steps<=hop),weights,means,grid)
                details=[dict(pattern=keys[i],mass=float(weights[i]),forward_graph_distance=int(distance[i]),
                    **observer.split[keys[i]]) for i in np.flatnonzero(extra)]
                rows.append(dict(seed=seed,method=c['name'],forward_proposals=len(observer.forward),split_proposals=len(observer.split),
                    additional_split_proposals=len(off_keys),additional_positive_modes=int(extra.sum()),
                    additional_posterior_mass=float(weights[extra].sum()),
                    new_component_mass=float(weights[extra&(distance<0)].sum()),
                    additional_pattern_key_bytes=sum(len(bytes.fromhex(k)) for k in off_keys),
                    scores=scores,additional_positive_details=details))
        (out/'diagnostics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        (out/'audits.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=seed-protocol['seeds'][0]+1,total=16,rows=len(rows))),flush=True)
    summary=[]
    for c in configs:
        rr=[r for r in rows if r['method']==c['name']]
        scores={k:{field:float(np.mean([r['scores'][k][field] for r in rr])) for field in ['mass','truncation_risk','positive_modes']} for k in rr[0]['scores']}
        summary.append(dict(method=c['name'],tasks_with_additional_positive_modes=sum(r['additional_positive_modes']>0 for r in rr),
            additional_positive_modes_total=sum(r['additional_positive_modes'] for r in rr),
            **{f'mean_{k}':float(np.mean([r[k] for r in rr])) for k in ['additional_split_proposals','additional_posterior_mass','new_component_mass']},scores=scores))
    result=dict(summary=summary,trace_audits=len(audits),all_original_paths_bitwise=True,source_hashes=len(hashes),
        scope='old-task diagnostic; numerical reference applied after candidate paths; no unseen-query or online-cost claim')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
