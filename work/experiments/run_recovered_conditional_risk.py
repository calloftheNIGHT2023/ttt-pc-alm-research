"""Frozen current-context conditional function-risk decomposition.

No teacher parameters or query answers. Reference sampling is an offline
evaluator, never provided to any frozen online discovery/update routine.
"""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
import neighbor_mode_memory as model


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(points,q):
    first=np.empty(len(q));second=np.empty(len(q))
    for start in range(0,len(q),256):
        h=np.broadcast_to(q[start:start+256],(len(points),len(q[start:start+256])))
        for j in range(points.shape[1]):h=model.base.g(h+points[:,j,None])
        first[start:start+256]=h.mean(0);second[start:start+256]=(h*h).mean(0)
    return first,second


def estimate(q,full,fullsecond,truncated,truncsecond,actual,pair,count):
    left,right=pair
    aa=(actual-full[left])*(actual-full[right])
    integrands=dict(actual_excess=aa.mean(0),truncation_excess=(truncated[left]-full[left])*(truncated[right]-full[right]),
        expected_sampling_excess=((truncsecond[left]+truncsecond[right])/2-truncated[left]*truncated[right])/count,
        bayes_risk=(fullsecond[left]+fullsecond[right])/2-full[left]*full[right])
    fine={k:float(np.trapezoid(v,x=q)) for k,v in integrands.items()}
    coarse={k:float(np.trapezoid(v[::2],x=q[::2])) for k,v in integrands.items()}
    fine['expected_total_excess']=fine['truncation_excess']+fine['expected_sampling_excess']
    coarse['expected_total_excess']=coarse['truncation_excess']+coarse['expected_sampling_excess']
    return dict(pair=pair,fine=fine,coarse=coarse,online_repetition_actual_excess=np.trapezoid(aa,x=q,axis=1).tolist())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    refroot=root/'results/recovered_support_reference';faces=refroot/'exact_faces';online=root/'results/recovered_online_comparison/development'
    parent=json.loads((faces/'protocol.json').read_text());hashes=dict(parent['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for path,source in [(faces/'independent_audit.json','audit_recovered_reference_faces.py'),
        (refroot/'analysis/summary.json','audit_recovered_support_reference.py')]:
        note=json.loads(path.read_text());assert note['passed'] and note['source_sha256']==sha(Path(__file__).with_name(source))
        for name,value in note.get('input_sha256',{}).items():assert sha(refroot/'development'/name)==value
    assert json.loads((faces/'summary.json').read_text())['complete_references']==16
    hashes[Path(__file__).name]=sha(Path(__file__))
    op=json.loads((online/'protocol.json').read_text());names=['alm_c5','adam60_c5','pc_c5','nodual_c5','direct4096_c5']
    pairs=[list(pair) for pair in itertools.combinations(range(4),2)]
    p=dict(source_sha256=hashes,seeds=op['seeds'],methods=names,reference_replicas=4,samples_per_region_per_replica=2048,
        frozen_online_repetitions=2,frozen_online_samples=2048,grid_points=1025,coarse_grid_points=513,rng_seed=482811,
        all_pairs=pairs,primary_pairs=[[0,1],[2,3]],primary_comparator='direct4096_c5',
        reference_protocol_sha256=sha(faces/'protocol.json'),reference_coverage_sha256=sha(faces/'coverage.json'),
        reference_audit_sha256=sha(faces/'independent_audit.json'),online_protocol_sha256=sha(online/'protocol.json'),
        scope='All 16 reused development supports; conditional risk with numerical complete-volume reference; no teacher/query answers, no online adaptation changes')
    out=root/'results/recovered_conditional_risk/development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    # Read only immutable state-file/hash fields; no query-loss values used.
    stages={(r['seed'],r['method'],r['repetition']):(r['state_file'],r['state_sha256']) for r in json.loads((online/'rows.json').read_text()) if r['n']==4 and r['method'] in names}
    covered={r['seed']:r for r in json.loads((faces/'coverage.json').read_text())};q=np.linspace(0,1,p['grid_points']);rows=[];audits=[]
    checks=dict(region_geometry_matches=0,reference_batches=0,reference_samples=0,frozen_states=0,online_prediction_replays=0)
    for seed in p['seeds']:
        start=time.perf_counter();row=covered[seed];assert row['complete'];path=refroot/'development'/row['reference_file'];assert sha(path)==row['reference_sha256']
        ref=json.loads(path.read_text());x=np.array(ref['x']);v=np.array(ref['v']);polys=sorted(ref['positive_regions'],key=lambda r:r['pattern'])
        volumes=np.array([r['volume'] for r in polys]);weights=volumes/volumes.sum();k=len(polys)
        mean=np.empty((4,k,len(q)));second=np.empty_like(mean);reference_points=np.empty((4,k,2048,4));sampling_seconds=0.;moment_seconds=0.
        for j,item in enumerate(polys):
            reg=np.frombuffer(bytes.fromhex(item['pattern']),np.uint8).reshape(4,4);_,_,matrix,rhs=model.pattern_matrix(x,v,reg)
            poly,note=model.posterior.polytope(matrix,rhs)
            assert poly is not None and np.isclose(poly['volume'],item['volume'],rtol=1e-9,atol=1e-30);checks['region_geometry_matches']+=1
            for r in range(4):
                begin=time.perf_counter();rng=np.random.default_rng(np.random.SeedSequence([p['rng_seed'],seed,j,r]))
                points=model.posterior.sample(poly,2048,rng);reference_points[r,j]=points;sampling_seconds+=time.perf_counter()-begin
                begin=time.perf_counter();mean[r,j],second[r,j]=moments(points,q);moment_seconds+=time.perf_counter()-begin
                checks['reference_batches']+=1;checks['reference_samples']+=len(points)
        full=np.einsum('k,rkq->rq',weights,mean);fullsecond=np.einsum('k,rkq->rq',weights,second)
        actual=np.empty((len(names),2,len(q)));masks=[];state_hashes=[]
        for j,name in enumerate(names):
            for rep in range(2):
                filename,digest=stages[seed,name,rep];path=online/filename;assert sha(path)==digest;checks['frozen_states']+=1
                with np.load(path) as z:
                    assert np.array_equal(x,z['x']) and np.array_equal(v,z['v']) and len(z['points'])==2048
                    actual[j,rep]=moments(z['points'],q)[0]
                    assert np.max(abs(actual[j,rep,::4]-z['prediction']))<1e-12;checks['online_prediction_replays']+=1
                state_hashes.append(dict(method=name,repetition=rep,state_file=filename,state_sha256=digest))
            found=next(c for c in row['coverage'] if c['method']==name);detail=online/found['detail_file'];assert sha(detail)==found['detail_sha256']
            masks.append([item['pattern'] in set(found['found_patterns']) for item in polys])
        masks=np.array(masks);task=[]
        for j,name in enumerate(names):
            w=weights*masks[j];mass=float(w.sum());w/=mass
            truncated=np.einsum('k,rkq->rq',w,mean);truncsecond=np.einsum('k,rkq->rq',w,second)
            estimates=[estimate(q,full,fullsecond,truncated,truncsecond,actual[j],pair,2048) for pair in pairs]
            selected=[e for e in estimates if e['pair'] in p['primary_pairs']];fields=list(selected[0]['fine'])
            task.append(dict(seed=seed,method=name,mass_fraction=mass,
                primary={key:float(np.mean([e['fine'][key] for e in selected])) for key in fields},
                all_pair_mean={key:float(np.mean([e['fine'][key] for e in estimates])) for key in fields},
                primary_grid_change={key:float(np.mean([e['fine'][key]-e['coarse'][key] for e in selected])) for key in fields},
                pair_estimates=estimates))
        filename=f'curves_{seed}.npz';file=out/filename
        np.savez_compressed(file,x=x,v=v,q=q,patterns=np.array([r['pattern'] for r in polys]),weights=weights,
            region_means=mean,region_seconds=second,reference_points=reference_points,actual_predictions=actual,method_masks=masks,methods=np.array(names))
        audit=dict(seed=seed,curve_file=filename,curve_sha256=sha(file),reference_sha256=row['reference_sha256'],frozen_states=state_hashes,
            region_count=k,sampling_seconds=sampling_seconds,moment_seconds=moment_seconds,elapsed_seconds=time.perf_counter()-start)
        audits.append(audit);rows.extend(task)
        for name,value in [('rows.json',rows),('audits.json',audits)]:
            (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=len(audits),total=len(p['seeds']),seed=seed,regions=k,seconds=audit['elapsed_seconds'],**checks)),flush=True)
    result=dict(execution_complete=True,tasks=len(audits),method_tasks=len(rows),sources=len(hashes),checks=checks,scope=p['scope'])
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
