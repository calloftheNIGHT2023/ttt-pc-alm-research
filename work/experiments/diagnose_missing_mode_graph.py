"""Support-only structural diagnostic; complete reference never feeds a learner."""
import argparse,hashlib,json
from collections import deque
from pathlib import Path
import numpy as np


def distances(adjacency,sources):
    result=np.full(len(adjacency),-1,int);queue=deque(map(int,np.flatnonzero(sources)))
    result[sources]=0
    while queue:
        i=queue.popleft()
        for j in np.flatnonzero(adjacency[i]&(result<0)):
            result[j]=result[i]+1;queue.append(int(j))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'missing_mode_graph'
    prior=json.loads((a.root/'archive_conditional_risk/protocol.json').read_text());states=json.loads((a.root/'archive_conditional_risk/risks.json').read_text())
    refs=json.loads((a.root/'first_write_reference/coverage.json').read_text())
    for name,h in prior['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    protocol=dict(seeds=prior['seeds'],methods=[c['name'] for c in prior['configs']],
        adjacency='one differing layer/observation activation code; second graph requires absolute code difference one',
        predeclared='all missing nodes: direct Hamming distance and positive-volume feasible-graph shortest path; no query inputs or targets',
        source_sha256={**prior['source_sha256'],Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='diagnostic with complete numerical reference; does not authorize oracle-guided online branch proposals')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[]
    for ref in refs:
        seed=ref['seed'];path=a.root/'first_write_reference'/ref['reference_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['reference_sha256']
        full=json.loads(path.read_text());assert full['numerical_volume_reference_complete']
        cells=sorted(full['positive_regions'],key=lambda r:r['pattern']);keys=[c['pattern'] for c in cells]
        patterns=np.array([np.frombuffer(bytes.fromhex(k),dtype=np.uint8) for k in keys]);w=np.array([c['volume'] for c in cells]);w/=w.sum()
        hamming=np.sum(patterns[:,None]!=patterns[None,:],axis=-1);difference=np.abs(patterns[:,None].astype(int)-patterns[None,:].astype(int)).sum(-1)
        adjacent=hamming==1;local_adjacent=adjacent&(difference==1)
        np.savez_compressed(out/f'graph_{seed}.npz',patterns=patterns,weights=w,hamming=hamming,adjacent=adjacent,local_adjacent=local_adjacent)
        for state in [r for r in states if r['seed']==seed]:
            path=a.root/'archive_conditional_risk'/state['state_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==state['state_sha256']
            with np.load(path) as z:found=set(z['found_patterns'])
            mask=np.array([k in found for k in keys]);assert mask.any() and sum(mask)==state['positive_regions']
            nearest=hamming[:,mask].min(1);steps=distances(adjacent,mask);localsteps=distances(local_adjacent,mask);missing=~mask
            details=[]
            for i in np.flatnonzero(missing):
                details.append(dict(pattern=keys[i],posterior_mass=float(w[i]),nearest_hamming=int(nearest[i]),
                    feasible_graph_distance=int(steps[i]),adjacent_branch_graph_distance=int(localsteps[i])))
            rows.append(dict(seed=seed,method=state['method'],positive_modes=len(keys),found_modes=int(mask.sum()),missing_modes=int(missing.sum()),
                found_mass=float(w[mask].sum()),missing_mass=float(w[missing].sum()),
                direct_single_flip_mass=float(w[missing&(nearest==1)].sum()),
                direct_adjacent_flip_mass=float(w[missing&(localsteps==1)].sum()),
                positive_graph_reachable_missing_mass=float(w[missing&(steps>=1)].sum()),
                adjacent_graph_reachable_missing_mass=float(w[missing&(localsteps>=1)].sum()),
                positive_graph_unreachable_missing_mass=float(w[missing&(steps<0)].sum()),
                maximum_nearest_hamming=int(nearest[missing].max()) if missing.any() else 0,
                maximum_reachable_graph_distance=int(steps.max()),missing_details=details))
    summary=[]
    fields=['found_mass','missing_mass','direct_single_flip_mass','direct_adjacent_flip_mass','positive_graph_reachable_missing_mass','adjacent_graph_reachable_missing_mass','positive_graph_unreachable_missing_mass']
    for method in protocol['methods']:
        rr=[r for r in rows if r['method']==method];assert len(rr)==16
        summary.append(dict(method=method,**{k:float(np.mean([r[k] for r in rr])) for k in fields},
            missing_modes_total=sum(r['missing_modes'] for r in rr),tasks_with_missing_modes=sum(r['missing_modes']>0 for r in rr),
            tasks_all_missing_one_flip=sum(all(d['nearest_hamming']==1 for d in r['missing_details']) for r in rr),
            tasks_all_missing_reachable=sum(all(d['feasible_graph_distance']>=0 for d in r['missing_details']) for r in rr),
            maximum_nearest_hamming=max(r['maximum_nearest_hamming'] for r in rr),maximum_reachable_graph_distance=max(r['maximum_reachable_graph_distance'] for r in rr)))
    (out/'diagnostics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),summary=summary),indent=2))


if __name__=='__main__':main()
