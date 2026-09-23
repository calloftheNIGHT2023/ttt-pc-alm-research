"""Full deterministic particle replay and independent piecewise forward moments."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import numpy as np
import shared_mode_readout as shared
from audit_local_dual_jump_modes import modes
from run_multiplier_fixed_point_screen import sha,dump


def piecewise_forward(q,points):
    h=np.broadcast_to(q,(len(points),len(q)))
    for j in range(4):
        z=h+points[:,j,None]
        h=np.maximum(0.,np.where(z<=.5,2*z,2-2*z))
    return h


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/confirmation_conditional_risk';inp=base/'moments';refdir=base/'reference';out=base/'moments_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    summary=json.loads((inp/'summary.json').read_text());p=json.loads((inp/'protocol.json').read_text());assert summary['passed'] and not summary['phase_accesses_query_targets'];hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    for n in ['protocol','rows','tasks','files']:assert sha(inp/f'{n}.json')==summary[f'{n}_sha256']
    dump(out/'protocol.json',dict(source_sha256=hashes,moments_summary_sha256=sha(inp/'summary.json'),phase_accesses_query_targets=False,
        scope='all6376 deterministic particle batches, independent branch-form forward on all257 q, moments and all64 aggregate files'))
    rows=json.loads((inp/'rows.json').read_text());tasks=json.loads((inp/'tasks.json').read_text());files=json.loads((inp/'files.json').read_text());lookup={(r['seed'],r['key'],r['batch']):r for r in rows};assert len(rows)==len(lookup)==6376
    refs={r['seed']:r for r in json.loads((refdir/'coverage.json').read_text())};counts=Counter();maxmean=0.;maxsecond=0.;maxsupport=0.;begin=time.perf_counter();gaps=[]
    for task in tasks:
        path=inp/task['file'];assert sha(path)==task['sha256']==files[task['file']];refrow=refs[task['seed']];assert sha(refdir/refrow['file'])==refrow['sha256'];ref=json.loads((refdir/refrow['file']).read_text());x=np.array(ref['x_observed']);v=np.array(ref['v_observed']);regions={r['pattern']:r for r in ref['reference']['positive_regions']}
        assert task['keys']==sorted(regions)
        with np.load(path) as a:
            q=a['q'];assert q.tobytes()==np.linspace(0,1,257).tobytes();assert a['means'].shape==a['seconds'].shape==(len(task['keys']),4,257)
            for index,key in enumerate(task['keys']):
                item=ref['geometry'][key];assert sha(refdir/item['file'])==item['sha256']
                with np.load(refdir/item['file']) as z:poly={k:z[k].copy() for k in ['center','scale','interior','facets','simplex_probs','volume','a','rhs']}
                assert float(poly['volume'])==regions[key]['volume']==a['volumes'][index]
                for batch in range(4):
                    r=lookup[task['seed'],key,batch];assert r['region_index']==index;file=inp/r['file'];assert sha(file)==r['sha256']==files[r['file']]
                    expected=shared.geometry.sample(poly,2048,np.random.default_rng(np.random.SeedSequence([266911,task['seed'],index,batch,2048])))
                    with np.load(file) as z:
                        assert expected.tobytes()==z['points'].tobytes();assert z['mean'].tobytes()==a['means'][index,batch].tobytes() and z['second'].tobytes()==a['seconds'][index,batch].tobytes()
                        h=piecewise_forward(q,z['points']);mu=h.sum(axis=0)/len(h);second=np.einsum('ij,ij->j',h,h)/len(h);gm=float(np.max(abs(mu-z['mean'])));gs=float(np.max(abs(second-z['second'])));assert gm<1e-12 and gs<1e-12;maxmean=max(maxmean,gm);maxsecond=max(maxsecond,gs)
                        support=float(np.max(abs(piecewise_forward(x,z['points'])-v)));assert support<=.001+1e-7;maxsupport=max(maxsupport,support);assert set(modes(x,z['points']))=={key}
                    gaps.append(dict(seed=task['seed'],key=key,batch=batch,mean_gap=gm,second_gap=gs));counts['particle_batches_replayed']+=1;counts['all_grid_values_checked']+=2048*257;counts['support_and_membership_particles']+=2048;counts['regional_moment_vectors']+=2
                counts['geometry_regions']+=1
        counts['task_aggregate_files']+=1
        print(json.dumps(dict(audited_tasks=counts['task_aggregate_files'],total=64,batches=counts['particle_batches_replayed'],seconds=time.perf_counter()-begin)),flush=True)
    assert counts['support_and_membership_particles']==summary['particles'];dump(out/'gaps.json',gaps);ans=dict(passed=True,counts=counts,max_independent_mean_gap=maxmean,max_independent_second_gap=maxsecond,max_support_particle_error=maxsupport,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),gaps_sha256=sha(out/'gaps.json'),phase_accesses_query_targets=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
