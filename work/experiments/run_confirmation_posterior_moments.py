"""266B fixed four-batch regional function moments; never uses query answers."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import shared_mode_readout as shared
from audit_local_dual_jump_modes import modes
from analyze_recovered_online_comparison import forward
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/confirmation_conditional_risk';inp=base/'reference';audit=base/'reference_audit';out=base/'moments';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    aa=json.loads((audit/'summary.json').read_text());assert aa['passed'] and aa['complete_up_to_certified_zero_volume']==64
    hashes=dict(json.loads((audit/'protocol.json').read_text())['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    design=root/'outputs/ttt-pc-alm-research/266_confirmation_conditional_risk_protocol.md';refp=json.loads((inp/'protocol.json').read_text());assert sha(design)==refp['design_sha256']
    coverage=json.loads((audit/'coverage.json').read_text());assert sha(audit/'coverage.json')==aa['coverage_sha256'];old=root/'results/matched_budget_confirmation/conditioned_confirmation';before=json.loads((old/'before_query_manifest.json').read_text());assert sha(old/'before_query_manifest.json')==refp['frozen_predictor_manifest_sha256']
    rows0=json.loads((old/'rows.json').read_text());lookup={(r['seed'],r['method']):r for r in rows0};reference_rows={r['seed']:r for r in json.loads((inp/'coverage.json').read_text())}
    p=dict(source_sha256=hashes,design_sha256=sha(design),reference_audit_sha256=sha(audit/'summary.json'),seeds=refp['seeds'],batches=4,particles_per_region_batch=2048,seed_prefix=266911,query_points=257,
        batch_pairs=[[0,1],[2,3]],phase_accesses_query_targets=False,scope='post-confirmation posterior diagnostic only; fixed numerical volumes, shared regional moments and unlabelled grid; no method or prediction changes')
    dump(out/'protocol.json',p);begin=time.perf_counter();rows=[];files={};tasks=[];maxsupport=0.
    for task in coverage:
        seed=task['seed'];row=reference_rows[seed];assert sha(inp/row['file'])==row['sha256'];ref=json.loads((inp/row['file']).read_text());x=np.array(ref['x_observed']);v=np.array(ref['v_observed'])
        original=lookup[seed,refp['primary']];assert sha(old/original['file'])==original['sha256']==before['prediction_files'][original['file']]
        with np.load(old/original['file']) as z:q=z['q_observed'].copy();assert x.tobytes()==z['x_observed'].tobytes() and v.tobytes()==z['v_observed'].tobytes()
        assert q.tobytes()==np.linspace(0,1,257).tobytes();regions={r['pattern']:r for r in ref['reference']['positive_regions']};keys=sorted(regions);means=np.empty((len(keys),4,257));seconds=np.empty_like(means);volumes=np.array([regions[k]['volume'] for k in keys]);assert len(keys)==task['positive_regions']
        for index,key in enumerate(keys):
            item=ref['geometry'][key];path=inp/item['file'];assert sha(path)==item['sha256']
            with np.load(path) as z:poly={k:z[k].copy() for k in ['center','scale','interior','facets','simplex_probs','volume','a','rhs']}
            assert float(poly['volume'])==volumes[index]
            for batch in range(4):
                rng=np.random.default_rng(np.random.SeedSequence([266911,seed,index,batch,2048]));points=shared.geometry.sample(poly,2048,rng)
                support=float(np.max(abs(forward(x,points)-v)));assert support<=.001+1e-7;maxsupport=max(maxsupport,support);assert set(modes(x,points))=={key}
                values=forward(q,points);assert np.min(values)>=0 and np.max(values)<=1
                mean=values.mean(0);second=(values*values).mean(0);means[index,batch]=mean;seconds[index,batch]=second
                filename=f'{seed}_{key}_b{batch}.npz';path=out/filename;assert not path.exists();np.savez_compressed(path,points=points,mean=mean,second=second);files[filename]=sha(path)
                rows.append(dict(seed=seed,key=key,region_index=index,batch=batch,file=filename,sha256=files[filename],max_support_error=support))
            if len(rows)%80==0:dump(out/'rows.json',rows);print(json.dumps(dict(region_batches_done=len(rows),total=4*aa['counts']['positive_volume_regions'],current_seed=seed,seconds=time.perf_counter()-begin)),flush=True)
        filename=f'{seed}_moments.npz';path=out/filename;assert not path.exists();np.savez_compressed(path,q=q,volumes=volumes,means=means,seconds=seconds);files[filename]=sha(path)
        tasks.append(dict(seed=seed,file=filename,sha256=files[filename],keys=keys,positive_regions=len(keys)));dump(out/'tasks.json',tasks);dump(out/'rows.json',rows)
        print(json.dumps(dict(tasks_done=len(tasks),total=64,region_batches_done=len(rows),seconds=time.perf_counter()-begin)),flush=True)
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'files.json',files);ans=dict(passed=True,tasks=len(tasks),regional_batches=len(rows),particles=len(rows)*2048,positive_regions=sum(t['positive_regions'] for t in tasks),max_support_particle_error=maxsupport,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),tasks_sha256=sha(out/'tasks.json'),files_sha256=sha(out/'files.json'),phase_accesses_query_targets=False,
        next='independent regional moment audit and fixed46-predictor conditional excess-risk decomposition; do not read query answers or change predictions')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
