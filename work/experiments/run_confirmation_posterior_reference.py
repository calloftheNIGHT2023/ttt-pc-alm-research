"""266A: bounded complete support reference for ALL evaluated confirmation tasks.

Post-result mechanism analysis, not a fresh blind confirmation. The runner reads
only frozen observed inputs and optimizer pools, never held-out answers/risks.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import conditioned_mode_geometry as corrected
import enumerate_support_modes as reference
import shared_mode_readout as shared
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    parent=root/'results/round_265_audit.json';old=json.loads(parent.read_text());assert old['passed'];hashes=dict(old['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    base=root/'results/matched_budget_confirmation';inp=base/'conditioned_confirmation';out=root/'results/confirmation_conditional_risk/reference';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    design=root/'outputs/ttt-pc-alm-research/266_confirmation_conditional_risk_protocol.md';p0=json.loads((inp/'protocol.json').read_text());rows=json.loads((inp/'rows.json').read_text());lookup={(r['seed'],r['method']):r for r in rows};methods=[c['name'] for c in p0['configs'] if lookup[p0['seeds'][0],c['name']]['readout']=='mode']
    before=json.loads((inp/'before_query_manifest.json').read_text());assert sha(inp/'rows.json')==before['rows_sha256']
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=sha(design),seeds=p0['seeds'],methods=methods,primary=p0['primary'],n_context=4,depth=4,max_lp_per_task=50000,max_search_seconds_per_task=180,
        frozen_predictor_manifest_sha256=sha(inp/'before_query_manifest.json'),phase_accesses_query_targets=False,
        scope='post-confirmation observation-only diagnosis, not new blind evidence; global exhaustive reference is not the online algorithm; numerical volumes not exact-real integrals')
    dump(out/'protocol.json',p);coverage=[];files={};begin=time.perf_counter();assert reference.memory.posterior is shared.geometry
    for seed in p['seeds']:
        savedrow=lookup[seed,p['primary']];path=inp/savedrow['file'];assert sha(path)==savedrow['sha256']==before['prediction_files'][savedrow['file']]
        with np.load(path) as z:x=z['x_observed'].copy();v=z['v_observed'].copy()
        captured={};repairs=[];refsolver=reference.Reference(x,v,max_lp=p['max_lp_per_task'],max_seconds=p['max_search_seconds_per_task']);start=time.perf_counter();failure=None
        with corrected.geometry_scope(repairs):
            guarded=reference.memory.posterior.polytope
            def materialize(g,rhs):
                poly,note=guarded(g,rhs)
                if poly is None:return poly,note
                point=poly['center']+poly['scale']*poly['interior'];pattern=reference.base.pattern(x,point);proof=shared.strict_interior(x,v,pattern,point)
                if not proof['accepted']:return None,dict(note,reason='unknown: exact interior failed')
                key=pattern.astype(np.uint8).tobytes().hex();assert key not in captured
                filename=f'{seed}_{key}.npz';file=out/filename;assert not file.exists();np.savez_compressed(file,g=g,original_rhs=rhs,**poly);files[filename]=sha(file)
                captured[key]=dict(file=filename,sha256=files[filename],proof=proof,note=note)
                return poly,note
            reference.memory.posterior.polytope=materialize
            try:
                ref=refsolver.enumerate()
            except corrected.suite.RECOVERABLE as exc:
                failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
                ref=dict(enumeration_completed=False,numerical_volume_reference_complete=False,lp_calls=refsolver.lp_calls,
                    certified_lp_rejections=len(refsolver.certificates),certificates=refsolver.certificates,positive_regions=[],final_geometry_unresolved=[],error=failure)
            finally:
                reference.memory.posterior.polytope=guarded
        filename=f'{seed}_reference.json';dump(out/filename,dict(seed=seed,x_observed=x.tolist(),v_observed=v.tolist(),reference=ref,geometry=captured,repair_log=repairs));files[filename]=sha(out/filename)
        item=dict(seed=seed,file=filename,sha256=files[filename],complete=ref['numerical_volume_reference_complete'],enumeration_completed=ref['enumeration_completed'],
            positive_regions=len(ref['positive_regions']),unresolved_final_regions=len(ref['final_geometry_unresolved']),lp_calls=ref['lp_calls'],certified_rejections=ref['certified_lp_rejections'],
            geometry_repairs=len(repairs),failure=failure,seconds=time.perf_counter()-start,coverage=[])
        if item['complete']:
            polys={r['pattern']:r for r in ref['positive_regions']};assert set(polys)==set(captured);total=sum(r['volume'] for r in polys.values());assert total>0;item['total_numerical_volume']=total
            for name in methods:
                row=lookup[seed,name];assert not row['metadata']['execution_failed'];found=set(row['metadata']['positive_modes']);assert found<=set(polys),(seed,name,sorted(found-set(polys)))
                volume=sum(polys[k]['volume'] for k in sorted(found));item['coverage'].append(dict(method=name,found_modes=sorted(found),missing_modes=sorted(set(polys)-found),
                    discovered_volume=volume,numerical_posterior_mass_fraction=volume/total))
        coverage.append(item);dump(out/'coverage.json',coverage)
        print(json.dumps(dict(tasks_done=len(coverage),total=64,seed=seed,complete=item['complete'],positive_regions=item['positive_regions'],lp_calls=item['lp_calls'],seconds=item['seconds'],failures=sum(r['failure'] is not None for r in coverage))),flush=True)
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'files.json',files);ans=dict(passed=True,tasks=len(coverage),complete_references=sum(r['complete'] for r in coverage),can_use_complete_posterior=all(r['complete'] for r in coverage),
        failures=sum(r['failure'] is not None for r in coverage),positive_regions=sum(r['positive_regions'] for r in coverage),lp_calls=sum(r['lp_calls'] for r in coverage),
        certified_rejections=sum(r['certified_rejections'] for r in coverage),seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),coverage_sha256=sha(out/'coverage.json'),files_sha256=sha(out/'files.json'),
        phase_accesses_query_targets=False,next='independent exact rejection/interior/frozen-pool audit; do not use incomplete tasks as a complete posterior')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
