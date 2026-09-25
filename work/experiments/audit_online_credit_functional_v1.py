"""326 independent piecewise particle readout and exact certificate audit."""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback
import numpy as np
from audit_post_escape_geometry_v1 import certificates
from audit_stasis_escape_geometry_v1 import inequalities
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def forward(points,q):
    z=np.broadcast_to(q,(len(points),len(q))).copy()
    for j in range(4):
        z=z+points[:,j,None]
        z=np.where((z<0)|(z>1),0,np.where(z<=.5,2*z,2-2*z))
    return z


def run(root,out):
    start=time.perf_counter();folder=root/'results/online_credit_branch_search/functional_v1'
    summary=read(folder/'summary.json');assert summary['passed']
    manifest=folder/'before_evaluation_manifest.json';assert sha(manifest)==summary['manifest_sha256']
    for n,d in read(manifest)['files_sha256'].items():assert sha(folder/n)==d
    protocol=read(folder/'protocol.json')
    for group in ['source_sha256','frozen_original_dependency_sha256']:
        for n,d in protocol[group].items():assert sha(root/'work/experiments'/n)==d
    raw=root/'results/certificate_activity_attribution/development'
    originals={r['seed']:r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    rows=read(folder/'rows.json');assert len(rows)==768
    assert {(r['seed'],r['policy'],r['channel']) for r in rows}=={(s,p,c) for s in range(5910000,5910064) for p in ['first_fit','uniform_state'] for c in ['dual','dual_plus_residual','residual','bp','random_sign','zero']}
    counts=Counter();maxima=Counter();checked={};pools={};old={};ledger=[]
    for i,row in enumerate(rows):
        seed=row['seed'];meta=read(folder/row['metadata_file'])
        if seed not in old:
            ref=originals[seed];assert sha(raw/ref['file'])==ref['sha256']
            with np.load(raw/ref['file'],allow_pickle=False) as z:old[seed]={n:z[n] for n in ['x_observed','v_observed','points','allocation','prediction']}
        src=old[seed];x,v=src['x_observed'],src['v_observed']
        with np.load(folder/row['file'],allow_pickle=False) as z:arrays={n:z[n] for n in ['points','allocation','prediction','point_prediction','selected_b']}
        points=arrays['points'];keys=meta['positive_modes'];allocation=arrays['allocation']
        assert points.shape==(2048,4) if keys else points.shape==(1,4)
        assert np.isfinite(points).all() and np.max(np.abs(points))<=.12+1e-7
        calc=forward(points,np.linspace(0,1,257)).mean(axis=0)
        error=float(np.max(np.abs(calc-arrays['prediction'])));assert error<=1e-12;maxima['readout_abs_error']=max(maxima['readout_abs_error'],error)
        calcpoint=forward(arrays['selected_b'][None],np.linspace(0,1,257))[0]
        error=float(np.max(np.abs(calcpoint-arrays['point_prediction'])));assert error<=1e-12;maxima['point_readout_abs_error']=max(maxima['point_readout_abs_error'],error)
        if keys:
            assert allocation.shape==(len(keys),) and np.all(allocation>=0) and sum(allocation)==2048
            support=float(np.max(abs(forward(points,x)-v)));assert support<=.001+1e-7
            maxima['sample_support_abs_error']=max(maxima['sample_support_abs_error'],support)
            offset=0
            for key,count in zip(keys,allocation):
                a,rhs=inequalities(x,v,key)
                if count:
                    aa=np.array(a,dtype=float);rr=np.array(rhs,dtype=float)
                    violation=float(np.max(points[offset:offset+count]@aa.T-rr));assert violation<=1e-7
                    maxima['sample_constraint_violation']=max(maxima['sample_constraint_violation'],violation)
                offset+=int(count);counts['sampled_region_allocations']+=1
            counts['support_checked_particles']+=len(points)
        for key,note in meta['new_mode_classifications_detail'].items():
            cache=seed,key
            if cache not in checked:
                a,rhs=inequalities(x,v,key);counts['unique_exact_certificates']+=certificates(a,rhs,note)
                checked[cache]=note;counts['unique_classifications']+=1
            else:
                # Runtime-only duration is not part of the mathematical certificate.
                assert note['classification']==checked[cache]['classification']
                assert note['certificates']==checked[cache]['certificates']
                assert note['volume']==checked[cache]['volume']
            counts['classification_instances']+=1
        poolkey=seed,tuple(keys)
        if poolkey in pools:
            for field in ['points','allocation','prediction']:assert arrays[field].tobytes()==pools[poolkey][field].tobytes();counts['same_pool_bitwise_arrays']+=1
        else:pools[poolkey]={f:arrays[f] for f in ['points','allocation','prediction']}
        if keys==meta['original_positive_modes']:
            for field in ['points','allocation','prediction']:assert arrays[field].tobytes()==src[field].tobytes();counts['unchanged_original_bitwise_arrays']+=1
        ledger.append(dict(seed=seed,policy=row['policy'],channel=row['channel'],selection_exact_forwards=meta['exact_trigger_forward_calls'],
                           finish_diagnostic_invocations=1,finish_exact_forwards=2,total_trigger_exact_forwards=meta['exact_trigger_forward_calls']+2,
                           source_field_trigger_diagnostic_forward_calls=meta['trigger_diagnostic_forward_calls']))
        assert meta['trigger_diagnostic_forward_calls']==1
        counts['calls']+=1;counts['query_coordinates']+=257;counts['point_query_coordinates']+=257
        if (i+1)%96==0:print(dict(calls=i+1,seconds=time.perf_counter()-start),flush=True)
    save(out/'forward_accounting_ledger.json',ledger)
    result=dict(passed=True,counts=dict(counts),maxima=dict(maxima),unique_pools=len(pools),seconds=time.perf_counter()-start,
        input_manifest_sha256=sha(manifest),source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'audit_post_escape_geometry_v1.py','audit_stasis_escape_geometry_v1.py']},
        protocol_sha256=sha(root/'outputs/ttt-pc-alm-research/326_readout_audit_protocol_v1.md'),
        outputs_sha256={'forward_accounting_ledger.json':sha(out/'forward_accounting_ledger.json')},
        query_targets_accessed=False,posterior_reference_accessed=False,numeric_sampling_tolerance=1e-7,
        resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
