"""307 fixed46 catalogue and support-only complete-call dispatch."""
from pathlib import Path
import time
import numpy as np
import counterfactual_resource_suite_v1 as old
import conditioned_mode_geometry as conditioned
import online_stasis_memory_v2 as candidate
import online_stasis_controls_v2 as controls
from diagnose_gradient_flat_split_states_v1 import read,save,sha

PRIMARY='online_stasis_alm_keep64'
PILOT_SEEDS=list(range(307000000,307000256))
OLD_SEEDS=[5910000,5910001,5910048]
DESIGN='outputs/ttt-pc-alm-research/307_online_stasis_fresh_pilot_protocol.md'
BASE='results/online_stasis_fresh_pilot'
SOURCES=['online_stasis_fresh_suite_v1.py','online_stasis_fresh_pilot_v1.py',
         'test_online_stasis_fresh_pilot_v1.py','continue_online_stasis_fresh_pilot_v1.ps1']
FIELDS=['prediction','point_prediction']


def catalogue(root):
    cfgs=old.catalogue(root)
    cfgs.append(dict(name=PRIMARY,family='fresh_online_candidate',group='online_stasis',rule='forward_stasis'))
    for family in controls.FAMILIES:
        if family!='alm_keep':
            cfgs.append(dict(name=f'online_stasis_{family}64',family='fresh_online_control',
                             group='same_trigger_control',shadow_family=family))
    assert len(cfgs)==len({c['name'] for c in cfgs})==46
    assert cfgs[37]['name']==PRIMARY
    return cfgs


def gate(root):
    hashes=old.gate(root)
    c=root/'results/online_stasis_resources/calibration_v1'
    a=root/'results/online_stasis_resources/audit_v1'
    f=root/'results/online_stasis_controls/functional_v2'
    b=root/'results/online_stasis_controls/audit_v1'
    assert sha(c/'summary.json')=='03367696c3d517a10ec8be5a228772594c30e7a20e361de57060cc2b52961dd1'
    assert sha(a/'summary.json')=='2ea6a36bf0a4cda46591d3858c43093ee0a51560668adaae47f4444672d26e59'
    assert sha(f/'summary.json')=='1bd3a6e0500ef79d09e53d0542bc51e7fced49ced34ea2d7792a85f5dc0c9cd7'
    assert sha(b/'summary.json')=='4ca1a5b46fc3dfeacc8a8f8b893fae42bc30e89cb75fdc06273f8145df9a9fdb'
    assert old.complete(a)['failed_timing_calls']==0 and old.complete(b)['counts']['matching_pools']==576
    for folder in [c,f]:
        for name,digest in read(folder/'protocol.json')['source_sha256'].items():
            assert sha(root/'work/experiments'/name)==digest,name
            hashes[name]=digest
    for name in SOURCES+['probe_confirmation_statistics.py','run_independent_hybrid_memory.py',
                         'analyze_recovered_online_comparison.py','counterfactual_fresh_pilot_v1.py']:
        hashes[name]=sha(root/'work/experiments'/name)
    return hashes


def invoke(cfg,x,v,q,seed,loaded):
    repairs=[];start=time.perf_counter()
    def runner(c,xx,vv,qq,ss,ll):
        if c['family']=='fresh_online_candidate':a,m=candidate.fit(xx,vv,qq,ss,trace=False)
        elif c['family']=='fresh_online_control':a,m=controls.fit(xx,vv,qq,ss,family=c['shadow_family'],trace=False)
        else:return old.fit(c,xx,vv,qq,ss,ll,trace=False)
        tick=time.perf_counter()
        for field in FIELDS:a[field]=old.project(a[field])
        m.update(range_projection_seconds=time.perf_counter()-tick,
                 projected_output_array_bytes=sum(a[field].nbytes for field in FIELDS))
        return a,m
    if cfg['family'].startswith('fresh_online'):
        a,m=old.resources.legacy.guarded_fit(cfg,x,v,q,seed,loaded,runner=runner)
    else:
        with conditioned.geometry_scope(repairs):
            a,m=old.resources.legacy.guarded_fit(cfg,x,v,q,seed,loaded,runner=runner)
        m.update(geometry_repair_log=repairs,geometry_repair_count=len(repairs))
    if m['execution_failed']:
        for field in FIELDS:a[field]=old.project(a[field])
    for field in FIELDS:
        assert a[field].shape==q.shape and np.isfinite(a[field]).all()
        assert np.all((a[field]>=0)&(a[field]<=1))
    m.update(charged_complete_seconds=time.perf_counter()-start,query_targets_accessed=False)
    return a,m


def references(root):
    cal=root/'results/online_stasis_resources/calibration_v1'
    functional=root/'results/online_stasis_controls/functional_v2'
    index={}
    for r in read(cal/'timings.json'):index.setdefault((r['seed'],r['method']),(cal,r))
    for r in read(functional/'rows.json'):
        if r['method']!='alm_keep':index[r['seed'],f'online_stasis_{r["method"]}64']=(functional,r)
    assert all((s,c['name']) in index for s in OLD_SEEDS for c in catalogue(root))
    return index


def check_reference(root,row,arrays,index):
    directory,r=index[row['seed'],row['method']]
    assert sha(directory/r['file'])==r['sha256']
    assert not row['metadata']['execution_failed']
    count=0
    with np.load(directory/r['file'],allow_pickle=False) as z:
        for key in z.files:
            expected=old.project(z[key]) if key in FIELDS else z[key]
            assert arrays[key].shape==expected.shape and arrays[key].dtype==expected.dtype
            assert arrays[key].tobytes()==expected.tobytes(),(row['seed'],row['method'],key)
            count+=1
    if 'positive_modes' in row['metadata']:
        assert row['metadata']['positive_modes']==r['metadata']['positive_modes']
    return count
