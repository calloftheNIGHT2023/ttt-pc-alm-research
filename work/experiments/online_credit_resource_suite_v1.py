"""327 unchanged online candidate and all37 old baselines plus longer Adam."""
import time
import numpy as np
import counterfactual_resource_suite_v1 as old
import online_credit_branch_search_v1 as candidate
import conditioned_mode_geometry as conditioned
from evaluate_complete_credit_mode_geometry_v1 import read,sha

PRIMARY='online_first_fit_dual'
SECONDARY='online_uniform_state_dual_plus_residual'
BASE='results/online_credit_resources'
SOURCES=['online_credit_resource_suite_v1.py','calibrate_online_credit_resources_v1.py','online_credit_branch_search_v1.py',
         'support_consistency_trigger_v1.py','branch_image_chain_v1.py','factorized_dual_branch_search_v1.py',
         'run_factorized_dual_branch_search_v1.py','complete_credit_mode_geometry_v1.py','conditioned_mode_geometry.py','audit_online_credit_resources_v1.py']


def catalogue(root):
    configs=old.catalogue(root)
    for p in candidate.POLICIES:
        for c in candidate.CHANNELS:
            configs.append(dict(name=f'online_{p}_{c}',family='online_credit',group='online_'+p,policy=p,channel=c))
    for steps in [1920,3840]:
        name=f'probe_then_adam{steps}_33'
        configs.append(dict(name=name,family='probe',group='probe_adam',config=dict(name=name,atomic='branch_probe',solver='adam',certificate=False,action='Q',steps=steps)))
    assert len(configs)==len({c['name'] for c in configs})==51
    return configs


def gate(root):
    hashes=old.gate(root)
    base=root/'results/online_credit_branch_search'
    functional=read(base/'functional_v1/summary.json');audit=read(base/'functional_audit_v1/summary.json')
    assert functional['passed'] and audit['passed'] and audit['counts']['calls']==768
    assert sha(base/'functional_v1/before_evaluation_manifest.json')==audit['input_manifest_sha256']==functional['manifest_sha256']
    for n,d in read(base/'functional_v1/protocol.json')['source_sha256'].items():assert sha(root/'work/experiments'/n)==d
    for n in SOURCES:hashes[n]=sha(root/'work/experiments'/n)
    return hashes


def invoke(cfg,x,v,q,seed,loaded):
    repairs=[]
    def runner(c,xx,vv,qq,ss,ll):
        if c['family']=='online_credit':
            a,m=candidate.fit(xx,vv,qq,ss,policy=c['policy'],channel=c['channel'],trace=False)
            tick=time.perf_counter()
            for field in old.FIELDS:a[field]=old.project(a[field])
            m['range_projection_seconds']=time.perf_counter()-tick
            return a,m
        with conditioned.geometry_scope(repairs):return old.fit(c,xx,vv,qq,ss,ll,trace=False)
    tick=time.perf_counter()
    a,m=old.resources.legacy.guarded_fit(cfg,x,v,q,seed,loaded,runner=runner)
    if m['execution_failed']:
        for f in old.FIELDS:a[f]=old.project(a[f])
    seconds=time.perf_counter()-tick
    m.update(geometry_repair_log=repairs,charged_complete_seconds=seconds,query_targets_accessed=False)
    return a,m,seconds


def check_reference(root,cfg,seed,arrays,meta,index):
    if meta['execution_failed']:return 0
    if cfg['family']!='online_credit':return old.check_frozen(seed,cfg,arrays,meta,index)
    folder=root/'results/online_credit_branch_search/functional_v1'
    filename=f"{seed}_{cfg['policy']}_{cfg['channel']}.npz";count=0
    sealed=read(folder/'before_evaluation_manifest.json')['files_sha256']
    assert sha(folder/filename)==sealed[filename]
    assert sha(folder/filename.replace('.npz','.json'))==sealed[filename.replace('.npz','.json')]
    with np.load(folder/filename,allow_pickle=False) as z:
        for n,value in arrays.items():
            assert n in z.files
            expected=old.project(z[n]) if n in old.FIELDS else z[n]
            assert value.shape==expected.shape and value.dtype==expected.dtype and value.tobytes()==expected.tobytes(),(seed,cfg['name'],n)
            count+=1
    expected=read(folder/filename.replace('.npz','.json'))
    assert meta['positive_modes']==expected['positive_modes']
    assert not meta['trace_enabled']
    return count
