"""293 fixed37 projected predictors, original support only, no evaluator."""
import os
from pathlib import Path
import time
import numpy as np
import psutil
import probe_credit_confirmation_suite as confirmation
import probe_credit_resource_suite as resources
import counterfactual_credit_branching_v1 as candidate
from verify_known_range_projection_v1 import project
from diagnose_gradient_flat_split_states_v1 import read, save, sha

PRIMARY = 'counterfactual_mode_change_alm'
SEEDS = [5910000,5910001,5910008,5910016,5910032,5910048,5910053,5910063]
PREFLIGHT_SEEDS = [5910000,5910001,5910053,5910063]
NEW_LOCAL = ['probe_alm40_33','probe_alm48_33','probe_alm64_33']
FIELDS = ['prediction','point_prediction']
NEW_SOURCES = ['counterfactual_resource_suite_v1.py','verify_counterfactual_resources_v1.py',
               'calibrate_counterfactual_resources_v1.py','audit_counterfactual_resources_v1.py']


def catalogue(root):
    cfgs = confirmation.catalogue(root)
    byname = {c['name']:c for c in cfgs}
    for cfg in resources.catalogue():
        if cfg['name'] in byname:
            assert cfg == byname[cfg['name']], cfg['name']
        else:
            byname[cfg['name']] = cfg
            cfgs.append(cfg)
    assert len(cfgs)==33
    for steps in [40,48,64]:
        name = f'probe_alm{steps}_33'
        cfgs.append(dict(name=name, family='local', group='extended_same_probe_alm',
            config=dict(name=name, atomic='branch_probe', solver='alm', certificate=False,
                        action='Q', prefix=steps, extra=steps)))
    cfgs.insert(0, dict(name=PRIMARY, family='counterfactual', group='new_candidate',
                       rule='changed_forward_mode'))
    assert len(cfgs)==len({c['name'] for c in cfgs})==37
    return cfgs


def fit(cfg, x, v, q, seed, loaded, trace=False):
    if cfg['family']=='counterfactual':
        a,m = candidate.fit(x,v,q,seed,rule=cfg['rule'],trace=trace)
    else:
        a,m = resources.fit(cfg,x,v,q,seed,loaded,trace=trace)
    begin=time.perf_counter()
    for field in FIELDS:
        a[field]=project(a[field])
    m.update(range_projection_seconds=time.perf_counter()-begin,
             projected_output_array_bytes=sum(a[f].nbytes for f in FIELDS),
             projected_output_scope='Two output arrays only, not peak memory',
             query_targets_accessed=False)
    return a,m


def complete(folder):
    s=read(folder/'summary.json')
    assert s['passed'], folder
    for name,digest in s.get('outputs_sha256',{}).items():
        assert sha(folder/name)==digest,(folder,name)
    return s


def gate(root):
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    old=read(root/'results/round_287_audit_v6.json')
    assert old['passed']
    hashes=dict(old['source_sha256'])
    for name,digest in hashes.items():
        assert sha(root/'work/experiments'/name)==digest,name
    f=root/'results/counterfactual_credit_branching/full_support_v2'
    full=complete(f)
    assert full['tasks']==64 and full['exact_shadow_rows']==17362 and not full['query_quality_evaluated']
    fp=read(f/'protocol.json')
    for name,digest in fp['source_sha256'].items():
        assert sha(root/'work/experiments'/name)==digest
        hashes[name]=digest
    extra=root/'results/known_range_projection'
    a=complete(extra/'audit_confirmation_v1')
    assert a['tasks']==8192 and a['comparisons']==104
    finished=read(extra/'pipeline_execution_v1/summary.json')
    assert finished['passed'] and finished['stages_completed']==6
    for n in NEW_SOURCES+['counterfactual_credit_branching_v1.py','verify_known_range_projection_v1.py',
                          'diagnose_gradient_flat_split_states_v1.py']:
        hashes[n]=sha(root/'work/experiments'/n)
    return hashes


def frozen_inputs(root):
    index=resources.frozen_inputs(root)
    for folder,filename in [
        ('probe_credit_resources/calibration','timings.json'),
        ('probe_credit_budget/development','rows.json'),
        ('probe_credit_budget_sensitivity/development','rows.json'),
        ('probe_credit_confirmation/head_preflight','primitive_rows.json')]:
        directory=root/'results'/folder
        for row in read(directory/filename):
            index.setdefault((row['seed'],row['method']),(directory,row))
    directory=root/'results/counterfactual_credit_branching/full_support_v2'
    for row in read(directory/'rows.json'):
        item=dict(row,method=PRIMARY,file=row['output_file'],sha256=row['output_sha256'])
        index[row['seed'],PRIMARY]=(directory,item)
    return index


def observed_inputs(root,index,seeds):
    data={}
    for seed in seeds:
        folder,row=index[seed,resources.PRIMARY]
        assert sha(folder/row['file'])==row['sha256']
        with np.load(folder/row['file'],allow_pickle=False) as z:
            data[seed]=(z['x_observed'].copy(),z['v_observed'].copy(),np.linspace(0,1,257))
    return data


def check_frozen(seed,cfg,a,m,index):
    item=index.get((seed,cfg['name']))
    if item is None:
        return 0
    directory,row=item
    assert sha(directory/row['file'])==row['sha256']
    count=0
    with np.load(directory/row['file'],allow_pickle=False) as z:
        for key in ['prediction','point_prediction','points','allocation','selected_b','best_bank',
                    'initial_b','initial_h','initial_u','atomic_trial_b']:
            if key in a and key in z:
                expected=project(z[key]) if key in FIELDS else z[key]
                assert a[key].shape==expected.shape and a[key].dtype==expected.dtype
                assert a[key].tobytes()==expected.tobytes(),(seed,cfg['name'],key)
                count+=1
    if 'positive_modes' in m:
        assert m['positive_modes']==row['metadata']['positive_modes']
    return count


def environment_snapshot(root):
    others=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time']):
        try:
            if p.pid==os.getpid():continue
            name=(p.info['name'] or '').lower()
            cmd=' '.join(p.info['cmdline'] or []).replace('\\','/').lower()
            if (name.startswith('python') and ('work/experiments/' in cmd or str(root).replace('\\','/').lower() in cmd)) or name in ['git-pack-objects.exe','git-index-pack.exe']:
                others.append(dict(pid=p.pid,name=name,create_time=p.info['create_time']))
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    return dict(unix_time=time.time(),cpu_percent=psutil.cpu_percent(interval=.1),
                memory=psutil.virtual_memory()._asdict(),other_research_or_git_pack_processes=others)


def select_budget(configs,means):
    cap=means[PRIMARY]
    within=[c['name'] for c in configs if means[c['name']]<=cap]
    background=[]
    for group in sorted({c['group'] for c in configs}):
        group_configs=[c for c in configs if c['group']==group]
        if not any(c['name'] in within for c in group_configs):
            background.append(min(group_configs,key=lambda c:(means[c['name']],c['name']))['name'])
    return dict(primary=PRIMARY,budget_seconds=cap,budget_factor=1.0,sensitivity_factor=1.10,
        within_budget=within,over_budget_background=background,
        sensitivity_110_percent=[c['name'] for c in configs if means[c['name']]<=cap*1.1],
        all_methods_retained=True,query_quality_used=False)
