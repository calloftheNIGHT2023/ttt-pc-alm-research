"""285 fixed25 complete implementations, inputs only; no evaluator access."""
import numpy as np
import probe_continuation_credit as probe
import probe_credit_budget_local as local
import matched_budget_suite as legacy
PRIMARY=probe.PRIMARY


def catalogue():
    old={c['name']:c for c in legacy.original.configs()};prior={c['name']:c for c in legacy.catalogue()}
    ans=[dict(name=PRIMARY,family='local',group='candidate',config={**probe.GOLD,'prefix':32,'extra':32}),dict(name='probe_archive_only',family='probe',group='archive',config=probe.CONFIGS[0])]
    for solver in ['pc','nodual']:
        for steps in [32,64,128,256]:
            name=f'probe_then_{solver}33' if steps==32 else f'probe_{solver}{steps}_33'
            ans.append(dict(name=name,family='local',group='probe_'+solver,config=dict(name=name,atomic='branch_probe',solver=solver,certificate=False,action='Q',prefix=steps,extra=steps)))
    for steps in [60,120,240,480,960]:
        name=f'probe_then_adam{steps}_33';ans.append(dict(name=name,family='probe',group='probe_adam',config=dict(name=name,atomic='branch_probe',solver='adam',certificate=False,action='Q',steps=steps)))
    for steps in [32,64,128,256]:
        name='credit_control_alm33' if steps==32 else f'plain_alm{steps}_33'
        ans.append(dict(name=name,family='local',group='plain_alm',config=dict(name=name,atomic='alm1',solver='alm',certificate=False,action='L',prefix=steps,extra=steps)))
    ans.append(dict(name='cold__adam240_r33',family='legacy',group='cold_adam',config=prior['cold__adam240_r33']))
    for size in [4096,16384,65536]:
        name=f'cold__prior{size}_ridge';ans.append(dict(name=name,family='prior',group='ridge',features=size))
    for name,group in [('cold__meta_shallow64_20','shallow'),('probe_all_alm64','full_probe')]:ans.append(dict(name=name,family='legacy',group=group,config=old[name]))
    assert len(ans)==len({c['name'] for c in ans})==25
    return ans


def fit(cfg,x,v,q,seed,loaded,trace=False):
    family=cfg['family']
    if family=='local':return local.fit(cfg['config'],x,v,q,seed,trace=trace)
    if family=='probe':return probe.fit(cfg['config'],x,v,q,seed,trace=trace)
    if family=='prior':return legacy.prior_fit(cfg['features'],x,v,q)
    assert family=='legacy';return legacy.fit(cfg['config'],x,v,q,seed,loaded)


def frozen_inputs(root):
    """Metadata used only OUTSIDE timed fit, for identity checks and observed inputs."""
    import json
    directories=[root/'results/probe_continuation_credit/development',root/'results/certificate_activity_attribution/development',root/'results/matched_budget_confirmation/conditioned_confirmation']
    ans={}
    for directory in directories:
        for row in json.loads((directory/'rows.json').read_text()):ans.setdefault((row['seed'],row['method']),(directory,row))
    return ans


def check_frozen(root,seed,cfg,a,m,index):
    from run_multiplier_fixed_point_screen import sha
    item=index.get((seed,cfg['name']))
    if item is None:return 0
    directory,row=item;assert sha(directory/row['file'])==row['sha256'];count=0
    with np.load(directory/row['file']) as z:
        for key in ['prediction','point_prediction','points','allocation','selected_b','best_bank','initial_b','initial_h','initial_u','atomic_trial_b']:
            if key in a and key in z:assert a[key].tobytes()==z[key].tobytes(),(seed,cfg['name'],key);count+=1
    if 'positive_modes' in m:assert m['positive_modes']==row['metadata']['positive_modes']
    return count
