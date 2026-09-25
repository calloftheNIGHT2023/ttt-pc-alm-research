"""358 real complete fits; immutable prediction seal precedes scoring."""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import cross_region_online_suite_v1 as suite
from posterior_confirmation_pipeline import discovery_box
from run_online_credit_fresh_v2 import signature,validate
from run_support_language_online_v1 import load,same


def verify(root,out,p):
    rows=suite.read(out/'rows.json');index={(r['seed'],r['method']):r for r in rows}
    assert len(rows)==len(index)==len(p['seeds'])*len(p['methods'])
    for r in rows:
        assert suite.sha(root/r['file'])==r['sha256'] and suite.sha(root/r['metadata_file'])==r['metadata_sha256']
        a=load(root/r['file']);validate(a);m=suite.read(root/r['metadata_file'])['metadata']
        assert not m['query_targets_accessed'] and m['execution_failed']==r['execution_failed']
        ref=p['observed_inputs'][str(r['seed'])];assert suite.sha(root/ref['file'])==ref['sha256']
        same(a,load(root/ref['file']),['x_observed','v_observed','q_observed'])
        if r['origin']=='new_358':assert r['seconds']==m['charged_complete_seconds']>0
        else:assert r['origin']=='frozen_pre358' and r['seconds'] is None
    for seed in p['seeds']:
        rr=sorted([r for r in rows if r['seed']==seed and r['origin']=='new_358'],key=lambda r:r['order'])
        expected=[p['configs'][int(j)]['name'] for j in np.random.default_rng(np.random.SeedSequence([358929,seed])).permutation(14)]
        assert [r['method'] for r in rr]==expected and suite.read(out/str(seed)/'commit.json')['rows']==rr
    return dict(predictors=len(rows),failures=sum(r['execution_failed'] for r in rows))


def run(root,out,stage):
    begin=time.perf_counter();hashes=suite.gate(root);configs=suite.catalogue(root)
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    pre=root/suite.BASE/'preflight_predictions_v1'
    if stage=='development':suite.complete(pre);assert suite.read(pre/'protocol.json')['source_sha256']==hashes
    seeds=suite.SEEDS[:1] if stage=='preflight' else suite.SEEDS
    inputs,input_manifest,_=suite.budget.observed(root,seeds)
    parent=root/'results/unvisited_online/development_predictions_v1';suite.complete(parent)
    old=[r for r in suite.read(parent/'rows.json') if r['seed'] in seeds];oldindex={(r['seed'],r['method']):r for r in old}
    env=suite.budget.resources.old.environment_snapshot(root);assert not env['other_research_or_git_pack_processes'],env
    p=dict(stage=stage,source_sha256=hashes,seeds=seeds,configs=configs,primary=suite.PRIMARY,
           methods=suite.read(parent/'protocol.json')['methods']+[c['name'] for c in configs],observed_inputs=input_manifest,
           parent_prediction_summary_sha256=suite.sha(parent/'summary.json'),query_targets_accessed=False,
           posterior_reference_accessed=False,new_blind_tasks=False,environment=env)
    suite.save(out/'protocol.json',p);ph=suite.sha(out/'protocol.json')
    rows=[dict(r,origin='frozen_pre358',historical_origin=r['origin'],historical_seconds=r['seconds'],seconds=None,order=None) for r in old]
    counts=dict(new_calls=0,invariant_arrays=0,native_replay_arrays=0,preflight_replay_arrays=0,
                trajectory_states=0,support_particles=0,component_pool_replays=0,component_screen_replays=0)
    with discovery_box(.12):
        for seed in seeds:
            d=out/str(seed);d.mkdir();records=[];x,v,q=inputs[seed];observed=signature(x,v,q)
            env=suite.budget.resources.old.environment_snapshot(root);assert not env['other_research_or_git_pack_processes'],env
            component=root/'results/cross_region_credit/development_v1'/str(seed)
            with np.load(component/'pool.npz',allow_pickle=False) as z:expected_pool=z['regions']
            for j in np.random.default_rng(np.random.SeedSequence([358929,seed])).permutation(14):
                cfg=configs[int(j)];name=cfg['name'];gc.collect();a,m,seconds=suite.invoke(cfg,x,v,q,seed)
                assert signature(x,v,q)==observed
                a.update(x_observed=x.copy(),v_observed=v.copy(),q_observed=q.copy());validate(a)
                oldr=oldindex[seed,cfg['reference_name']];oa=load(root/oldr['file']);om=suite.read(root/oldr['metadata_file'])['metadata']
                if not m['execution_failed']:
                    assert m['visited_modes']==om['visited_modes'] and m['original_positive_modes']==om['original_positive_modes']
                    assert m['trajectory_state_sha256']==om['trajectory_state_sha256'];counts['trajectory_states']+=len(m['trajectory_state_sha256'])
                    if 'channel' in cfg:
                        counts['invariant_arrays']+=same(a,oa,['best_bank','selected_b','point_prediction','trigger_b','trigger_h','trigger_u','trigger_credit','trigger_location'])
                        assert m['selected_state']==om['selected_state'] and set(m['original_positive_modes'])<=set(m['positive_modes'])
                        assert a['search_regions'].tobytes()==expected_pool.tobytes();counts['component_pool_replays']+=1
                        if cfg['settings'].get('screen',True):
                            label=cfg['channel']+('_reuse' if cfg['settings']['propagate'] else '_independent')
                            if cfg['settings'].get('steps',128)==256:label='zero_independent256'
                            with np.load(component/(label+'.npz'),allow_pickle=False) as z:
                                assert a['search_first_step'].tobytes()==z['first_step'].tobytes()
                                assert a['search_proof_credit'].tobytes()==z['proof_credit'].tobytes()
                            counts['component_screen_replays']+=1
                        ids=np.flatnonzero(a['search_first_step']==0);cap=cfg['settings'].get('geometry_budget',8)
                        if cap is not None:ids=ids[:cap]
                        assert [pp['mode'] for pp in m['proposal']['proposals']]==[a['search_regions'][i].tobytes().hex() for i in ids]
                    else:counts['native_replay_arrays']+=same(a,oa,list(a))
                    if m['positive_modes']==om['positive_modes']:same(a,oa,['points','allocation','prediction'])
                    if m['positive_modes']:
                        yy=np.broadcast_to(x,(len(a['points']),4))
                        for l in range(4):yy=np.maximum(0.,1.-abs(2*(yy+a['points'][:,l,None])-1.))
                        assert float(np.max(abs(yy-v)))<=.001+1e-7;counts['support_particles']+=len(a['points'])
                if stage=='development' and seed==seeds[0]:counts['preflight_replay_arrays']+=same(a,load(pre/str(seed)/(name+'.npz')),list(a))
                ap=d/(name+'.npz');mp=d/(name+'.json')
                with ap.open('xb') as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
                suite.save(mp,dict(seed=seed,method=name,order=len(records),protocol_sha256=ph,metadata=m))
                records.append(dict(seed=seed,method=name,origin='new_358',order=len(records),file=str(ap.relative_to(root)),sha256=suite.sha(ap),
                                    metadata_file=str(mp.relative_to(root)),metadata_sha256=suite.sha(mp),seconds=seconds,execution_failed=m['execution_failed']))
                counts['new_calls']+=1
            suite.save(d/'commit.json',dict(seed=seed,protocol_sha256=ph,rows=records,query_targets_accessed=False));rows.extend(records)
            print(dict(stage=stage,tasks=seeds.index(seed)+1,total=len(seeds),new_calls=counts['new_calls'],seconds=time.perf_counter()-begin),flush=True)
    suite.save(out/'rows.json',rows);checks=verify(root,out,p);assert suite.gate(root)==hashes
    suite.save(out/'before_query_manifest.json',dict(protocol_sha256=ph,rows_sha256=suite.sha(out/'rows.json'),source_sha256=hashes,counts=counts,checks=checks,query_targets_accessed=False))
    result=dict(passed=True,tasks=len(seeds),counts=counts,checks=checks,seconds=time.perf_counter()-begin,
                query_targets_accessed=False,core_research_goal_complete=False,
                outputs_sha256={n:suite.sha(out/n) for n in ['protocol.json','rows.json','before_query_manifest.json']})
    suite.save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','development'],required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];out=root/suite.BASE/(args.stage+'_predictions_v1');out.mkdir(parents=True,exist_ok=False)
    try:run(root,out,args.stage)
    except Exception:suite.save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
