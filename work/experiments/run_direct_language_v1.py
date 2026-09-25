"""365 immutable four-way actual-online prediction runner; no query answers."""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import frontier_online_suite_v1 as previous
import direct_language_inference_v1 as direct
from posterior_confirmation_pipeline import discovery_box
from run_online_credit_fresh_v2 import signature,validate
from run_support_language_online_v1 import load,same

BASE='results/direct_language';SEEDS=list(range(328000000,328000032))
NAMES=['direct_language_all','frontier_dual_frontier_g8','frontier_c20_all','frontier_native_adam240']
read,sha,save,complete=previous.read,previous.sha,previous.save,previous.complete


def gate(root):
    h=previous.gate(root)
    for p in ['outputs/ttt-pc-alm-research/365_direct_language_protocol_v1.md']+[
        'work/experiments/'+n for n in ['direct_language_inference_v1.py','run_direct_language_v1.py','evaluate_direct_language_v1.py']]:
        h[p]=sha(root/p)
    return h


def verify(root,out,p):
    rows=read(out/'rows.json');assert len(rows)==len(p['seeds'])*4
    for seed in p['seeds']:
        rr=[r for r in rows if r['seed']==seed]
        expected=[NAMES[int(i)] for i in np.random.default_rng(np.random.SeedSequence([365929,seed])).permutation(4)]
        assert [r['method'] for r in rr]==expected and read(out/str(seed)/'commit.json')['rows']==rr
        ref=p['observed_inputs'][str(seed)];assert sha(root/ref['file'])==ref['sha256']
        for r in rr:
            assert sha(root/r['file'])==r['sha256'] and sha(root/r['metadata_file'])==r['metadata_sha256']
            a=load(root/r['file']);validate(a);same(a,load(root/ref['file']),['x_observed','v_observed','q_observed'])
            m=read(root/r['metadata_file'])['metadata'];assert not m['execution_failed'] and not m['query_targets_accessed']
            assert r['seconds']==m['charged_complete_seconds']>0
    return dict(predictors=len(rows),failures=0)


def run(root,out,stage):
    begin=time.perf_counter();hashes=gate(root)
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    pre=root/BASE/'preflight_predictions_v1'
    if stage=='development':complete(pre);assert read(pre/'protocol.json')['source_sha256']==hashes
    seeds=SEEDS[:1] if stage=='preflight' else SEEDS
    inputs,manifest,_=previous.budget.observed(root,seeds)
    parent=root/previous.BASE/'development_predictions_v1';complete(parent)
    old={(r['seed'],r['method']):r for r in read(parent/'rows.json')}
    cfgs={c['name']:c for c in previous.catalogue(root)}
    env=previous.budget.resources.old.environment_snapshot(root);assert not env['other_research_or_git_pack_processes'],env
    p=dict(seeds=seeds,methods=NAMES,primary='frontier_dual_frontier_g8',source_sha256=hashes,
           observed_inputs=manifest,environment=env,query_targets_accessed=False,new_blind_tasks=False)
    save(out/'protocol.json',p);rows=[];counts=dict(calls=0,preflight_arrays=0,control_arrays=0)
    with discovery_box(.12):
        for seed in seeds:
            directory=out/str(seed);directory.mkdir();records=[];x,v,q=inputs[seed];before=signature(x,v,q)
            env=previous.budget.resources.old.environment_snapshot(root);assert not env['other_research_or_git_pack_processes'],env
            for j in np.random.default_rng(np.random.SeedSequence([365929,seed])).permutation(4):
                name=NAMES[int(j)];gc.collect();tick=time.perf_counter()
                if name=='direct_language_all':a,m=direct.fit(x,v,q,seed)
                else:a,m,_=previous.invoke(cfgs[name],x,v,q,seed)
                seconds=time.perf_counter()-tick;m['charged_complete_seconds']=seconds
                assert not m['execution_failed'] and signature(x,v,q)==before
                a.update(x_observed=x.copy(),v_observed=v.copy(),q_observed=q.copy());validate(a)
                if name!='direct_language_all':
                    r=old[seed,name];assert sha(root/r['file'])==r['sha256']
                    counts['control_arrays']+=same(a,load(root/r['file']),list(a))
                    assert m['positive_modes']==read(root/r['metadata_file'])['metadata']['positive_modes']
                if stage=='development' and seed==seeds[0]:counts['preflight_arrays']+=same(a,load(pre/str(seed)/(name+'.npz')),list(a))
                ap=directory/(name+'.npz');mp=directory/(name+'.json')
                with ap.open('xb') as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
                save(mp,dict(seed=seed,method=name,metadata=m))
                records.append(dict(seed=seed,method=name,order=len(records),file=str(ap.relative_to(root)),sha256=sha(ap),
                    metadata_file=str(mp.relative_to(root)),metadata_sha256=sha(mp),seconds=seconds,execution_failed=False))
                counts['calls']+=1
            save(directory/'commit.json',dict(rows=records,query_targets_accessed=False));rows.extend(records)
            print(dict(stage=stage,tasks=seeds.index(seed)+1,total=len(seeds),calls=counts['calls'],seconds=time.perf_counter()-begin),flush=True)
    save(out/'rows.json',rows);checks=verify(root,out,p);assert gate(root)==hashes
    save(out/'before_query_manifest.json',dict(rows_sha256=sha(out/'rows.json'),protocol_sha256=sha(out/'protocol.json'),query_targets_accessed=False))
    result=dict(passed=True,counts=counts,checks=checks,seconds=time.perf_counter()-begin,core_research_goal_complete=False,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','rows.json','before_query_manifest.json']})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['preflight','development'],required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];out=root/BASE/(args.stage+'_predictions_v1');out.mkdir(parents=True,exist_ok=False)
    try:run(root,out,args.stage)
    except Exception:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
