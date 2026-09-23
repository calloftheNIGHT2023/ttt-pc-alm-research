"""287 prediction runner: durable call journal, immutable commits, tested resume.

No query teacher, evaluator, quality stopping, or scientific change from v1.
The fault hooks are old-context infrastructure tests, forbidden for confirmation.
"""
import argparse,json,os,re,shutil,time
from pathlib import Path
import numpy as np
import psutil
import torch
import probe_credit_confirmation_suite as suite
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha,dump


def exclusive_json(path,value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())


def partial_accounting(out,taskroot,current,protocol_hash):
    """Snapshot preserved attempts; in-flight charges remain explicitly unknown."""
    records=[]
    for directory in sorted(taskroot.glob('attempt_*')):
        if directory==current:continue
        assert directory.is_dir() and directory.resolve().is_relative_to(out.resolve())
        starts={p.stem.removesuffix('.start'):json.loads(p.read_text()) for p in directory.glob('*.start.json')}
        calls={p.stem.removesuffix('.call'):json.loads(p.read_text()) for p in directory.glob('*.call.json')}
        assert set(calls)<=set(starts)
        for name,call in calls.items():
            assert call['method']==name and call['protocol_sha256']==protocol_hash
            assert starts[name]['seed']==call['seed'] and starts[name]['execution_order']==call['execution_order']
            assert call['metadata']['charged_complete_seconds']>0
        records.append(dict(directory=str(directory.relative_to(out)),completed_calls=len(calls),
            known_completed_call_seconds=sum(c['metadata']['charged_complete_seconds'] for c in calls.values()),
            unresolved_started_calls=sorted(set(starts)-set(calls)),
            files={str(p.relative_to(out)):sha(p) for p in sorted(directory.iterdir()) if p.is_file()}))
    return dict(attempts=records,completed_calls=sum(r['completed_calls'] for r in records),
        known_completed_call_seconds=sum(r['known_completed_call_seconds'] for r in records),
        unresolved_started_calls=sum(len(r['unresolved_started_calls']) for r in records),
        unknown_charge_policy='Started calls lacking a durable completion record have unknown cost, never zero; attempt I/O and process overhead excluded from these algorithm subtotals')


def verify_commit(out,seed,names,protocol_hash):
    path=out/'tasks'/str(seed)/'commit.json';c=json.loads(path.read_text(encoding='utf-8'))
    assert c['seed']==seed and c['protocol_sha256']==protocol_hash and c['methods']==names and c['query_targets_accessed'] is False
    for name,h in c['files'].items():
        target=(out/name).resolve();assert target.is_relative_to(out.resolve()) and sha(target)==h,(seed,name)
    assert len(c['files'])==3*len(names)+1
    rows=json.loads((out/c['rows_file']).read_text());assert len(rows)==len(names) and {r['method'] for r in rows}==set(names)
    expected=[names[int(i)] for i in np.random.default_rng(np.random.SeedSequence([287929,seed])).permutation(len(names))]
    assert [r['method'] for r in rows]==expected
    expected_files={c['rows_file']}
    for order,r in enumerate(rows):
        assert r['seed']==seed and r['execution_order']==order and c['files'][r['file']]==r['sha256']
        call=json.loads((out/r['call_record']).read_text());start=json.loads((out/r['start_record']).read_text())
        assert call['seed']==start['seed']==seed and call['method']==start['method']==r['method']
        assert call['execution_order']==start['execution_order']==order and call['protocol_sha256']==start['protocol_sha256']==protocol_hash
        assert call['metadata']==r['metadata'] and r['metadata']['charged_complete_seconds']>0
        expected_files.update([r['file'],r['call_record'],r['start_record']])
    assert expected_files==set(c['files'])
    assert c['failures']==sum(r['metadata']['execution_failed'] for r in rows)
    assert c['charged_seconds']==sum(r['metadata']['charged_complete_seconds'] for r in rows)
    partial=partial_accounting(out,path.parent,out/c['attempt_directory'],protocol_hash);assert partial==c['interrupted_attempt_costs']
    return dict(seed=seed,file=str(path.relative_to(out)),sha256=sha(path),predictors=len(names),failures=c['failures'],charged_seconds=c['charged_seconds'],
        frozen_gold_arrays=c['frozen_gold_arrays'],interrupted_attempts=len(partial['attempts']),extra_completed_calls=partial['completed_calls'],
        extra_known_completed_call_seconds=partial['known_completed_call_seconds'],extra_unresolved_started_calls=partial['unresolved_started_calls'])


def acquire_lock(out):
    path=out/'RUNNING.lock';process=psutil.Process();identity=dict(pid=process.pid,create_time=process.create_time())
    if path.exists():
        old=json.loads(path.read_text())
        try:alive=psutil.Process(old['pid']).create_time()==old['create_time']
        except psutil.NoSuchProcess:alive=False
        if alive:raise RuntimeError('Another live prediction process owns this stage; do not restart it')
        archived=out/f'stale_lock_{time.time_ns()}.json'
        assert path.resolve().parent==out.resolve() and archived.resolve().parent==out.resolve();path.rename(archived)
    exclusive_json(path,identity);return path,identity


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--stage',choices=['preflight','confirmation'],default='preflight')
    faults=ap.add_mutually_exclusive_group();faults.add_argument('--preflight-stop-after-calls',type=int);faults.add_argument('--preflight-stop-after-tasks',type=int)
    args=ap.parse_args();root=args.project.resolve();src=Path(__file__).parent
    assert args.stage=='preflight' or (args.preflight_stop_after_calls is None and args.preflight_stop_after_tasks is None),'No confirmation fault hooks'
    for value in [args.preflight_stop_after_calls,args.preflight_stop_after_tasks]:assert value is None or value>0
    base=root/'results/probe_credit_confirmation';out=base/('runner_preflight_v2' if args.stage=='preflight' else 'predictions');out.mkdir(parents=True,exist_ok=True)
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    gate=base/'head_audit';gg=json.loads((gate/'summary.json').read_text());assert gg['passed'] and gg['may_preflight_confirmation_runner']
    gp=json.loads((gate/'protocol.json').read_text());hashes=dict(gp['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    head=base/'head_preflight';assert sha(head/'summary.json')==gp['head_preflight_sha256'];hp=json.loads((head/'protocol.json').read_text())
    design=root/'outputs/ttt-pc-alm-research/287_probe_credit_confirmation_protocol.md';assert sha(design)==hp['design_sha256']
    plan=json.loads((base/'planning/summary.json').read_text());assert plan['passed'] and plan['planned_tasks']==8192
    cfgs=suite.catalogue(root);names=[c['name'] for c in cfgs];assert all(re.fullmatch('[A-Za-z0-9_]+',name) for name in names)
    start=time.perf_counter();loaded,manifest=suite.resources.legacy.oldfit.meta.load(root);loading=time.perf_counter()-start;assert manifest==hp['checkpoint_manifest']
    seeds=[5910000,5910063] if args.stage=='preflight' else list(range(5500000,5508192));index=None;preflight_hash=None
    if args.stage=='preflight':
        index=suite.resources.frozen_inputs(root)
        for family in ['probe_credit_budget','probe_credit_budget_sensitivity']:
            directory=root/'results'/family/'development'
            for row in json.loads((directory/'rows.json').read_text()):index[row['seed'],row['method']]=(directory,row)
        assert all((seed,name) in index for seed in seeds for name in names)
    else:
        pre=base/'runner_preflight_v2';ps=json.loads((pre/'summary.json').read_text())
        assert ps['passed'] and ps['checkpoint_replay_verified'] and ps['actual_resume_test_passed'] and ps['predictors']==54
        assert ps['extra_completed_calls']==2 and ps['extra_unresolved_started_calls']==0
        preflight_hash=sha(pre/'summary.json');pr=json.loads((pre/'protocol.json').read_text());assert pr['source_sha256']==hashes and pr['configs']==cfgs
        for n,h in ps['outputs_sha256'].items():assert sha(pre/n)==h,n
        for seed in pr['seeds']:verify_commit(pre,seed,names,sha(pre/'protocol.json'))
        independent=base/'runner_audit_v2';ig=json.loads((independent/'summary.json').read_text())
        assert ig['passed'] and ig['may_launch_confirmation'] and ig['runner_preflight_summary_sha256']==preflight_hash
    p=dict(stage=args.stage,source_sha256=hashes,head_audit_sha256=sha(gate/'summary.json'),runner_preflight_sha256=preflight_hash,
        planning_summary_sha256=sha(base/'planning/summary.json'),design_sha256=sha(design),configs=cfgs,primary=suite.PRIMARY,seeds=seeds,methods=names,checkpoint_manifest=manifest,
        observations_per_task=4,query_points=257,order_seed=287929,particles=2048,query_targets_accessed=False,new_contexts=args.stage=='confirmation',trace=False,
        threads=dict(blas=1,omp=1,torch=torch.get_num_threads()),checkpoint_rule='Immutable task commits; durable starts/completions; preserved interrupted attempts separately charged; no query-based resume',
        failure_policy='Original numerical guard, charged attempt and zero-bias fallback; no algorithm retry or task replacement',free_disk_floor_bytes=20*2**30,
        partial_cost_policy='Known completed calls separately summed; in-flight calls without completion record explicitly unknown',preflight_faults_forbidden_on_confirmation=True)
    lock,identity=acquire_lock(out)
    try:
        protocol=out/'protocol.json'
        if protocol.exists():assert json.loads(protocol.read_text())==p,'Protocol/source changed; cannot resume frozen run'
        else:exclusive_json(protocol,p)
        protocol_hash=sha(protocol);invocation_id=time.time_ns();invocation=out/f'invocation_{invocation_id}.json'
        exclusive_json(invocation,dict(pid=os.getpid(),stage=args.stage,model_loading_seconds=loading,started_unix=time.time(),protocol_sha256=protocol_hash,
            preflight_stop_after_calls=args.preflight_stop_after_calls,preflight_stop_after_tasks=args.preflight_stop_after_tasks))
        if (out/'summary.json').exists():
            existing=json.loads((out/'summary.json').read_text());assert existing['passed']
            for seed in seeds:verify_commit(out,seed,names,protocol_hash)
            print(json.dumps(dict(passed=True,already_complete=True,tasks=len(seeds))),flush=True);return
        begin=time.perf_counter();commits=[];new_tasks=0;resumed_tasks=0;new_calls=0
        def planned_stop(kind,seed):
            record=dict(planned_preflight_stop=True,kind=kind,seed=seed,new_calls=new_calls,new_tasks=new_tasks,resumed_tasks=resumed_tasks,
                query_targets_accessed=False,invocation_file=invocation.name,protocol_sha256=protocol_hash)
            exclusive_json(out/f'preflight_stop_{invocation_id}.json',record);print(json.dumps(record),flush=True)
        with discovery_box(.12):
            for seed in seeds:
                taskroot=out/'tasks'/str(seed)
                if (taskroot/'commit.json').exists():commits.append(verify_commit(out,seed,names,protocol_hash));resumed_tasks+=1;continue
                assert shutil.disk_usage(out).free>=p['free_disk_floor_bytes'],'Free disk below20GiB; committed tasks preserved'
                taskroot.mkdir(parents=True,exist_ok=True);attempt=taskroot/f'attempt_{time.time_ns()}';assert attempt.resolve().is_relative_to(out.resolve());attempt.mkdir()
                if args.stage=='preflight':
                    directory,old=index[seed,suite.PRIMARY];assert sha(directory/old['file'])==old['sha256']
                    with np.load(directory/old['file']) as z:x=z['x_observed'].copy();v=z['v_observed'].copy();q=z['q_observed'].copy()
                else:
                    xx,vv=observations(seed);x=xx[:4].copy();v=vv[:4].copy();q=np.linspace(0,1,257);del xx,vv
                assert not np.isin(q,x).any()
                taskrows=[];files={};taskbegin=time.perf_counter();task_gold=0
                for order_index in np.random.default_rng(np.random.SeedSequence([287929,seed])).permutation(len(cfgs)):
                    cfg=cfgs[int(order_index)];order=len(taskrows);repairs=[]
                    start_path=attempt/(cfg['name']+'.start.json')
                    exclusive_json(start_path,dict(seed=seed,method=cfg['name'],execution_order=order,started_unix=time.time(),protocol_sha256=protocol_hash))
                    start=time.perf_counter()
                    with conditioned.geometry_scope(repairs):a,m=suite.resources.legacy.guarded_fit(cfg,x,v,q,seed,loaded,runner=suite.fit)
                    elapsed=time.perf_counter()-start;m.update(charged_complete_seconds=elapsed,geometry_repair_log=repairs,geometry_repair_count=len(repairs),
                        method_kind='head' if cfg['family']=='prior' or cfg['group'] in ['additional_head','shallow'] else 'mode_pool')
                    if m['execution_failed']:m['positive_modes']=[];m['fallback']=True
                    call_path=attempt/(cfg['name']+'.call.json')
                    exclusive_json(call_path,dict(seed=seed,method=cfg['name'],execution_order=order,metadata=m,protocol_sha256=protocol_hash))
                    if args.stage=='preflight':assert not m['execution_failed'];task_gold+=suite.resources.check_frozen(root,seed,cfg,a,m,index)
                    path=attempt/(cfg['name']+'.npz')
                    with path.open('xb') as f:np.savez_compressed(f,x_observed=x,v_observed=v,q_observed=q,**a);f.flush();os.fsync(f.fileno())
                    for item in [start_path,call_path,path]:files[str(item.relative_to(out))]=sha(item)
                    relative=str(path.relative_to(out));taskrows.append(dict(seed=seed,method=cfg['name'],file=relative,sha256=files[relative],execution_order=order,
                        call_record=str(call_path.relative_to(out)),start_record=str(start_path.relative_to(out)),metadata=m));del a,m;new_calls+=1
                    if args.preflight_stop_after_calls==new_calls:planned_stop('partial_task',seed);return
                rows_path=attempt/'rows.json';exclusive_json(rows_path,taskrows);files[str(rows_path.relative_to(out))]=sha(rows_path)
                commit=dict(seed=seed,protocol_sha256=protocol_hash,methods=names,files=files,rows_file=str(rows_path.relative_to(out)),query_targets_accessed=False,
                    failures=sum(r['metadata']['execution_failed'] for r in taskrows),charged_seconds=sum(r['metadata']['charged_complete_seconds'] for r in taskrows),
                    task_wall_including_io_seconds=time.perf_counter()-taskbegin,frozen_gold_arrays=task_gold,attempt_directory=str(attempt.relative_to(out)),
                    interrupted_attempt_costs=partial_accounting(out,taskroot,attempt,protocol_hash))
                exclusive_json(taskroot/'commit.json',commit);commits.append(verify_commit(out,seed,names,protocol_hash));new_tasks+=1
                progress=dict(stage=args.stage,tasks_completed=len(commits),tasks_total=len(seeds),predictors=len(commits)*len(cfgs),numerical_failures=sum(c['failures'] for c in commits),
                    new_tasks_this_invocation=new_tasks,resumed_tasks=resumed_tasks,invocation_seconds=time.perf_counter()-begin,query_targets_accessed=False,
                    algorithm_seconds=sum(c['charged_seconds'] for c in commits),extra_completed_calls=sum(c['extra_completed_calls'] for c in commits),
                    extra_known_completed_call_seconds=sum(c['extra_known_completed_call_seconds'] for c in commits),extra_unresolved_started_calls=sum(c['extra_unresolved_started_calls'] for c in commits))
                dump(out/'progress.json',progress)
                if len(commits)%4==0 or args.stage=='preflight':print(json.dumps(progress),flush=True)
                if args.preflight_stop_after_tasks==new_tasks:planned_stop('committed_task',seed);return
                if len(commits)%256==0:
                    for n,h in hashes.items():assert sha(src/n)==h,n
        replay=[verify_commit(out,seed,names,protocol_hash) for seed in seeds];assert replay==commits
        for n,h in hashes.items():assert sha(src/n)==h,n
        extra_calls=sum(c['extra_completed_calls'] for c in commits);extra_unknown=sum(c['extra_unresolved_started_calls'] for c in commits)
        test_passed=False
        if args.stage=='preflight':
            stops=[json.loads(f.read_text()) for f in sorted(out.glob('preflight_stop_*.json'))]
            assert len(stops)==2 and [s['kind'] for s in stops]==['partial_task','committed_task']
            assert stops[0]['new_calls']==2 and stops[0]['new_tasks']==0 and stops[1]['new_calls']==27 and stops[1]['new_tasks']==1
            assert resumed_tasks==new_tasks==1 and extra_calls==2 and extra_unknown==0;test_passed=True
        before=dict(protocol_sha256=protocol_hash,task_commits=commits,query_targets_accessed=False,tasks=len(seeds),predictors=len(seeds)*len(cfgs),source_sha256=hashes)
        exclusive_json(out/'before_query_manifest.json',before)
        ans=dict(passed=True,stage=args.stage,tasks=len(seeds),predictors=len(seeds)*len(cfgs),numerical_failures=sum(c['failures'] for c in commits),checkpoint_replay_verified=True,
            actual_resume_test_passed=test_passed,preflight_frozen_gold_arrays=sum(c['frozen_gold_arrays'] for c in commits),resumed_tasks=resumed_tasks,new_tasks_this_invocation=new_tasks,
            invocation_seconds=time.perf_counter()-begin,query_targets_accessed=False,algorithm_seconds=sum(c['charged_seconds'] for c in commits),extra_completed_calls=extra_calls,
            extra_known_completed_call_seconds=sum(c['extra_known_completed_call_seconds'] for c in commits),extra_unresolved_started_calls=extra_unknown,
            outputs_sha256={n:sha(out/n) for n in ['protocol.json','before_query_manifest.json']},
            next='Independent runner audit then confirmation' if args.stage=='preflight' else 'All-predictor forward/support audit and fixed-first64 original trajectory audit before query evaluation')
        exclusive_json(out/'summary.json',ans);print(json.dumps(ans),flush=True)
    finally:
        if lock.exists() and json.loads(lock.read_text())==identity:lock.unlink()


if __name__=='__main__':main()
