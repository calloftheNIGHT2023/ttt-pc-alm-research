"""Independent old-task census, gold identity and interruption accounting audit."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import probe_credit_confirmation_suite as suite
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';inp=base/'runner_preflight_v2';out=base/'runner_audit_v2';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'protocol.json').exists();s=json.loads((inp/'summary.json').read_text());p=json.loads((inp/'protocol.json').read_text())
    assert s['passed'] and s['actual_resume_test_passed'] and s['checkpoint_replay_verified'] and not s['query_targets_accessed']
    assert p['seeds']==[5910000,5910063] and not p['new_contexts'] and not (inp/'RUNNING.lock').exists()
    for n,h in s['outputs_sha256'].items():assert sha(inp/n)==h,n
    configs=suite.catalogue(root);assert configs==p['configs'];names=[c['name'] for c in configs]
    hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'protocol.json',dict(source_sha256=hashes,runner_preflight_summary_sha256=sha(inp/'summary.json'),query_targets_accessed=False,new_contexts_accessed=False))
    index=suite.resources.frozen_inputs(root)
    for family in ['probe_credit_budget','probe_credit_budget_sensitivity']:
        directory=root/'results'/family/'development'
        for row in json.loads((directory/'rows.json').read_text()):index[row['seed'],row['method']]=(directory,row)
    manifest=json.loads((inp/'before_query_manifest.json').read_text());assert manifest['tasks']==2 and manifest['predictors']==54
    counts=Counter();charged=0.;extra=0.;preserved={};configs={c['name']:c for c in configs};commit_hashes={}
    for seed in p['seeds']:
        cp=inp/'tasks'/str(seed)/'commit.json';c=json.loads(cp.read_text());assert c['seed']==seed and c['protocol_sha256']==sha(inp/'protocol.json')
        item=next(r for r in manifest['task_commits'] if r['seed']==seed);assert sha(cp)==item['sha256'];commit_hashes[str(cp.relative_to(inp))]=sha(cp)
        assert c['methods']==names and not c['query_targets_accessed']
        for rel,h in c['files'].items():assert (inp/rel).resolve().is_relative_to(inp.resolve()) and sha(inp/rel)==h;counts['committed_files']+=1
        rows=json.loads((inp/c['rows_file']).read_text());assert len(rows)==27
        order=[names[int(i)] for i in np.random.default_rng(np.random.SeedSequence([287929,seed])).permutation(27)]
        assert [r['method'] for r in rows]==order
        for i,r in enumerate(rows):
            assert r['seed']==seed and r['execution_order']==i and not r['metadata']['execution_failed']
            call=json.loads((inp/r['call_record']).read_text());start=json.loads((inp/r['start_record']).read_text())
            assert call['metadata']==r['metadata'] and call['seed']==start['seed']==seed and call['method']==start['method']==r['method']
            assert call['execution_order']==start['execution_order']==i and call['protocol_sha256']==start['protocol_sha256']==c['protocol_sha256']
            charged+=call['metadata']['charged_complete_seconds'];counts['durable_completed_calls']+=1
            assert sha(inp/r['file'])==r['sha256']
            with np.load(inp/r['file']) as z:
                a={k:z[k].copy() for k in z.files};assert np.array_equal(a['q_observed'],np.linspace(0,1,257))
            d,old=index[seed,r['method']]
            with np.load(d/old['file']) as z:
                for key in ['x_observed','v_observed','q_observed']:assert a[key].tobytes()==z[key].tobytes();counts['observed_input_arrays']+=1
            counts['frozen_gold_arrays']+=suite.resources.check_frozen(root,seed,configs[r['method']],a,r['metadata'],index);counts['predictors']+=1
        partial=c['interrupted_attempt_costs'];subtotal=0.;nc=0
        for attempt in partial['attempts']:
            directory=inp/attempt['directory'];assert directory!=inp/c['attempt_directory']
            actual={str(f.relative_to(inp)):sha(f) for f in directory.iterdir() if f.is_file()};assert actual==attempt['files'];preserved.update(actual)
            calls=sorted(directory.glob('*.call.json'));starts=sorted(directory.glob('*.start.json'));assert len(calls)==len(starts)==2
            assert len(list(directory.glob('*.npz')))==2 and not (directory/'rows.json').exists() and not attempt['unresolved_started_calls']
            sec=0.
            for f in calls:
                rec=json.loads(f.read_text());assert rec['protocol_sha256']==c['protocol_sha256'] and rec['seed']==seed
                assert rec['method']==order[rec['execution_order']];sec+=rec['metadata']['charged_complete_seconds']
                with np.load(directory/(rec['method']+'.npz')) as z:a={k:z[k].copy() for k in z.files}
                counts['partial_gold_arrays']+=suite.resources.check_frozen(root,seed,configs[rec['method']],a,rec['metadata'],index)
            assert sec==attempt['known_completed_call_seconds'];subtotal+=sec;nc+=len(calls);counts['interrupted_attempts']+=1
        assert subtotal==partial['known_completed_call_seconds'] and nc==partial['completed_calls'] and partial['unresolved_started_calls']==0
        extra+=subtotal;counts['extra_completed_calls']+=nc;counts['tasks']+=1
    invocations=[json.loads(f.read_text()) for f in sorted(inp.glob('invocation_*.json'))]
    assert len(invocations)==3 and [r['preflight_stop_after_calls'] for r in invocations]==[2,None,None]
    assert [r['preflight_stop_after_tasks'] for r in invocations]==[None,1,None]
    stops=[json.loads(f.read_text()) for f in sorted(inp.glob('preflight_stop_*.json'))]
    assert [(r['kind'],r['new_calls'],r['new_tasks']) for r in stops]==[('partial_task',2,0),('committed_task',27,1)]
    assert s['resumed_tasks']==s['new_tasks_this_invocation']==1
    assert counts['predictors']==54 and counts['committed_files']==164 and counts['interrupted_attempts']==1 and counts['extra_completed_calls']==2
    assert counts['frozen_gold_arrays']==s['preflight_frozen_gold_arrays']==348
    assert abs(charged-s['algorithm_seconds'])<1e-12 and extra==s['extra_known_completed_call_seconds']>0
    dump(out/'files.json',dict(commits=commit_hashes,preserved_partial_files=preserved))
    ans=dict(passed=True,counts=counts,may_launch_confirmation=True,runner_preflight_summary_sha256=sha(inp/'summary.json'),
        charged_successful_seconds=charged,separate_interrupted_completed_call_seconds=extra,unresolved_started_calls=0,
        phase_accesses_new_contexts_or_targets=False,outputs_sha256={n:sha(out/n) for n in ['protocol.json','files.json']},
        scope='Controlled process-stop and resume using old contexts; not proof against every OS/filesystem crash and not new scientific evidence')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
