"""Twenty-four contemporaneous controls, complete own-state online streams."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import online_primal_upper_h2 as candidate

previous=candidate.previous
memory=candidate.memory


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def trial(*args):
    saved=previous.recovery
    try:
        previous.recovery=SimpleNamespace(fit=candidate.fit)
        return previous.trial(*args)
    finally:previous.recovery=saved


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/bounded_error_primal_gate/component';audit=json.loads((parent/'integer_audit.json').read_text());assert audit['passed']
    failed=root/'results/online_primal_gate/development/protocol.json'
    hashes=dict(json.loads(failed.read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    assert sha(Path(__file__).with_name('audit_primal_gate_integer.py'))==audit['source_sha256']
    for name in ['audit_primal_gate_integer.py','online_primal_upper_h2.py',Path(__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    old=root/'results/recovered_online_comparison/development';oldp=json.loads((old/'protocol.json').read_text())
    p=dict(source_sha256=hashes,parent_protocol_sha256=sha(parent/'protocol.json'),integer_audit_sha256=sha(parent/'integer_audit.json'),
        failed_attempt_protocol_sha256=sha(failed),reference_protocol_sha256=sha(old/'protocol.json'),configs=candidate.configs(),seeds=oldp['seeds'],repetitions=2,
        stages=oldp['stages'],posterior_samples=2048,rng_seed=oldp['rng_seed'],order_seed=483331,query_points=257,
        primary=candidate.PRIMARY,primary_comparator=previous.PRIMARY,net_comparator='alm_c5',
        recovery_feature_budgets=list(candidate.recovery.FEATURE_BUDGETS),verification=candidate.verify(),
        scope='All original 18 methods plus six common gates; reused development tasks; no confirmation or official TTT',
        failure_policy='Preserve failed attempts and partial states; no successful-only overall MSE')
    out=root/'results/online_primal_gate/development_v2';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    refs={(r['seed'],r['method'],r['repetition'],r['n']):r for r in json.loads((old/'rows.json').read_text())}
    rows=[];episodes=[];failures=[];q=np.linspace(0,1,p['query_points']);order=np.random.default_rng(p['order_seed'])
    jobs=[(c,rep) for c in p['configs'] for rep in range(p['repetitions'])]
    checks=dict(all_states_bytewise_against_203=0,unchanged_first_stage_details=0,unchanged_proof_records=0)
    for seed in p['seeds']:
        xx,vv=previous.olddriver.observations(seed);seedrows=[]
        for i in order.permutation(len(jobs)):
            cfg,rep=jobs[i];rr,artifacts,failure=trial(xx,vv,q,cfg,seed,rep,p,out)
            if failure:failures.append(failure)
            for row,arrays in zip(rr,artifacts):
                ref=refs[seed,cfg.get('ungated_comparator',cfg['name']),rep,row['n']]
                assert sha(old/ref['state_file'])==ref['state_sha256']
                with np.load(old/ref['state_file']) as z:assert all(arrays[k].tobytes()==z[k].tobytes() for k in arrays),(seed,cfg['name'],row['n'])
                checks['all_states_bytewise_against_203']+=1
                if 'detail_file' in row:
                    assert sha(old/ref['detail_file'])==ref['detail_sha256']
                    detail=json.loads((out/row['detail_file']).read_text());baseline=json.loads((old/ref['detail_file']).read_text())
                    for key in ['credit_proofs','credit_bank','positive_mode_keys','discovery_bank_sha256','completion_proposals','geometry_calls']:
                        assert detail.get(key)==baseline.get(key),(cfg['name'],key)
                    for key in ['input_pattern_hash','screened_pattern_hash']:
                        assert [c.get(key) for c in detail.get('screen_calls',[])]==[c.get(key) for c in baseline.get('screen_calls',[])]
                    checks['unchanged_first_stage_details']+=1;checks['unchanged_proof_records']+=len(detail.get('credit_proofs',[]))
            ep=dict(seed=seed,method=cfg['name'],repetition=rep,complete=failure is None,stages=len(rr),
                failed_n=None if failure is None else failure['failed_n'],
                charged_seconds=sum(r['total_seconds'] for r in rr)+(0 if failure is None else failure['failed_seconds']),
                recovery_stages=sum(bool(r['recovery'] and r['recovery']['triggered']) for r in rr))
            episodes.append(ep);seedrows.extend(rr)
            print(json.dumps(dict(completed=len(episodes),total=len(jobs)*len(p['seeds']),seed=seed,method=cfg['name'],rep=rep,
                complete=ep['complete'],recoveries=ep['recovery_stages'],seconds=ep['charged_seconds'])),flush=True)
        truth=memory.base.forward(q,np.random.default_rng(seed).uniform(-.12,.12,4))
        for row in seedrows:
            with np.load(out/row['state_file']) as z:row['query_mse']=float(np.trapezoid((z['prediction']-truth)**2,x=q))
        rows.extend(seedrows)
        for name,value in [('rows.json',rows),('episodes.json',episodes),('failures.json',failures)]:
            (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    result=dict(execution_complete=True,episodes=len(episodes),stages=len(rows),failures=len(failures),
        recovery_events=sum(e['recovery_stages'] for e in episodes),checks=checks,sources=len(hashes))
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
