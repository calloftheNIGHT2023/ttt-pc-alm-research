"""296: old2 compatibility plus old64 first-global functional gate, no targets."""
import argparse
from pathlib import Path
import time
import numpy as np
import counterfactual_credit_branching_v1 as old
import counterfactual_credit_branching_v2 as new
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from diagnose_counterfactual_conditional_risk_v1 import read,save,sha


def run(root,out):
    raw=root/'results/certificate_activity_attribution/development'
    before=read(raw/'before_evaluation_manifest.json')
    assert sha(raw/'rows.json')==before['rows_sha256']
    assert sha(raw/'protocol.json')==before['protocol_sha256']
    originals={r['seed']:r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    prop=root/'results/counterfactual_branch_proposals/development_v1'
    ps=read(prop/'summary.json')
    assert ps['passed'] and ps['tasks']==64
    for name,digest in ps['outputs_sha256'].items():
        assert sha(prop/name)==digest
    proposals={r['seed']:r for r in read(prop/'tasks.json')}
    shadow=root/'results/state_matched_dual_intervention/development_v2'
    ss=read(shadow/'summary.json')
    assert ss['passed']
    frozen=read(root/'results/round_287_audit_v6.json')
    for name,digest in frozen['source_sha256'].items():
        assert sha(root/'work/experiments'/name)==digest
    sources={name:sha(root/'work/experiments'/name) for name in
             ['counterfactual_credit_branching_v1.py','counterfactual_credit_branching_v2.py',
              'test_counterfactual_credit_branching_v3.py','diagnose_counterfactual_conditional_risk_v1.py']}
    save(out/'protocol.json',dict(seeds=list(range(5910000,5910064)),compatibility_seeds=[5910000,5910001],
        source_sha256=sources,parent_summary_sha256=sha(prop/'summary.json'),
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/296_first_novel_branch_gate_protocol.md'),
        rule='first_global_forward_mode',query_targets_accessed=False,quality_evaluated=False,
        resources_matched=False,frozen_dependency_seal_sha256=sha(root/'results/round_287_audit_v6.json')))
    cfg=next(dict(c) for c in new.original.CONFIGS if c['name']=='credit_control_probe33')
    rows,compatibility=[],[]
    begin=time.perf_counter()
    for seed in range(5910000,5910064):
        path=raw/originals[seed]['file']
        assert sha(path)==originals[seed]['sha256']==proposals[seed]['original_sha256']
        with np.load(path,allow_pickle=False) as z:
            x,v=z['x_observed'],z['v_observed']
            stored={name:z[name] for name in z.files if name.startswith(('prefix_','anchor_'))}
            identity={name:z[name] for name in ['selected_b','best_bank','point_prediction']}
        q=np.linspace(0,1,257)
        with discovery_box(.12):
            if seed in [5910000,5910001]:
                with conditioned.geometry_scope([]):
                    base,_=new.original.fit(cfg,x,v,q,seed,trace=True)
                for rule in ['none','changed_forward_mode']:
                    with conditioned.geometry_scope([]):
                        expected,old_meta=old.fit(x,v,q,seed,rule=rule,trace=True)
                    with conditioned.geometry_scope([]):
                        actual,new_meta=new.fit(x,v,q,seed,rule=rule,trace=True)
                    assert set(expected)==set(actual)
                    for name in expected:
                        assert expected[name].tobytes()==actual[name].tobytes(),(seed,rule,name)
                    if rule=='none':
                        for name in base:
                            assert base[name].tobytes()==actual[name].tobytes(),(seed,rule,name)
                    assert old_meta['positive_modes']==new_meta['positive_modes']
                    compatibility.append(dict(seed=seed,rule=rule,arrays=len(expected),passed=True))
            repairs=[]
            with conditioned.geometry_scope(repairs):
                started=time.perf_counter()
                candidate,meta=new.fit(x,v,q,seed,trace=True)
                seconds=time.perf_counter()-started
        for name,value in {**stored,**identity}.items():
            assert candidate[name].tobytes()==value.tobytes(),(seed,name)
        expected=proposals[seed]['rules']['first_global_forward_mode']
        locations=candidate['counterfactual_locations'].tolist()
        assert locations==expected['locations']
        assert meta['counterfactual_state_steps']==expected['proposals']==len(locations)
        spath=shadow/f'{seed}.npz'
        assert sha(spath)==ss['outputs_sha256'][f'{seed}.npz']
        with np.load(spath,allow_pickle=False) as z:
            bank=np.array([z[('prefix' if phase==0 else 'anchor')+'_shadow_b'][step-1,origin]
                           for phase,step,origin in locations]).reshape(-1,4)
        assert candidate['counterfactual_b'].tobytes()==bank.tobytes(),seed
        assert set(meta['positive_modes'])==set(originals[seed]['metadata']['positive_modes'])|set(expected['new_positive_modes'])
        assert set(originals[seed]['metadata']['visited_modes'])<=set(meta['visited_modes'])
        file=f'{seed}.npz'
        with (out/file).open('xb') as stream:
            np.savez_compressed(stream,**candidate)
        rows.append(dict(seed=seed,file=file,sha256=sha(out/file),source_file_sha256=originals[seed]['sha256'],
            metadata=meta,preserved_arrays=len(stored)+len(identity),functional_only_seconds=seconds,
            geometry_repairs=repairs))
        if (seed-5910000+1)%8==0:
            print(dict(tasks=seed-5910000+1,passed=True,seconds=time.perf_counter()-begin),flush=True)
    save(out/'compatibility.json',compatibility)
    save(out/'rows.json',rows)
    for name,digest in sources.items():
        assert sha(root/'work/experiments'/name)==digest
    summary=dict(passed=True,tasks=64,compatibility_cases=len(compatibility),
        compatibility_arrays=sum(r['arrays'] for r in compatibility),
        preserved_arrays=sum(r['preserved_arrays'] for r in rows),
        extra_shadow_steps=sum(r['metadata']['counterfactual_state_steps'] for r in rows),
        mean_functional_only_seconds=float(np.mean([r['functional_only_seconds'] for r in rows])),
        query_targets_accessed=False,quality_evaluated=False,resources_matched=False,core_research_goal_complete=False,
        seconds=time.perf_counter()-begin,outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='outputs_sha256'},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    root=parser.parse_args().project.resolve()
    out=root/'results/counterfactual_credit_branching/first_global_support_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:
        run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True))
        raise


if __name__=='__main__':
    main()
