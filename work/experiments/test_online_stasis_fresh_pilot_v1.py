"""307 setup: frozen sources, seed-registry scan, old-only arithmetic tests."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import numpy as np
import online_stasis_fresh_suite_v1 as suite
import online_stasis_fresh_pilot_v1 as pilot
from diagnose_gradient_flat_split_states_v1 import read,save,sha


def run(root,out):
    hashes=suite.gate(root);cfgs=suite.catalogue(root)
    assert len(cfgs)==46 and cfgs[:37]==suite.old.catalogue(root)
    assert cfgs[37]['name']==suite.PRIMARY
    assert len(suite.PILOT_SEEDS)==256 and suite.PILOT_SEEDS[0]==307000000 and suite.PILOT_SEEDS[-1]==307000255
    assert not set(suite.PILOT_SEEDS)&(set(suite.OLD_SEEDS)|set(range(5910000,5910064))|set(range(5500000,5508192))|set(range(294000000,294000128)))
    # Boundaries exclude decimal durations that contain the same digit sequence.
    pattern=r'(?<![0-9.])307000[0-9]{3}(?![0-9.])'
    globs=['*protocol*.json','*tasks*.json','*episodes*.json','*seeds*.json','*manifest*.json']
    select=[arg for g in globs for arg in ['-g',g]]+['-g','!results/online_stasis_fresh_pilot/**']
    command=['rg','--pcre2','--no-ignore','-n',*select,pattern,'results']
    probe=subprocess.run(command,cwd=root,capture_output=True,text=True,encoding='utf-8')
    assert probe.returncode==1 and not probe.stdout and not probe.stderr,(probe.returncode,probe.stdout,probe.stderr)
    inventory=subprocess.run(['rg','--files','--no-ignore','results',*select],cwd=root,capture_output=True,text=True,encoding='utf-8')
    assert inventory.returncode==0 and not inventory.stderr
    names=sorted(inventory.stdout.splitlines())
    allnames=subprocess.run(['rg','--files','--no-ignore','results','-g','!results/online_stasis_fresh_pilot/**'],cwd=root,capture_output=True,text=True,encoding='utf-8')
    assert allnames.returncode==0 and not allnames.stderr
    assert not any(re.search(r'307000[0-9]{3}',n) for n in allnames.stdout.splitlines())
    save(out/'seed_registry_scan.json',dict(passed=True,command=command,returncode=probe.returncode,
        checked_manifest_paths=names,manifest_path_count=len(names),all_result_paths=len(allnames.stdout.splitlines()),
        path_inventory_sha256=hashlib.sha256(inventory.stdout.encode()).hexdigest(),
        scope='Existing result paths and JSON protocol/task/episode/seed/manifest inventories, excluding this new experiment; not a claim about inaccessible external archives',
        query_targets_accessed=False))
    # Only previously used task labels test the evaluator implementation.
    truth_gap=0.;cases=0
    for seed in suite.OLD_SEEDS:
        q=np.linspace(0,1,257);b=np.random.default_rng(seed).uniform(-.12,.12,4)
        truth_gap=max(truth_gap,float(np.max(abs(pilot.scalar_truth(seed,q)-pilot.forward(q,b[None])[0]))));cases+=1
    assert truth_gap<2e-12
    names=[c['name'] for c in cfgs];fake=[]
    for name in names:
        seconds=1. if name in [suite.PRIMARY,names[0]] else 1.1 if name==names[1] else 2.
        fake.append(dict(method=name,seconds=seconds,metadata=dict(execution_failed=False,nested=dict(state_bytes=100))))
    cost=pilot.cost_table(fake,names)
    assert set(cost['within_budget'])=={suite.PRIMARY,names[0]}
    assert set(cost['sensitivity_110_percent'])=={suite.PRIMARY,names[0],names[1]}
    assert all(r['maximum_reported_byte_fields']=={'nested.state_bytes':100} for r in cost['methods'])
    reference=suite.references(root);assert all((s,c['name']) in reference for s in suite.OLD_SEEDS for c in cfgs)
    bootstrap=pilot.stats.selftest()
    save(out/'protocol.json',dict(source_sha256=hashes,design_sha256=sha(root/suite.DESIGN),
        old_query_seeds=suite.OLD_SEEDS,fresh_query_targets_accessed=False,primary_index=37,methods=46,main_comparisons=45))
    result=dict(passed=True,methods=46,main_comparisons=45,fresh_tasks=256,old_scalar_cases=cases,maximum_old_truth_gap=truth_gap,
        resource_boundary_selftest=True,bootstrap_selftest=bootstrap,seed_registry_passed=True,
        fresh_query_targets_accessed=False,core_research_goal_complete=False,
        outputs_sha256={name:sha(out/name) for name in ['seed_registry_scan.json','protocol.json']})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();out=root/suite.BASE/'tests_v1';out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise
