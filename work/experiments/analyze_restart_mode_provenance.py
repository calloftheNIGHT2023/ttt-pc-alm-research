"""Post271 support-only provenance: every D/A restart from complementary masks."""
import argparse,json
from pathlib import Path
import numpy as np
from audit_local_dual_jump_modes import modes
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/fixed_restart_credit';inp=base/'development';audit=base/'evaluation_audit';out=base/'restart_provenance';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    aa=json.loads((audit/'summary.json').read_text());assert aa['passed'];hashes=dict(json.loads((audit/'protocol.json').read_text())['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    p0=json.loads((inp/'protocol.json').read_text());rows0={(r['seed'],r['method']):r for r in json.loads((inp/'rows.json').read_text())}
    old=root/'results/matched_budget_confirmation/conditioned_confirmation';oldrows={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text())}
    reference=root/'results/confirmation_conditional_risk/reference';refs={r['seed']:r for r in json.loads((reference/'coverage.json').read_text())}
    dump(out/'protocol.json',dict(source_sha256=hashes,evaluation_audit_sha256=sha(audit/'summary.json'),phase_accesses_query_targets=False,
        scope='Post-result provenance on all64 tasks/all17 restart indices; no new algorithm, no labels, no risk-based routing; full reference only for diagnosis'))
    tasks=[];restart_rows=[];empty_cases=[]
    for seed in p0['seeds']:
        refrow=refs[seed];assert sha(reference/refrow['file'])==refrow['sha256'];ref=json.loads((reference/refrow['file']).read_text());vol={r['pattern']:r['volume'] for r in ref['reference']['positive_regions']};x=np.array(ref['x_observed'])
        ra=rows0[seed,'mixed_DA_even64'];rb=rows0[seed,'mixed_DA_odd64'];assert sha(inp/ra['file'])==ra['sha256'] and sha(inp/rb['file'])==rb['sha256']
        with np.load(inp/ra['file']) as a,np.load(inp/rb['file']) as b:
            pools={'D':[],'A':[]}
            for action in ['D','A']:
                for r in range(17):
                    z=a if a['assigned_actions'][r]==action else b;assert z['assigned_actions'][r]==action
                    visits=modes(x,z['history_b'][:,r]);positive=sorted(set(visits)&set(vol));pools[action].append(set(positive))
                    restart_rows.append(dict(seed=seed,action=action,restart=r,positive_modes=positive,first_positive_step=next((t for t in range(65) if next(iter(modes(x,z['history_b'][t,r:r+1]))) in vol),None)))
            full={action:set().union(*pp) for action,pp in pools.items()}
            for action,name in [('D','minimum_dual_alm64'),('A','minimum_activity_alm64')]:
                assert sorted(full[action])==oldrows[seed,name]['metadata']['positive_modes'],(seed,action)
            if not full['D']:
                empty_cases.append(dict(seed=seed,activity_positive_restart_indices=[r for r in range(17) if pools['A'][r]],activity_modes=len(full['A'])))
            switched=[]
            for r in range(17):
                new=set().union(*(pools['D'][i] if i!=r else pools['A'][i] for i in range(17)))
                switched.append(dict(restart=r,positive_modes=len(new),gain=sorted(new-full['D']),loss=sorted(full['D']-new),
                    mass_change=(sum(vol[k] for k in new)-sum(vol[k] for k in full['D']))/sum(vol.values())))
            tasks.append(dict(seed=seed,full_dual_modes=sorted(full['D']),full_activity_modes=sorted(full['A']),one_activity_replacement=switched))
    byrestart=[]
    for r in range(17):
        rr=[t['one_activity_replacement'][r] for t in tasks]
        byrestart.append(dict(restart=r,tasks_with_gained_modes=sum(bool(z['gain']) for z in rr),tasks_with_lost_modes=sum(bool(z['loss']) for z in rr),
            mean_mass_change=float(np.mean([z['mass_change'] for z in rr])),rescued_empty_tasks=[t['seed'] for t in tasks if not t['full_dual_modes'] and t['one_activity_replacement'][r]['positive_modes']]))
    for n,v in [('tasks.json',tasks),('restart_rows.json',restart_rows),('by_restart.json',byrestart),('empty_cases.json',empty_cases)]:dump(out/n,v)
    ans=dict(passed=True,tasks=64,per_restart_rows=len(restart_rows),single_replacement_pool_diagnostics=64*17,full_pool_identities=128,empty_cases=empty_cases,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','tasks.json','restart_rows.json','by_restart.json','empty_cases.json']},phase_accesses_query_targets=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
