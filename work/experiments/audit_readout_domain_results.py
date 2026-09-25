"""Post-run integrity and mechanism accounting; no fitting or selection."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
runs=['common_cell_readout/development','common_cell_readout/discovery_frontier',
      'bounded_activity/development','guarded_activity/development']
integrity=[]
for run in runs:
    path=root/'results'/run;protocol=json.loads((path/'protocol.json').read_text());rows=json.loads((path/'episodes.json').read_text())
    match=all(hashlib.sha256((root/'work/experiments'/name).read_bytes()).hexdigest()==digest for name,digest in protocol['source_sha256'].items())
    assert match,run
    integrity.append(dict(run=run,rows=len(rows),sources_match=match))
bounded=json.loads((root/'results/bounded_activity/development/episodes.json').read_text())
guarded=json.loads((root/'results/guarded_activity/development/episodes.json').read_text())
domains=[]
for name in dict.fromkeys(r['method'] for r in bounded):
    group=[r for r in bounded if r['method']==name and 'unreachable_activity_fraction' in r['fit_meta']]
    if not group:continue
    total=sum(r['n_context']*r['fit_meta']['sweeps'] for r in group)
    domains.append(dict(method=name,weighted_unreachable_fraction=sum(r['fit_meta']['unreachable_activity_fraction']*r['n_context']*r['fit_meta']['sweeps'] for r in group)/total,
                        maximum_domain_violation=max(r['fit_meta']['maximum_activity_domain_violation'] for r in group)))
guards=[]
for name in dict.fromkeys(r['method'] for r in guarded):
    group=[r['fit_meta'] for r in guarded if r['method']==name and 'block_samples' in r['fit_meta']]
    if not group:continue
    total=sum(r['block_samples'] for r in group)
    guards.append(dict(method=name,blocks=total,clipped_worse_fraction=sum(r['clipped_worse_than_old'] for r in group)/total,
                       retained_old_fraction=sum(r['retained_old'] for r in group)/total,
                       maximum_conditional_energy_increase=max(r['maximum_conditional_energy_increase'] for r in group)))
mapping={'clipped32':'clipped_orthogonal32','clipped128':'clipped_orthogonal128','orthogonal128':'orthogonal128',
         'orthogonal240':'orthogonal240','adam240':'adam240_r16','lbfgs64':'lbfgs64'}
old={(r['method'],r['seed'],r['n_context']):r for r in bounded};gaps=[]
for row in guarded:
    if row['method'] in mapping:
        prior=old[mapping[row['method']],row['seed'],row['n_context']]
        gaps.append(abs(row['query_mse']-prior['query_mse']))
assert max(gaps)==0
result=dict(integrity=integrity,total_stage_rows=sum(r['rows'] for r in integrity),domains=domains,guards=guards,
            repeated_baseline_stage_pairs=len(gaps),maximum_repeated_query_mse_gap=max(gaps),
            scope='all reused development streams; no new independent confirmation')
out=root/'results/guarded_activity/mechanism_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
