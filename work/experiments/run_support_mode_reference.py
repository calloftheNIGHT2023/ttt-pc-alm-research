"""Bounded support-only reference for first-write mode coverage."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import enumerate_support_modes as reference
model=reference.memory;base=reference.base


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    previous=json.loads(Path('results/posterior_state_reuse/online_development/protocol.json').read_text())
    for name,h in previous['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    configs=[c for c in previous['configs'] if c['name'] in ['alm20_prior256','adam60_prior256','adam240_prior256','direct128_prior256','nodual20_prior768','pc80_prior768']]
    sources=[Path(__file__),Path(reference.__file__)]
    protocol=dict(seeds=list(range(5900000,5900016)),n_context=4,depth=4,configs=configs,
        max_lp_per_task=50000,max_search_seconds_per_task=180,
        source_sha256={**previous['source_sha256'],**{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}},
        audit=reference.verify(),scope='old development support only; no query inputs or targets; exhaustive reference is not an online candidate or ALM-specific mechanism')
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[]
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24)
        v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24);x=x[:4];v=v[:4]
        ref=reference.Reference(x,v,max_lp=protocol['max_lp_per_task'],max_seconds=protocol['max_search_seconds_per_task']).enumerate()
        filename=f'reference_{seed}.json';(a.out/filename).write_text(json.dumps(dict(seed=seed,x=x.tolist(),v=v.tolist(),**ref),indent=2),encoding='utf-8')
        item=dict(seed=seed,reference_file=filename,reference_sha256=hashlib.sha256((a.out/filename).read_bytes()).hexdigest(),
            enumeration_completed=ref['enumeration_completed'],numerical_volume_reference_complete=ref['numerical_volume_reference_complete'],
            positive_regions=len(ref['positive_regions']),unresolved_final_regions=len(ref['final_geometry_unresolved']),
            lp_calls=ref['lp_calls'],certified_rejections=ref['certified_lp_rejections'],elapsed_seconds=ref['elapsed_seconds'],coverage=[])
        if ref['numerical_volume_reference_complete']:
            polys={p['pattern']:p for p in ref['positive_regions']};total=sum(p['volume'] for p in polys.values())
            for c in configs:
                bank,_=model.discover(x,v,None,c);keys={r.tobytes().hex() for r in model.interface.archived.signatures(x,bank)}
                found=sorted(keys&polys.keys());_,_,meta=model.materialize(x,v,np.zeros(4),bank)
                volume=sum(polys[k]['volume'] for k in found)
                assert len(found)==meta['positive_volume_regions'],(seed,c['name'],len(found),meta['positive_volume_regions'])
                assert np.isclose(volume/.24**4,meta['discovered_prior_mass'],rtol=1e-8,atol=1e-30)
                item['coverage'].append(dict(method=c['name'],positive_regions=len(found),numerical_posterior_mass_fraction=volume/total,
                    found_patterns=found,missing_patterns=sorted(polys.keys()-keys)))
        rows.append(item);(a.out/'coverage.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=len(rows),total=16,seed=seed,complete_reference=item['numerical_volume_reference_complete'],
            positive_regions=item['positive_regions'],lp_calls=item['lp_calls'],seconds=item['elapsed_seconds'])),flush=True)
    print(json.dumps(dict(complete=True,tasks=len(rows),complete_references=sum(r['numerical_volume_reference_complete'] for r in rows))),flush=True)


if __name__=='__main__':main()
