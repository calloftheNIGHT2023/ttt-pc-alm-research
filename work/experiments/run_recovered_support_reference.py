"""Bounded support-only reference for current recovered-comparison tasks.

This is an offline diagnostic. It never changes an online state and does not
read its query inputs, query predictions, teacher parameters, or query losses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import enumerate_support_modes as reference


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/recovered_online_comparison/development';parent=json.loads((inp/'protocol.json').read_text())
    assert json.loads((root/'results/round_204_audit.json').read_text())['passed']
    hashes=dict(parent['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for path in [Path(__file__),Path(reference.__file__)]:hashes[path.name]=sha(path)
    cfgs=[c for c in parent['configs'] if c['family']!='regression']
    p=dict(source_sha256=hashes,seeds=parent['seeds'],configs=cfgs,n_context=4,depth=4,max_lp_per_task=50000,
        max_search_seconds_per_task=180,verification=reference.verify(),parent_protocol_sha256=sha(inp/'protocol.json'),
        scope='Offline support-only current-development coverage; no online state changes or query-based selection; numerical volume reference, not exact posterior integration')
    out=root/'results/recovered_support_reference/development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8');rows=[]
    for seed in p['seeds']:
        source=inp/f'state_{seed}_{parent["primary"]}_0_4.npz'
        with np.load(source) as z:x=z['x'].copy();v=z['v'].copy()
        assert len(x)==4
        result=reference.Reference(x,v,max_lp=p['max_lp_per_task'],max_seconds=p['max_search_seconds_per_task']).enumerate()
        filename=f'reference_{seed}.json';file=out/filename
        file.write_text(json.dumps(dict(seed=seed,x=x.tolist(),v=v.tolist(),**result),indent=2),encoding='utf-8')
        row=dict(seed=seed,reference_file=filename,reference_sha256=sha(file),support_state_file=source.name,support_state_sha256=sha(source),
            complete=result['numerical_volume_reference_complete'],enumeration_completed=result['enumeration_completed'],
            unresolved_final_regions=len(result['final_geometry_unresolved']),lp_calls=result['lp_calls'],
            exact_rejections=result['certified_lp_rejections'],positive_regions=len(result['positive_regions']),
            seconds=result['elapsed_seconds'],coverage=[])
        if row['complete']:
            polys={r['pattern']:r for r in result['positive_regions']};total=sum(r['volume'] for r in polys.values())
            for cfg in cfgs:
                detail=inp/f'detail_{seed}_{cfg["name"]}.json';meta=json.loads(detail.read_text())
                found=set(meta['positive_mode_keys']);assert found<=set(polys)
                row['coverage'].append(dict(method=cfg['name'],detail_file=detail.name,detail_sha256=sha(detail),
                    found_patterns=sorted(found),missing_patterns=sorted(set(polys)-found),positive_regions=len(found),
                    mass_fraction=sum(polys[k]['volume'] for k in found)/total))
        rows.append(row);(out/'coverage.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=len(rows),total=len(p['seeds']),seed=seed,complete=row['complete'],
            positive_regions=row['positive_regions'],lp_calls=row['lp_calls'],seconds=row['seconds'])),flush=True)
    result=dict(execution_complete=True,tasks=len(rows),complete_references=sum(r['complete'] for r in rows),
        lp_calls=sum(r['lp_calls'] for r in rows),exact_rejections=sum(r['exact_rejections'] for r in rows),
        source_hashes=len(hashes),scope=p['scope'])
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
