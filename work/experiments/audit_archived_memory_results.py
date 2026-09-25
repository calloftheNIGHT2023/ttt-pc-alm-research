import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--original',type=Path,required=True);a=p.parse_args()
proto=json.loads((a.results/'protocol.json').read_text());rows=json.loads((a.results/'episodes.json').read_text());old=json.loads((a.original/'episodes.json').read_text())
assert len(rows)==proto['count']*len(proto['configs'])*len(proto['stages'])
assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto['source_sha256'].items())
lookup={(r['method'],r['seed'],r['n_context']):r for r in old};checks=[];archive=[]
for c in proto['configs']:
    group=[r for r in rows if r['method']==c['name']]
    if not c['archive']:
        mse=[];anchor=[];modes=[]
        for r in group:
            previous=lookup[(c['name'],r['seed'],r['n_context'])]
            mse.append(abs(r['query_mse']-previous['query_mse']));anchor.append(float(np.max(np.abs(np.array(r['anchor_output'])-previous['anchor_output']))))
            modes.append(r['positive_volume_regions']==previous['positive_volume_regions'])
        assert max(mse)==0 and max(anchor)==0 and all(modes)
        checks.append(dict(method=c['name'],stage_pairs=len(group),max_mse_difference=max(mse),max_anchor_difference=max(anchor),all_region_counts_equal=all(modes)))
    archive.append(dict(method=c['name'],archive_active_stages=sum(r['archive_active'] for r in group),
                        max_archive_numeric_key_bytes=max(r['archive_numeric_key_bytes'] for r in group),
                        max_combined_patterns=max(r['combined_patterns'] for r in group),max_geometry_bank_patterns=max(r['geometry_bank_patterns'] for r in group),
                        total_archive_seconds=sum(r['archive_seconds'] for r in group),total_screen_seconds=sum(r['screen_seconds'] for r in group)))
result=dict(passed=True,source_hashes_match=True,unchanged_no_archive_results=checks,archive_diagnostics=archive,
            scope='array/key subtotal is not peak memory; online times include monitoring work')
(a.results/'reproduction_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
