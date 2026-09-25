import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--original',type=Path,required=True);a=p.parse_args()
rows=json.loads((a.results/'episodes.json').read_text());old=json.loads((a.original/'episodes.json').read_text());proto=json.loads((a.results/'protocol.json').read_text())
assert len(rows)==proto['count']*len(proto['configs'])*len(proto['stages'])
assert all(hashlib.sha256(Path(__file__).with_name(k).read_bytes()).hexdigest()==v for k,v in proto['source_sha256'].items())
lookup={(r['method'],r['seed'],r['n_context']):r for r in old};checks=[]
for name in ['alm64_wide','alm64_priorbox','lbfgs64_wide','lbfgs128_priorbox']:
    gaps=[];anchor=[];modes=[]
    for r in rows:
        if r['method']!=name:continue
        o=lookup[(name,r['seed'],r['n_context'])];gaps.append(abs(r['query_mse']-o['query_mse']))
        anchor.append(float(np.max(np.abs(np.array(r['anchor_output'])-o['anchor_output']))))
        modes.append(r['positive_volume_regions']==o['positive_volume_regions'])
    assert max(gaps)==0 and max(anchor)==0 and all(modes)
    checks.append(dict(method=name,stage_pairs=len(gaps),max_query_mse_difference=max(gaps),max_anchor_difference=max(anchor),all_region_counts_equal=all(modes)))
result=dict(passed=True,source_hashes_match=True,old_frozen_results_reproduced=checks)
(a.results/'reproduction_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
