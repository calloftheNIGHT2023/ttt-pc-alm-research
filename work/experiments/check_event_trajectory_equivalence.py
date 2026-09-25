import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
old=json.loads((args.root/'development/episodes.json').read_text());new=json.loads((args.root/'frontier/episodes.json').read_text());pairs=[]
for row in old:
    if not row['method'].startswith('affine'):continue
    match=next(r for r in new if (r['method'],r['seed'],r['n_context'])==(row['method'],row['seed'],row['n_context']))
    diff=float(np.max(np.abs(np.array(row['anchor_output'])-np.array(match['anchor_output']))))
    pairs.append({'method':row['method'],'seed':row['seed'],'n_context':row['n_context'],'parameter_max_abs_delta':diff,
        'mse_abs_delta':abs(row['query_mse']-match['query_mse'])})
result={'stages_checked':len(pairs),'max_parameter_delta':max(r['parameter_max_abs_delta'] for r in pairs),
    'max_mse_delta':max(r['mse_abs_delta'] for r in pairs),'pairs':pairs,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
assert result['max_parameter_delta']<1e-8
(args.root/'event_trajectory_equivalence.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='pairs'}))
