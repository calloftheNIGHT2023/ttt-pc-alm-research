"""Post-seal 363 equivalence and actual-pool attribution, not new selection."""
from collections import defaultdict
from pathlib import Path
import hashlib
import numpy as np
from report_search_radius_development_v1 import read,sha,save,complete


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];base=root/'results/frontier_online';complete(base/'development_audit_v1')
    pred=base/'development_predictions_v1';p=read(pred/'protocol.json');rows=read(pred/'rows.json')
    index={(r['seed'],r['method']):r for r in rows};names=[c['name'] for c in p['configs']]
    arrays={};pools={};counts=0
    for seed in p['seeds']:
        for name in names:
            row=index[seed,name];assert sha(root/row['file'])==row['sha256']
            with np.load(root/row['file'],allow_pickle=False) as z:
                payload=[]
                for key in ['points','allocation','prediction']:
                    a=z[key];payload.append(key.encode()+a.dtype.str.encode()+str(a.shape).encode()+a.tobytes());counts+=1
            arrays[seed,name]=hashlib.sha256(b''.join(payload)).hexdigest()
            pools[seed,name]=tuple(read(root/row['metadata_file'])['metadata'].get('positive_modes',[]))
    by_prediction=defaultdict(list);by_pool=defaultdict(list)
    for name in names:
        by_prediction[tuple(arrays[s,name] for s in p['seeds'])].append(name)
        by_pool[tuple(pools[s,name] for s in p['seeds'])].append(name)
    comparisons=[]
    for left in names:
        for right in names:
            if left==right:continue
            comparisons.append(dict(left=left,right=right,
                bitwise_predictor_equal_tasks=sum(arrays[s,left]==arrays[s,right] for s in p['seeds']),
                same_positive_pool_tasks=sum(pools[s,left]==pools[s,right] for s in p['seeds'])))
    out=base/'attribution_v1';out.mkdir(parents=True,exist_ok=False)
    save(out/'pairs.json',comparisons)
    result=dict(passed=True,tasks=len(p['seeds']),array_hashes=counts,methods=len(names),
        identical_predictor_groups=list(by_prediction.values()),identical_positive_pool_groups=list(by_pool.values()),
        prediction_rows_sha256=sha(pred/'rows.json'),independent_audit_sha256=sha(base/'development_audit_v1/summary.json'),
        posthoc_descriptive_attribution=True,core_research_goal_complete=False,outputs_sha256={'pairs.json':sha(out/'pairs.json')})
    save(out/'summary.json',result);print(result,flush=True)
