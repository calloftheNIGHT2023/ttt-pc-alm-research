"""258B frozen, query-free first-crossing certificates on all old states."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import minimum_sufficient_dual as core
from run_local_dual_jump_v2 import dump
from run_multiplier_fixed_point_screen import sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/minimum_sufficient_dual/threshold';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    parent=root/'results/minimum_sufficient_dual/short_resources';p=json.loads((parent/'protocol.json').read_text());summary=json.loads((parent/'summary.json').read_text());assert summary['passed'] and sha(parent/'protocol.json')==summary['protocol_sha256']
    hashes=dict(p['source_sha256'])
    for n,h in hashes.items():assert sha(src/n)==h,n
    hashes.update({n:sha(src/n) for n in ['minimum_sufficient_dual.py',Path(__file__).name]})
    screen=root/'results/local_dual_jump/screen_v2';screen_summary=json.loads((screen/'summary.json').read_text());assert sha(screen/'files.json')==screen_summary['files_sha256']
    primitive=core.verify()
    protocol=dict(source_sha256=hashes,parent_summary_sha256=sha(parent/'summary.json'),screen_files_sha256=sha(screen/'files.json'),
        states=272,direction='exact interpretation of old actual binary u / old grid tau',bisections=32,max_rounding_attempts=17,
        scope='same selected j,i; every original state retained; no new query targets or reserved tasks',query_targets_accessed=False,primitive=primitive)
    dump(out/'protocol.json',protocol);counts=Counter();files=[];rows=[];start=time.perf_counter()
    for item in json.loads((screen/'files.json').read_text()):
        source=root/'results/cold_stagnation_switch/development'/item['source_file'];assert sha(source)==item['source_sha256'];assert sha(screen/item['file'])==item['sha256']
        with np.load(source) as a:x=a['x'];bb=a['b'][16];hh=a['h'][16]
        records=[]
        for old in json.loads((screen/item['file']).read_text()):
            r=old['restart'];begin=time.perf_counter();actual=core.minimum_event(x,bb[r],hh[:,r],old['selected']);elapsed=time.perf_counter()-begin
            rec=dict(seed=item['seed'],restart=r,old_event=old['selected'],**actual);records.append(rec);counts['states']+=1
            row=dict(seed=item['seed'],restart=r,seconds=elapsed,fallbacks=actual['fallbacks'],selected=actual['selected'])
            if actual['selected'] is not None:
                counts['selected']+=1;counts['fallback_states']+=actual['fallbacks']>0;counts['fallback_attempts']+=actual['fallbacks'];counts['pieces']+=len(actual['certificate']['pieces'])
                new=actual['selected'];before=old['selected'];counts['same_competitor']+=new['k']==before['k'];counts['strict_tau_decrease']+=new['tau']<before['tau']
                row.update(tau_ratio=new['tau']/before['tau'],norm_ratio=float(np.linalg.norm(new['u'])/np.linalg.norm(before['u'])),activity_difference=new['z']-before['z'])
            rows.append(row)
        filename=f"{item['seed']}_thresholds.json";dump(out/filename,records);files.append(dict(seed=item['seed'],file=filename,sha256=sha(out/filename),source_file=item['source_file'],source_sha256=item['source_sha256']))
        print(json.dumps(dict(seed=item['seed'],counts=counts)),flush=True)
    assert counts['states']==272 and counts['selected']==260
    dump(out/'files.json',files);dump(out/'rows.json',rows)
    ans=dict(passed=True,counts=counts,seconds=time.perf_counter()-start,protocol_sha256=sha(out/'protocol.json'),files_sha256=sha(out/'files.json'),rows_sha256=sha(out/'rows.json'),query_targets_accessed=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()
