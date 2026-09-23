"""258A: selected-event equality for all saved states and extra bounded cases."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import local_dual_jump_short as short
import local_dual_jump_transfer as transfer
from run_local_dual_jump_v2 import dump
from run_multiplier_fixed_point_screen import sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/minimum_sufficient_dual/short_primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    parent=root/'results/round_257_audit.json';old=json.loads(parent.read_text());hashes=dict(old['source_sha256'])
    for n,h in hashes.items():assert sha(src/n)==h,n
    hashes.update({n:sha(src/n) for n in ['local_dual_jump_short.py',Path(__file__).name]});design=root/'outputs/ttt-pc-alm-research/258_minimum_sufficient_dual_design.md';assert sha(design)==old['report_sha256'][design.name]
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=sha(design),extra_random_cases=64,random_seed=258731,query_targets_accessed=False)
    dump(out/'protocol.json',p);screen=root/'results/local_dual_jump/screen_v2';transferdir=root/'results/local_dual_jump/transfer';lookup={(r['seed'],r['restart'],r['method']):r for r in json.loads((transferdir/'support_rows.json').read_text())}
    counts=Counter();rows=[];begin=time.perf_counter()
    with transfer.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        for item in json.loads((screen/'files.json').read_text()):
            source=root/'results/cold_stagnation_switch/development'/item['source_file'];assert sha(source)==item['source_sha256'];assert sha(screen/item['file'])==item['sha256']
            with np.load(source) as a:x=a['x'];v=a['v'];bb=a['b'][16];hh=a['h'][16];oldbest=a['best'][16]
            for saved in json.loads((screen/item['file']).read_text()):
                r=saved['restart'];start=time.perf_counter();actual=short.scan(x,bb[r],hh[:,r]);elapsed=time.perf_counter()-start;assert actual['selected']==saved['selected']
                counts.update(actual['counts']);counts['states']+=1;counts['selected']+=actual['selected'] is not None;counts['logical_trials']+=sum(len(b['trials']) for b in actual['blocks'])
                for method in ['dual_jump','activity_only','reorder_no_dual']:
                    got,_=transfer.run(bb[r],hh[:,r],oldbest[r],x,v,actual['selected'],method);ref=lookup[item['seed'],r,method];assert sha(transferdir/ref['file'])==ref['sha256']
                    with np.load(transferdir/ref['file']) as a:
                        for key,value in got.items():assert value.tobytes()==a[key].tobytes();counts['atomic_arrays']+=1
                rows.append(dict(seed=item['seed'],restart=r,selected=actual['selected'],counts=actual['counts'],logical_trials=sum(len(b['trials']) for b in actual['blocks']),seconds=elapsed))
            print(json.dumps(dict(seed=item['seed'],states=counts['states'])),flush=True)
    rng=np.random.default_rng(p['random_seed'])
    for case in range(p['extra_random_cases']):
        x=rng.uniform(0,1,4);b=rng.uniform(-.12,.12,4);h=rng.uniform(0,1,(4,4))
        if case%8==0:h[case%4]=0.
        expected=short.original.scan(x,b,h);actual=short.scan(x,b,h);assert actual['selected']==expected['selected'];counts['extra_random_equal']+=1
    dump(out/'rows.json',rows);result=dict(passed=True,counts=counts,seconds=time.perf_counter()-begin,source_sha256=hashes,
        protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),query_targets_accessed=False)
    dump(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)

if __name__=='__main__':main()
