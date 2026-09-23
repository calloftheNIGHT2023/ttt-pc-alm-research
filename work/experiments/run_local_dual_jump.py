"""Frozen all272 prefix16 states; float proposals and exact branch evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import local_dual_jump as jump
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/local_dual_jump/screen';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    parent=root/'results/round_250_audit.json';old=json.loads(parent.read_text());hashes=dict(old['source_sha256'])
    for n,h in hashes.items():assert sha(src/n)==h,n
    design=root/'outputs/ttt-pc-alm-research/251_local_dual_jump_design.md';assert sha(design)==old['report_sha256'][design.name]
    hashes.update({n:sha(src/n) for n in ['local_dual_jump.py',Path(__file__).name]});verification=jump.verify()
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=sha(design),verification=verification,seeds=list(range(5900000,5900016)),
        prefix=16,source_method='nodual128',restarts=17,taus=jump.TAUS,selection='minimum tau then j,i,k; float proposed and ideal+actual-binary-write exact verified',query_targets_accessed=False)
    dump(out/'protocol.json',p);source=root/'results/cold_stagnation_switch/development';saved={(r['seed'],r['method']):r for r in json.loads((source/'support_rows.json').read_text())};counts=Counter();files=[];selected=[];start=time.perf_counter()
    for seed in p['seeds']:
        r=saved[seed,'nodual128'];file=source/r['file'];assert sha(file)==r['file_sha256'];a=np.load(file);assert not np.any(a['u'][16]);rows=[];begin=time.perf_counter()
        for restart in range(17):
            record=jump.scan(a['x'],a['b'][16,restart],a['h'][16,:,restart]);record.update(seed=seed,restart=restart)
            rows.append(record);counts['states']+=1;counts['blocks']+=len(record['blocks'])
            for block in record['blocks']:
                for trial in block['trials']:
                    counts['tau_trials']+=1;counts['float_proposals']+=trial['float_decision']['accepted'];counts['exact_positive_trials']+=trial['exact_decision']['accepted'];counts['accepted_trials']+=trial['accepted']
                    counts['float_false_positive']+=trial['float_decision']['accepted'] and not trial['exact_decision']['accepted']
                    counts['exact_only_missed']+=trial['exact_decision']['accepted'] and not trial['float_decision']['accepted']
                    counts['rounded_write_rejected']+=trial['float_decision']['accepted'] and trial['exact_decision']['accepted'] and not trial['rounded_write']['accepted']
            if record['selected'] is not None:selected.append(dict(seed=seed,restart=restart,**record['selected']));counts['selected_states']+=1
        path=out/f'{seed}.json';dump(path,rows);files.append(dict(seed=seed,source_file=r['file'],source_sha256=r['file_sha256'],file=path.name,sha256=sha(path),seconds=time.perf_counter()-begin))
        dump(out/'files.json',files);dump(out/'selected.json',selected);print(json.dumps(dict(seed=seed,selected=sum(r['selected'] is not None for r in rows),seconds=files[-1]['seconds'])),flush=True)
    for n,h in hashes.items():assert sha(src/n)==h,n
    result=dict(passed=True,counts=counts,seconds=time.perf_counter()-start,protocol_sha256=sha(out/'protocol.json'),files_sha256=sha(out/'files.json'),selected_sha256=sha(out/'selected.json'),
        selected_tau_counts={str(t):sum(r['tau']==t for r in selected) for t in jump.TAUS},query_targets_accessed=False)
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
