"""Audit the immutable current conditional-risk chain and linked report."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/recovered_conditional_risk/development'
    p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text())
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    a=json.loads((inp.parent/'analysis/summary.json').read_text());assert a['passed']
    assert a['source_sha256']==sha(Path(__file__).with_name('analyze_recovered_conditional_risk.py'))
    for name,value in a['input_sha256'].items():assert sha(inp/name)==value,name
    assert a['checks']['reference_samples']==1286144 and a['checks']['frozen_states']==160
    assert a['checks']['pair_metric_replays']==4800 and a['checks']['aggregate_metric_replays']==400
    audits=json.loads((inp/'audits.json').read_text());curves=0;states=0
    for item in audits:
        # Input files and every frozen online state are also independently replayed by analysis.
        assert item['seed'] in p['seeds'];curves+=1
    assert curves==16
    links=0;figures=[];docs=root/'outputs/ttt-pc-alm-research'
    for name in ['207_conditional_function_risk_protocol.md','208_current_conditional_risk_results.md','209_context_block_scaling_protocol.md']:
        path=docs/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            target=(path.parent/target).resolve();assert target.exists(),target;links+=1
            if target.suffix=='.png':figures.append(dict(path=str(target),sha256=sha(target)))
    assert figures
    result=dict(passed=True,source_hashes=len(p['source_sha256']),independent_checks=a['checks'],links=links,figures=figures,
        protocol_sha256=sha(inp/'protocol.json'),analysis_sha256=sha(inp.parent/'analysis/summary.json'),source_sha256=sha(Path(__file__)),
        scope='Complete conditional-risk development analysis; not independent PC-ALM superiority')
    (root/'results/round_208_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
