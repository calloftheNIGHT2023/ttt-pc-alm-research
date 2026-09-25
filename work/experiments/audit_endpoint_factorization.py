"""Frozen-source/artifact audit and scientific figure for endpoint reduction."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/endpoint_factorization/development';out=root/'results/endpoint_factorization/analysis'
    p=json.loads((inp/'protocol.json').read_text());s=json.loads((inp/'summary.json').read_text());assert s['passed']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert sha(inp/'protocol.json')==s['protocol_sha256']
    assert s['source_sha256']==sha(Path(__file__).with_name('benchmark_endpoint_factorization.py'))
    audits=json.loads((inp/'audits.json').read_text());rows=json.loads((inp/'rows.json').read_text())
    assert len(audits)==s['banks']==96 and len(rows)==s['rows']==576
    assert sum(a['values_bitwise'] for a in audits)==s['float_values_bitwise']==1889166
    assert sum(a['proofs'] for a in audits)==s['exact_proof_replays']==732
    for a in audits:
        path=root/f'results/light_h2_credit/full_bank_ceiling/bank_{a["seed"]}_{a["method"]}.npz'
        assert sha(path)==a['data_sha256']
    out.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(12.2,4.9),layout='constrained')
    x=np.arange(6);labels=['ALM\njoint','ALM\nresidual','Adam\njoint','Adam\nresidual','PC','No dual']
    axes[0].bar(x-.18,[r['mean_seconds']['full_candidates']*1000 for r in s['summaries']],.35,label='Original candidates',color='#89949e')
    axes[0].bar(x+.18,[r['mean_seconds']['endpoints']*1000 for r in s['summaries']],.35,label='Duplicate-free endpoints',color='#267e94')
    axes[0].set_xticks(x,labels);axes[0].set_ylabel('Cold float-bound component (ms)');axes[0].set_title('96 real pools; identical float values')
    n=np.arange(1,25);axes[1].plot(n,n*(n+2),label='Old: n(n+2)',color='#89949e')
    axes[1].plot(n,3*n,label='Hidden layers: at most 3n',color='#267e94')
    axes[1].plot(n,2*n,label='First layer: 2n',color='#399577',linestyle='--')
    axes[1].set_xlabel('Support observations n');axes[1].set_ylabel('Candidate-observation terms per direction')
    axes[1].set_title('Operation-count model, not measured scaling')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    fig.suptitle('Fixed-box tent structure: exact candidate-set reduction, not a general-network claim')
    figure=out/'endpoint_factorization.png';fig.savefig(figure,dpi=170);plt.close(fig)
    path=root/'outputs/ttt-pc-alm-research/199_endpoint_reduction_design.md';links=0
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
        if '://' in target or target.startswith('#'):continue
        assert (path.parent/target).resolve().exists(),target;links+=1
    result=dict(passed=True,sources=len(p['source_sha256']),banks=len(audits),timings=len(rows),
        float_values_bytewise=s['float_values_bitwise'],unchanged_exact_proofs=s['exact_proof_replays'],
        links=links,figure_sha256=sha(figure),protocol_sha256=sha(inp/'protocol.json'),source_sha256=sha(Path(__file__)),
        scope='Source and artifact audit of frozen component identity experiment; online combination remains untested')
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
