"""Plot the exact finite-search witness without claiming ALM exclusivity."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    out=root/'results/context_block_scaling/reachability';p=json.loads((out/'integer_audit.json').read_text());assert p['passed']
    witness=p['strict_certificates'];assert len(witness)==1
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');w=witness[0]
    axes[0].bar(['256','4096','16384'],list(w['exact_minimum_edits'].values()),color=['#8cabb2','#5d919d','#277d88'])
    axes[0].axhline(2,color='#d58228',linestyle='--',label='Frozen H2 maximum: 2 edits')
    axes[0].set(xlabel='Every prior point included, plus zero anchor',ylabel='Exact minimum branch-code edits',title='One support-feasible mode outside the search radius',ylim=(0,6))
    axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
    axes[1].bar(['ALM','Adam60','PC','No dual','Direct'],[1,1,1,0,0],color=['#277d88','#778aab','#9d89a8','#bbbbbb','#bbbbbb'])
    axes[1].set(ylabel='Mode found in frozen output (yes / no)',yticks=[0,1],yticklabels=['No','Yes'],ylim=(0,1.2),title='A shared internal-update gain, not ALM-only')
    fig.suptitle('Exact integer audit: task 5920001, 24 observed support pairs; no query answers',fontsize=11)
    path=out/'finite_search_witness.png';fig.savefig(path,dpi=170);plt.close(fig)
    result=dict(passed=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256((out/'integer_audit.json').read_bytes()).hexdigest(),figure_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
