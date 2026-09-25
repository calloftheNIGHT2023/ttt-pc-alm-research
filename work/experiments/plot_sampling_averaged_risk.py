"""Show mean prediction bias, sampling variance and one frozen realization."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    p=json.loads((args.input/'protocol.json').read_text());s=json.loads((args.input/'summary.json').read_text());assert s['complete'] and s['context_methods']==320
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=s['summary'];y=np.arange(len(rows));bias=np.array([r['ideal_truncation'] for r in rows]);sampling=np.array([r['expected_sampling'] for r in rows]);fixed=np.array([r['fixed_state_excess'] for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(15,10),sharey=True)
    for ax in axes:
        ax.barh(y,bias,label='Missing-region squared bias',color='#177e89')
        ax.barh(y,sampling,left=bias,label='Expected 512-particle variance',color='#d39c41')
        ax.scatter(fixed,y,marker='x',color='#222222',s=35,label='Frozen seed 6173 prediction',zorder=4)
        ax.set_xlabel('First-write conditional excess risk');ax.grid(axis='x',alpha=.2);ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    axes[0].set_yticks(y,[r['method'] for r in rows],fontsize=8);axes[0].invert_yaxis();axes[0].set_title('All 20 methods')
    axes[1].set_xlim(0,.0002);axes[1].set_title('Zoom to low-error methods (larger values clipped)');axes[1].legend(fontsize=8,loc='lower right')
    fig.suptitle('Separate learned region coverage from particle-seed fluctuation');fig.tight_layout();fig.savefig(args.input/'sampling_averaged_risk.png',dpi=170);plt.close(fig)
    result=dict(source_sha256=sha(Path(__file__)),summary_sha256=sha(args.input/'summary.json'),figure_sha256=sha(args.input/'sampling_averaged_risk.png'))
    (args.input/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
