"""Frozen admission summary and exact partition-size illustration."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results/light_tied_routing';inp=root/'diagnostic'
    summary=json.loads((inp/'summary.json').read_text());costs=json.loads((inp/'costs.json').read_text());complexity=json.loads((root/'complexity.json').read_text())
    methods=list(dict.fromkeys(r['method'] for r in summary['summary']));lookup={(r['method'],r['variant']):r for r in summary['summary']}
    labels=['ALM local','ALM energy','ALM residual','ALM BP','Adam16','Ordinary PC','No dual'];x=np.arange(len(methods));w=.25
    fig,axes=plt.subplots(1,3,figsize=(19,6),layout='constrained');colors=['#177e89','#80b1b6','#c18c40']
    for offset,variant,label,color in zip([-1,0,1],['tied_forward','tied_both','coordinate_forward'],['Tied forward','Tied forward + split','Coordinate scan'],colors):
        axes[0].bar(x+offset*w,[lookup[m,variant]['new_positive'] for m in methods],w,label=label,color=color)
    axes[0].set(title='New valid modes beyond each original path',ylabel='Task-mode pairs');axes[0].set_xticks(x,labels,rotation=28,ha='right');axes[0].legend(fontsize=9)
    for offset,key,label,color in [(-.5,'tied_seconds','Tied construction',colors[0]),(.5,'coordinate_seconds','Coordinate construction',colors[2])]:
        axes[1].bar(x+offset*.36,[sum(r[key] for r in costs if r['method']==m) for m in methods],.36,label=label,color=color)
    axes[1].set(title='Proposal construction only (16 old tasks)',ylabel='Seconds; excludes route/geometry/readout');axes[1].set_xticks(x,labels,rotation=28,ha='right');axes[1].legend(fontsize=9)
    rr=complexity['rows'];depth=[r['depth'] for r in rr]
    axes[2].semilogy(depth,[r['full_forward_first_bias_intervals'] for r in rr],'o-',color=colors[2],label='Full forward, first bias')
    axes[2].semilogy(depth,[r['all_hidden_tied_intervals'] for r in rr],'s-',color=colors[0],label='All local tied blocks')
    axes[2].set(title='Exact constructed partition-size separation',xlabel='Depth L',ylabel='Number of intervals');axes[2].set_xticks(depth);axes[2].legend(fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Lightweight typed routing: preserve useful modes, account for the simple strong control',fontsize=19)
    fig.text(.5,-.05,'Counts are diagnostic, not online query gains. The local family is smaller; fewer intervals do not guarantee better risk.',ha='center',fontsize=12)
    path=inp/'light_routing_admission.png';fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    audit=dict(source_sha256=sha(Path(__file__)),summary_sha256=sha(inp/'summary.json'),costs_sha256=sha(inp/'costs.json'),complexity_sha256=sha(root/'complexity.json'),figure_sha256=sha(path))
    (inp/'figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
