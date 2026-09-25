"""Plot the completed frozen tied-block admission without rerunning solvers."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results/tied_local_block/diagnostic'
    summary=json.loads((root/'summary.json').read_text());costs=json.loads((root/'costs.json').read_text());lookup={(r['method'],r['variant']):r for r in summary['summary']}
    methods=list(dict.fromkeys(r['method'] for r in summary['summary']));labels=['ALM local','ALM energy path','ALM residual','ALM BP credit','Adam16','Ordinary PC','No dual']
    x=np.arange(len(methods));w=.26;colors=['#8290a5','#167e89','#bc705d'];fig,axes=plt.subplots(2,2,figsize=(16,10),layout='constrained')
    for off,var,label,color in zip([-1,0,1],['scalar_cuts','union_cuts','union_energy'],['Scalar + cuts','Union + cuts','Union, energy only'],colors):
        rates=[lookup[m,var]['accepted']/lookup[m,var]['inputs']*100 for m in methods]
        axes[0,0].bar(x+off*w,rates,w,label=label,color=color)
        fresh=[lookup[m,var]['new_positive_vs_feedback'] for m in methods]
        bars=axes[0,1].bar(x+off*w,fresh,w,color=color)
        for bar,value in zip(bars,fresh):axes[0,1].text(bar.get_x()+w/2,value+.025,str(value),ha='center',fontsize=9,color=color)
    axes[0,0].set(title='One-step repair acceptance',ylabel='Accepted old triggered inputs (%)',ylim=(0,112));axes[0,0].legend(fontsize=10)
    axes[0,1].set(title='New valid modes beyond each original feedback path',ylabel='Task-mode pairs (not independent queries)',ylim=(0,max(1,max(r['new_positive_vs_feedback'] for r in summary['summary'])+1)))
    tied=[lookup[m,'union_cuts']['tied'] for m in methods]
    rescued=[sum(r['scalar_empty_rescued'] for r in costs if r['method']==m) for m in methods]
    axes[1,0].bar(x-w/2,tied,w,color=colors[1],label='Tied candidate selected');axes[1,0].bar(x+w/2,rescued,w,color='#d0a24f',label='Previously empty scalar pool rescued')
    axes[1,0].set(title='What the extra proposals changed',ylabel='Triggered input count');axes[1,0].legend(fontsize=10)
    millis=[1000*sum(r['shared_bank_seconds'] for r in costs if r['method']==m)/lookup[m,'union_cuts']['inputs'] for m in methods]
    bars=axes[1,1].bar(x,millis,.6,color=colors[0])
    for bar,value in zip(bars,millis):axes[1,1].text(bar.get_x()+.3,value+.5,f'{value:.1f}',ha='center',fontsize=10)
    axes[1,1].set(title='Shared candidate construction cost (diagnostic only)',ylabel='Milliseconds per old triggered input',ylim=(0,max(millis)*1.2))
    for ax in axes.ravel():
        ax.set_xticks(x,labels,rotation=22,ha='right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(f"Exact tied-block admission: {summary['original_state_replays']} frozen paths / {summary['old_records_replayed']:,} inputs",fontsize=19)
    fig.text(.5,-.035,'Higher repair feasibility is not yet better unseen-query prediction. Counts reuse old paths; this is not an online timing comparison.',ha='center',fontsize=12)
    figure=root/'tied_block_admission.png';fig.savefig(figure,dpi=160,bbox_inches='tight');plt.close(fig)
    audit=dict(source_sha256=sha(Path(__file__)),summary_sha256=sha(root/'summary.json'),costs_sha256=sha(root/'costs.json'),figure_sha256=sha(figure))
    (root/'figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
