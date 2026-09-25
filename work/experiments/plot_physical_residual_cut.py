"""Plot-only comparison of mathematical admission and its limits."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args();out=args.project/'results/physical_residual_cut/diagnostic'
    data=json.loads((out/'summary.json').read_text());physical=[r for r in data['summary'] if r['variant']=='physical_all_old_cuts']
    oldroot=args.project/'results/feedback_repair_paths/diagnostic';previous=json.loads((oldroot/'summary.json').read_text())['summary'];lookup={r['method']:r for r in previous}
    y=np.arange(4);labels=['ALM local','ALM energy path','Same ALM / BP','Adam16 / BP'];fig,axes=plt.subplots(1,3,figsize=(13,5),sharey=True)
    values=[100*r['accepted']/r['inputs'] for r in physical]
    axes[0].barh(y,values,color='#177E89');axes[0].set_xlim(0,100);axes[0].set_xlabel('Exact feasible repair / old triggers (%)')
    for i,r in enumerate(physical):axes[0].text(values[i]+1,i,f'{r["accepted"]}/{r["inputs"]}',va='center',fontsize=9)
    oldrate=[100*lookup[r['method']]['closed_conflict_after']/lookup[r['method']]['accepted'] for r in physical]
    axes[1].barh(y-.16,oldrate,height=.30,color='#D79434',label='Old adopted repair')
    axes[1].barh(y+.16,[0]*4,height=.30,color='#177E89',label='New accepted repair')
    for i in y:axes[1].text(1,i+.16,'0%',color='#177E89',va='center',fontsize=9)
    axes[1].set_xlim(0,110);axes[1].set_xlabel('Still in old closed conflict (%)');axes[1].legend(loc='lower right',fontsize=8)
    axes[2].barh(y,[r['new_positive_vs_feedback'] for r in physical],color='#177E89');axes[2].set_xlim(0,1);axes[2].set_xticks([0,1]);axes[2].set_xlabel('Extra valid modes vs original feedback')
    for i in y:axes[2].text(.03,i,'0',va='center')
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();fig.suptitle('Physical cuts fix closed-boundary semantics; task gain still requires dynamic testing',fontsize=12)
    fig.text(.5,.025,'Middle panel is conditional on acceptance. Rejected proposals leave the old state unchanged. These are offline proposals, not query gains.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(out/'physical_cut_admission.png',dpi=170);plt.close(fig)
    audit=dict(source_sha256=sha(Path(__file__)),input_sha256=sha(out/'summary.json'),old_path_summary_sha256=sha(oldroot/'summary.json'),figure_sha256=sha(out/'physical_cut_admission.png'))
    (out/'figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit))


if __name__=='__main__':main()
