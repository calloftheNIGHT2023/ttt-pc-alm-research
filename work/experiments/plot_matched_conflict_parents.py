"""Static diagnostic figure; does not modify any frozen runtime dependency."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args();root=args.project/'results/matched_conflict_parents/diagnostic';result=json.loads((root/'summary.json').read_text());rows=result['summary']
    labels=['Local credit','Canonical invalid','LP margin','Positive volume','Random 0','Random 1','Random 2','Random 3','BP + LP fill'];y=np.arange(len(rows))
    colors=['#177E89' if r['method']=='local' else '#D78032' if r['method']=='positive_volume' else '#748299' for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(12.5,5.7),sharey=True,gridspec_kw={'width_ratios':[1,1.3,1]})
    values=[[r['quality']['k24']['new_positive_modes'] for r in rows],[r['quality']['k24']['mean_ideal_truncation'] for r in rows],[r['geometry_candidates'] for r in rows]]
    titles=['New valid modes (K24)','Mean ideal truncation risk (K24)','Full geometry candidates']
    for ax,xx,title in zip(axes,values,titles):
        ax.barh(y,xx,color=colors,height=.64);ax.set_title(title,fontsize=11,pad=12);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_xlim(0,7);axes[0].set_xticks([0,2,4,6])
    axes[1].set_xscale('log');axes[1].set_xlim(1e-7,5e-5);axes[1].axvline(rows[0]['quality']['before']['mean_ideal_truncation'],color='#AA5555',linestyle='--',linewidth=1,label='No extension');axes[1].legend(fontsize=8,loc='lower right')
    axes[2].set_xlim(0,140);axes[2].set_xticks([0,40,80,120])
    fig.suptitle('Same ALM trajectory, same 65 parent slots, same K24 budget',fontsize=14,y=.97)
    fig.text(.5,.025,'16 old development contexts. Geometry count is not wall time. Ideal truncation is not measured query MSE.',ha='center',fontsize=9,color='#4A5260')
    fig.tight_layout(rect=[0,.065,1,.93]);path=root/'matched_parent_results.png';fig.savefig(path,dpi=170);plt.close(fig)
    audit={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),'figure_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (root/'figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit))


if __name__=='__main__':main()
