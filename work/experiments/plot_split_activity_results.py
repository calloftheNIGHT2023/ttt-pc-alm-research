"""Readable aligned plots from already-audited summaries; no new fitting."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    rows=json.loads((a.input/'summary.json').read_text())['summary'];pos=np.arange(len(rows))
    colors=['#137f79' if r['method'].startswith('alm') else '#d69532' if r['method'].startswith('adam') else '#71808c' for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(13,5.9),sharey=True,gridspec_kw={'width_ratios':[1.2,1,1]})
    fields=['mean_ideal_truncation','mean_query_mse','median_full_seconds']
    for ax,field in zip(axes,fields):
        ax.barh(pos,[r[field] for r in rows],color=colors);ax.grid(axis='x',alpha=.2)
        if field!='mean_ideal_truncation':
            values=[r[field] for r in rows];ax.set_xlim(0,max(values)*1.26)
            for y,value in zip(pos,values):ax.text(value+max(values)*.025,y,f'{value:.5f}' if field=='mean_query_mse' else f'{value:.3f}',va='center',fontsize=7)
    axes[0].set_yticks(pos,[r['method'] for r in rows],fontsize=8);axes[0].invert_yaxis()
    axes[0].set_xscale('symlog',linthresh=1e-7);axes[0].set_xlabel('Ideal missing-mode risk (symlog)')
    axes[1].set_xlabel('Mean unseen-query MSE');axes[2].set_xlabel('Median full stream time (s)')
    fig.suptitle('Activity-mode memory: all 13 configurations, 16 old development streams')
    fig.tight_layout();fig.savefig(a.input/'split_activity_results.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
