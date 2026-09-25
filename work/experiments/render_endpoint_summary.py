"""Uncluttered display of precomputed, unadjusted paired development intervals."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    out=root/'results/online_endpoint_h2/analysis';path=out/'summary.json';p=json.loads(path.read_text());assert p['complete']
    lookup={r['comparator']:r for r in p['alm_endpoint_comparisons']}
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.8),layout='constrained')
    targets=['alm_c5','alm_native_full_c5','alm_native_full_c5_interval','alm_residual_full_c5_interval_endpoints','direct4096_c5']
    labels=['ALM without credit (C5)','ALM credit + rational','ALM credit + interval','ALM residual + endpoints','Direct prior4096 (C5)']
    for ax,metric,title in zip(axes,['first_time','time'],['First online stage','Full four-stage stream']):
        for i,target in enumerate(targets):
            row=lookup[target][metric];mean=row['mean']*1000;low,high=np.array(row['ci95'])*1000
            ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',color='#257b68' if high<0 else '#777777',capsize=4)
        ax.axvline(0,color='#b26137',linestyle='--',linewidth=1)
        ax.set_yticks(range(len(labels)),labels if ax is axes[0] else ['']*len(labels));ax.invert_yaxis()
        ax.set(xlabel='Endpoint-interval ALM minus comparator (ms)',title=title)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Same ALM predictions; timing benefit depends on comparison and horizon',fontsize=13)
    fig.supxlabel('16 reused development tasks; task-paired bootstrap 95% intervals, not multiplicity-adjusted. Negative is faster.',fontsize=9)
    file=out/'endpoint_paired_comparisons.png';fig.savefig(file,dpi=170);plt.close(fig)
    result=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        figure_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),scope='Display only; no new statistics or adaptive selection')
    (out/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
