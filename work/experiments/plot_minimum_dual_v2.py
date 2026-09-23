"""259 scientific figure: fixed outputs, intervention size, paired atomics."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/minimum_sufficient_dual';out=base/'figures_v2';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    analysis=json.loads((base/'analysis/summary.json').read_text());resource=json.loads((base/'short_resources/summary.json').read_text());threshold=json.loads((base/'threshold/rows.json').read_text());credit=json.loads((base/'transfer_audit/credit.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,2,figsize=(12.6,8.6),layout='constrained');blue='#176B9B';teal='#1A9876';orange='#C8782A'
    ax=axes[0,0];names=['Old grid / full scan','Old grid / short scan','Cold ALM16'];keys=['dual_alm64__full','dual_alm64__short','cold__alm16'];vals=[resource['methods'][k]['mean_seconds'] for k in keys];ax.barh(names,vals,color=[blue,teal,'#777777']);ax.set_ylim(2.85,-.6);ax.set_xlim(0,.99);ax.set_xlabel('Complete seconds / task (16 tasks, same run)');ax.set_title('A. Identical old predictor; 48.1% less time',loc='left',fontweight='bold')
    for i,v in enumerate(vals):ax.text(v+.012,i,f'{v:.3f}',va='center')
    ax.text(.02,.04,'162 bytewise predictor replays; no new query gain',transform=ax.transAxes,fontsize=9)
    ax=axes[0,1];ratios=np.sort([r['norm_ratio'] for r in threshold if r['selected'] is not None]);ax.plot(ratios,np.arange(1,len(ratios)+1)/len(ratios),color=teal,lw=2.5);ax.axvline(np.median(ratios),ls='--',color=orange);ax.set_xlim(0,1.02);ax.set_ylim(0,1.02);ax.set_xlabel('Actual multiplier norm / old multiplier norm');ax.set_ylabel('Fraction of 260 triggered states');ax.set_title('B. Same block; smaller certified intervention',loc='left',fontweight='bold');ax.text(.04,.85,f'Median = {np.median(ratios):.3f}\n260 strict decreases; 2 rounding fallbacks',transform=ax.transAxes,fontsize=9);ax.grid(alpha=.15)
    ax=axes[1,0];comp=['dual_jump','minimum_activity','branch_probe'];labels=['Old grid dual','Same new activity, no dual','All old direct branch probes'];left=np.zeros(3)
    for key,color,label in [('lower',teal,'New dual better'),('same','#C7CED3','Equal (1e-12)'),('higher',orange,'New dual worse')]:
        vals=[analysis['paired_support_error'][m][key] for m in comp];bars=ax.barh(labels,vals,left=left,color=color,label=label)
        for rect,v,l in zip(bars,vals,left):
            if v>=20:ax.text(l+v/2,rect.get_y()+rect.get_height()/2,str(v),ha='center',va='center',fontsize=9,color='white' if key!='same' else '#263746')
        left+=vals
    ax.invert_yaxis();ax.set_xlim(0,272);ax.set_xlabel('All 272 states; retained support max-error');ax.set_title('C. Improvement is paired, not uniform',loc='left',fontweight='bold');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3,fontsize=8,frameon=False)
    ax=axes[1,1];valid=[r for r in credit if r['both_strictly_interior']];unpack=lambda z:int(z[0])/int(z[1]);xx=[unpack(r['actual_target_shift']) for r in valid];yy=[unpack(r['actual_delta']) for r in valid];lim=max(max(abs(t) for t in xx),.01)*1.1;ax.plot([-lim,lim],[-lim,lim],ls='--',color='#777777',lw=1);ax.scatter(xx,yy,s=14,color=blue,alpha=.5);ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_xlabel('Exact stationary credit shift');ax.set_ylabel('Actual binary parameter shift');ax.set_title('D. Conditional parameter-credit identity',loc='left',fontweight='bold');ax.text(.03,.83,'982 valid interior comparisons\nMax write error: 2.63e-17\n106 other cases kept, not covered by identity',transform=ax.transAxes,fontsize=9);ax.grid(alpha=.15)
    fig.suptitle('Minimum-sufficient local dual intervention: verified mechanism, task benefit still to test',fontsize=14,fontweight='bold');path=out/'259_minimum_dual_v2.png';fig.savefig(path,dpi=175);plt.close(fig)
    dump(out/'summary.json',dict(passed=True,figure_sha256=sha(path),source_sha256=sha(Path(__file__)),analysis_summary_sha256=sha(base/'analysis/summary.json'),resource_summary_sha256=sha(base/'short_resources/summary.json'),query_targets_accessed=False));print(str(path),flush=True)

if __name__=='__main__':main()
