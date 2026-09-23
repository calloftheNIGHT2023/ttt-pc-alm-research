"""Show the quantifier gap: stored trajectory versus a relaxed forward cone."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/forward_credit_envelope'
    a=read(base/'audit/summary.json');assert a['passed'];s=read(base/'development/summary.json');old=read(root/'results/baseline_history/development/summary.json')
    rows=[dict(positive=539,nonpositive=161),next(r for r in old['summaries'] if r['method']=='adam60_history_full')['effective_counts'],*[r['counts'] for r in s['summaries']]]
    labels=['ALM\nreference','Executed Adam\ncomplete history','Ideal forward\nenvelope','Perturbed forward\nenvelope','All modes\nboth signs']
    fig,axes=plt.subplots(1,2,figsize=(13,6.2),gridspec_kw={'width_ratios':[1.25,1]},layout='constrained');bottom=np.zeros(5);x=np.arange(5)
    for status,color,title in [('positive','#148a7c','Exact positive'),('nonpositive','#bdc7d3','Exact nonpositive'),('unknown','#dc9b43','Unknown')]:
        values=np.array([r.get(status,0) for r in rows]);axes[0].bar(x,values,bottom=bottom,color=color,label=title)
        for i,num in enumerate(values):
            if num>=8:axes[0].text(i,bottom[i]+num/2,str(num),ha='center',va='center',color='white' if status=='positive' else '#233448',fontsize=10)
        bottom+=values
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,730),ylabel='Common task-regions (700 total)',title='All regions, not only the eight old cases');axes[0].tick_params(axis='x',labelsize=8)
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3,frameon=False,fontsize=8)
    axes[1].axis('off')
    boxes=[(.5,.84,'Verified: fixed executed Adam history\n8 exact separations remain','#e4f2ef'),(.5,.55,'Verified: larger forward envelope\nall 8 admit exact positive certificates','#e8eef7'),(.5,.22,'Still open: shared-parameter realization\nCan actual joint forward credits do this?','#fff2de')]
    for xx,yy,label,color in boxes:axes[1].text(xx,yy,label,ha='center',va='center',transform=axes[1].transAxes,fontsize=10,bbox=dict(boxstyle='round,pad=.85',facecolor=color,edgecolor='#788795'))
    axes[1].annotate('Expand the credit class',xy=(.5,.64),xytext=(.5,.72),ha='center',fontsize=9,arrowprops=dict(arrowstyle='->'),xycoords='axes fraction')
    axes[1].annotate('Positive relaxation does not imply realization',xy=(.5,.31),xytext=(.5,.41),ha='center',fontsize=8.5,arrowprops=dict(arrowstyle='->',linestyle='--'),xycoords='axes fraction')
    fig.suptitle('Forward-credit capacity: a positive outer relaxation is not a successful BP optimizer\n16 old tasks; exact mathematical diagnostics only, no new query-risk result',fontsize=11)
    out=base/'analysis';out.mkdir(parents=True,exist_ok=True);path=out/'forward_credit_envelope.png';assert not path.exists();fig.savefig(path,dpi=175);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),main_sha256=sha(base/'development/summary.json'),audit_sha256=sha(base/'audit/summary.json'),figure_sha256=sha(path))
    (out/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
