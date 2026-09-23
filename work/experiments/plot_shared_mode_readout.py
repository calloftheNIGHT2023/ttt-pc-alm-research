"""Show marginal mode effect separately from all-task and real-cost results."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/shared_mode_readout';out=base/'analysis/shared_mode_readout.png';assert not out.exists()
    m=json.loads((base/'analysis/methods.json').read_text());summary=json.loads((base/'analysis/summary.json').read_text());rr=json.loads((base/'development/marginal_evaluation.json').read_text())
    fig,ax=plt.subplots(2,2,figsize=(13,8.8));fig.subplots_adjust(top=.86,bottom=.14,hspace=.45,wspace=.32)
    names=['stagnation_switch128','alm16','adam60','gn20','pc128'];short=['Switch128','ALM16','Adam60','GN20','PC128'];xx=np.arange(5)
    ax[0,0].bar(xx-.18,[m[n]['original_point_mse'] for n in names],width=.36,color='#a1acb6',label='Original selected point')
    ax[0,0].bar(xx+.18,[m[n]['mse'] for n in names],width=.36,color='#147e87',label='Same2048-particle readout')
    ax[0,0].set_xticks(xx,short);ax[0,0].set(ylabel='16-task mean query MSE',title='A. Shared readout exposes discovery quality');ax[0,0].legend(fontsize=9)
    for i,r in enumerate(rr):ax[0,1].plot([0,1],[r['prior_mse'],r['full_mse']],'-o',color='#147e87',alpha=.65,markersize=5)
    ax[0,1].set_xticks([0,1],['Other9-method union','Add unique switch mode']);ax[0,1].set(xlim=(-.25,1.25),ylabel='Query MSE, task5900007',title='B. Only the new explanation is added')
    ax[0,1].text(.05,.06,'New mode:36.81% of this discovered union\nFive shared-sample repeats all improve',transform=ax[0,1].transAxes,fontsize=10)
    n2=['meta_ridge128','prior4096_ridge','alm16','alm128','adam60','adam240','gn20','stagnation_switch128'];lab=['Meta ridge','Prior ridge','ALM16','ALM128','Adam60','Adam240','GN20','Switch128']
    offsets=[(5,4),(-45,8),(-5,-16),(5,-2),(-47,6),(5,0),(-33,-14),(5,0)]
    for n,l,o in zip(n2,lab,offsets):
        c='#c56837' if n=='stagnation_switch128' else '#147e87' if n.startswith('alm') else '#6a7d90';x=m[n]['mean_seconds']*1000;y=m[n]['mse']
        ax[1,0].scatter(x,y,color=c,s=45);ax[1,0].annotate(l,(x,y),xytext=o,textcoords='offset points',fontsize=9)
    ax[1,0].set_xscale('log');ax[1,0].set(xlabel='True cold fit + own geometry + read (ms)',ylabel='16-task mean query MSE',title='C. Geometry and collection costs included',xlim=(1.2,800),ylim=(.037,.08))
    vals=[np.mean([r['cross_term'] for r in rr]),np.mean([r['positive_square_term'] for r in rr]),np.mean([r['difference'] for r in rr])]
    bars=ax[1,1].bar([0,1,2],vals,color=['#147e87','#c56837','#36566c']);ax[1,1].axhline(0,color='#666',lw=.8)
    ax[1,1].set_xticks([0,1,2],['Alignment term','Positive square term','Net risk change']);ax[1,1].set(ylabel='Risk contribution',title='D. Marginal-risk identity, not mode counting',ylim=(-.012,.005))
    for b,v in zip(bars,vals):ax[1,1].text(b.get_x()+b.get_width()/2,v+(.0004 if v>=0 else -.0009),f'{v:+.5f}',ha='center',fontsize=10)
    for a in ax.ravel():a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
    fig.suptitle('A newly discovered feasible mode has measurable unseen-query value',fontsize=17,y=.965)
    fig.text(.5,.915,'116 exact-positive-interior regions;960 frozen predictions;180 full cold predictor replays',ha='center',fontsize=11,color='#425360')
    fig.text(.5,.045,'Old development tasks only. ALM16 is a pre-existing secondary control, not a retroactively changed primary.\nFive repeats measure sampling variation, not new tasks. Union pools are charged capacity diagnostics, not deployable winners.',ha='center',fontsize=9,color='#425360')
    fig.savefig(out,dpi=170);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),figure_sha256=sha(out),analysis_sha256=sha(base/'analysis/summary.json'),methods_sha256=sha(base/'analysis/methods.json'))
    dump(base/'analysis/figure_audit.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
