"""Layout-only v2: separate long tick labels from explanatory footer."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/cold_stagnation_switch';out=base/'analysis/cold_stagnation_switch_v2.png';assert not out.exists()
    methods=json.loads((base/'analysis/methods.json').read_text());events=json.loads((base/'analysis/trigger_causality.json').read_text())
    fig,ax=plt.subplots(1,3,figsize=(15,5.4));fig.subplots_adjust(top=.77,bottom=.30,wspace=.36)
    ax[0].hist([e['trigger_after'] for e in events],bins=np.arange(0,145,16),color='#16838a',edgecolor='white');ax[0].axvline(102.5,color='#c76338',ls='--')
    ax[0].set(xlabel='Sweep when switch is triggered',ylabel='Restart count',title='A. Late activation: median102.5 /128',xlim=(0,132))
    labels=['Triggered','Branch changed','Support improved','Final tasks (of16)'];values=[26,25,23,2]
    bars=ax[1].bar(np.arange(4),values,color=['#8bb6b8','#5ca1a5','#16838a','#c76338']);ax[1].set_xticks(np.arange(4),labels,rotation=20,ha='right')
    ax[1].set(ylabel='Count (last bar: tasks)',title='B. Changes rarely reach selected output',ylim=(0,31))
    for b,v in zip(bars,values):ax[1].text(b.get_x()+b.get_width()/2,v+.4,str(v),ha='center')
    names=['stagnation_switch128','alm128','nodual128','pc128','adam240','gn20','prior4096_ridge','meta_ridge128','meta_shallow64_20']
    short=['Switch128','ALM128','No dual128','PC128','Adam240','GN20','Prior ridge','Meta ridge128','Shallow20'];offsets=[(5,-15),(5,5),(-75,7),(-17,-15),(-20,7),(-8,-16),(-45,-18),(-8,9),(-10,7)]
    for n,s,o in zip(names,short,offsets):
        r=methods[n];color='#c76338' if n=='stagnation_switch128' else ('#16838a' if 'ridge' in n else '#65788e')
        ax[2].scatter(r['mean_seconds']*1000,r['mean_mse'],c=color,s=45,zorder=3)
        ax[2].annotate(s,(r['mean_seconds']*1000,r['mean_mse']),xytext=o,textcoords='offset points',fontsize=8,
            arrowprops=dict(arrowstyle='-',color='#aab1b7',lw=.6) if n=='nodual128' else None)
    ax[2].set_xscale('log');ax[2].set(xlabel='Untraced fit + query read (ms)',ylabel='16-task mean query MSE',title='C. Strong regressions remain ahead',ylim=(.055,.115),xlim=(1.3,210))
    for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
    fig.suptitle('Cold-start translation: the current trigger is late and output selection discards gains',fontsize=16,y=.96)
    fig.text(.5,.855,'16 unfiltered old tasks ×20 configurations; all predictions saved before query evaluation',ha='center',fontsize=11,color='#485662')
    fig.text(.5,.06,'26 /272 restarts trigger; 14 /16 final predictions remain identical to no-dual. No threshold retuning.\nTiming:960 untraced replays; memory:40 separate traced fits. Plot subset; all20 methods retained in the report.',ha='center',fontsize=9,color='#485662')
    fig.savefig(out,dpi=170);plt.close(fig)
    audit=dict(passed=True,source_sha256=sha(Path(__file__)),figure_sha256=sha(out),analysis_sha256=sha(base/'analysis/summary.json'),methods_sha256=sha(base/'analysis/methods.json'),supersedes='v1 figure layout only; numeric content unchanged')
    dump(base/'analysis/figure_audit_v2.json',audit);print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
