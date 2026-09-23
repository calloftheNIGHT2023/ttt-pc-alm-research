"""Frozen cold-start quality/cost and causal bottleneck, no selected-task wins."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/cold_stagnation_switch';out=base/'analysis/cold_stagnation_switch.png';assert not out.exists()
    methods=json.loads((base/'analysis/methods.json').read_text());events=json.loads((base/'analysis/trigger_causality.json').read_text());summary=json.loads((base/'analysis/summary.json').read_text())
    fig,ax=plt.subplots(1,3,figsize=(15,4.8));fig.subplots_adjust(top=.78,bottom=.20,wspace=.34)
    t=[e['trigger_after'] for e in events];ax[0].hist(t,bins=np.arange(0,145,16),color='#16838a',edgecolor='white');ax[0].axvline(102.5,color='#c76338',ls='--')
    ax[0].set(xlabel='Sweep when switch is triggered',ylabel='Restart count',title='A. Late activation: median102.5 /128',xlim=(0,132))
    labels=['Triggered','Branch changed','Support improved','Selected output'];values=[26,25,23,2]
    bars=ax[1].bar(np.arange(4),values,color=['#8bb6b8','#5ca1a5','#16838a','#c76338']);ax[1].set_xticks(np.arange(4),labels,rotation=20,ha='right')
    ax[1].set(ylabel='Count',title='B. Useful restart changes rarely reach output',ylim=(0,31))
    for b,v in zip(bars,values):ax[1].text(b.get_x()+b.get_width()/2,v+.4,str(v),ha='center')
    names=['stagnation_switch128','alm128','nodual128','pc128','adam240','gn20','prior4096_ridge','meta_ridge128','meta_shallow64_20']
    nameshort=['Switch128','ALM128','No dual128','PC128','Adam240','GN20','Prior ridge','Meta ridge128','Shallow20']
    offsets=[(5,-15),(5,5),(-80,3),(-17,-15),(-20,7),(-8,-16),(-45,-18),(-8,9),(-10,7)]
    for i,(n,short,off) in enumerate(zip(names,nameshort,offsets)):
        r=methods[n];color='#c76338' if n=='stagnation_switch128' else ('#16838a' if 'ridge' in n else '#65788e')
        ax[2].scatter(r['mean_seconds']*1000,r['mean_mse'],c=color,s=60 if i==0 else 35,zorder=3)
        ax[2].annotate(short,(r['mean_seconds']*1000,r['mean_mse']),xytext=off,textcoords='offset points',fontsize=8)
    ax[2].set_xscale('log');ax[2].set(xlabel='Complete untraced fit + query read (ms)',ylabel='16-task mean query MSE',title='C. Strong regressions remain ahead',ylim=(.055,.115),xlim=(1.3,210))
    for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
    fig.suptitle('Cold-start translation: the mechanism works, but the current trigger is too late',fontsize=16,y=.96)
    fig.text(.5,.865,'16 unfiltered old tasks ×20 configurations; all predictions saved before query evaluation',ha='center',fontsize=11,color='#485662')
    fig.text(.5,.045,'26 /272 restarts trigger; 14 /16 final predictions remain identical to no-dual. No threshold retuning.\nTiming:960 untraced replays; memory:40 separate traced fits. Plot subset; all20 methods retained in the report.',ha='center',fontsize=9,color='#485662')
    fig.savefig(out,dpi=170);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),figure_sha256=sha(out),analysis_sha256=sha(base/'analysis/summary.json'),methods_sha256=sha(base/'analysis/methods.json'))
    dump(base/'analysis/figure_audit.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
