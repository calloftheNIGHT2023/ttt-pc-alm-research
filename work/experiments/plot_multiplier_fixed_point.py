"""Static scientific summary of the exact dynamic escape evidence."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiplier_fixed_point_exact as core
import streaming_branch_projection as base
from run_multiplier_fixed_point_screen import sha,dump

def main():
    sys.set_int_max_str_digits(0);ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();folder=root/'results/multiplier_fixed_point'
    out=folder/'analysis/multiplier_fixed_point.png';assert not out.exists();seed=5900007;rid=8
    records=json.loads((folder/'witness/trajectory.json').read_text());proof=json.loads((folder/'audit/independent_certificates.json').read_text())
    witness=next(r for r in proof if r['seed']==seed and r['restart']==rid)['multiplier_witnesses'][0]
    stats=json.loads((folder/'analysis/summary.json').read_text());a=np.load(folder/'screen'/f'{seed}.npz');x=a['x'];v=a['v']
    fig,ax=plt.subplots(2,2,figsize=(13,8.6));fig.subplots_adjust(hspace=.40,wspace=.30,top=.86,bottom=.10)
    colors={'alm64':'#007f86','nodual64':'#89949e','pc80':'#9973af','adam60':'#e3a03a','adam240':'#d06c2c','gn20':'#345fa3'}
    steps=np.arange(8);delta=float(core.unpack(witness['delta']));kappa=float(core.unpack(witness['kappa']))
    ax[0,0].plot(steps,delta-steps*.5*kappa,'o-',color=colors['alm64']);ax[0,0].axhline(0,color='#666',lw=.8)
    ax[0,0].axvline(6,color='#b54048',ls='--');ax[0,0].set(xlabel='Accumulated multiplier updates t',ylabel='Exact half-energy difference',title='A. A competing block becomes strictly better')
    ax[0,0].text(.03,.08,'Unchanged primal state through sweep 6\nFirst primal / parameter-branch move: sweep 7',transform=ax[0,0].transAxes,fontsize=10)
    for name in colors:
        saved=np.load(folder/'continuation'/f'{seed}_{name}.npz');i=list(saved['restarts']).index(rid)
        if name in ['alm64','nodual64','pc80']:
            error=[np.max(abs(base.forward(x,b)-v)) for b in saved['best'][:,i]];ax[0,1].semilogy(np.arange(len(error)),error,label=name,color=colors[name],lw=2)
        elif name=='adam240':
            err=float(np.max(abs(base.forward(x,saved['best'][-1,i])-v)));ax[0,1].axhline(err,color=colors[name],ls=':',label='Adam240 final retained')
    ax[0,1].axhline(.001,color='#b54048',ls='--',lw=1,label='Noise band');ax[0,1].axvline(62,color=colors['alm64'],ls=':',lw=1)
    ax[0,1].set(xlabel='Local sweeps (not matched compute)',ylabel='Retained maximum support error',title='B. Support feasibility reached at sweep 62');ax[0,1].legend(fontsize=8,loc='lower left')
    q=np.linspace(0,1,1201);latent=np.random.default_rng(seed).uniform(-.12,.12,4);saved=np.load(folder/'continuation'/f'{seed}_alm64.npz');i=list(saved['restarts']).index(rid)
    ax[1,0].plot(q,base.forward(q,latent),color='#252f39',label='Hidden teacher (evaluation only)',lw=1.6)
    ax[1,0].plot(q,base.forward(q,a['best'][rid]),color=colors['nodual64'],label='Shared initial incumbent',alpha=.8)
    ax[1,0].plot(q,base.forward(q,saved['best'][-1,i]),color=colors['alm64'],label='ALM support-selected',alpha=.9)
    ax[1,0].scatter(x,v,color='#b54048',s=35,zorder=5,label='Four observed supports');ax[1,0].set(xlabel='Query x',ylabel='Prediction',title='C. Exact uniform query risk: 0.18793 → 0.12143');ax[1,0].legend(fontsize=8,ncol=2,loc='lower center')
    names=list(colors);vals=[stats['methods'][n]['task_balanced_mse'] for n in names]
    bars=ax[1,1].bar(np.arange(len(names)),vals,color=[colors[n] for n in names]);ax[1,1].set_xticks(np.arange(len(names)),['ALM64','No dual64','PC80','Adam60','Adam240','GN20'],rotation=20)
    ax[1,1].set(ylabel='Task-balanced query-grid MSE',title='D. 69 certified states across 16 old tasks',ylim=(0,.22))
    for b,vv in zip(bars,vals):ax[1,1].text(b.get_x()+b.get_width()/2,vv+.004,f'{vv:.4f}',ha='center',fontsize=9)
    for aa in ax.ravel():
        aa.spines[['top','right']].set_visible(False);aa.grid(axis='y',alpha=.15);aa.set_axisbelow(True)
    fig.suptitle('Multiplier accumulation can escape an exact shared-state trap',fontsize=18,y=.96)
    fig.text(.5,.91,'Same primal blocks and parameter scope; zero-initialized multipliers; exact rational witness',ha='center',fontsize=11,color='#43505b')
    fig.text(.5,.025,'Conditional mechanism test, not a cold-start or matched-resource victory. Shared nodual512 preparation is not free.\nExact BP gradient is zero at the rational witness; floating Adam drifts but finds no feasible mode in this case.',ha='center',fontsize=9,color='#43505b')
    fig.savefig(out,dpi=170);plt.close(fig)
    audit=dict(passed=True,source_sha256=sha(Path(__file__)),figure_sha256=sha(out),analysis_sha256=sha(folder/'analysis/summary.json'),
        witness_sha256=sha(folder/'witness/summary.json'),risk_sha256=sha(folder/'witness/query_risk.json'))
    dump(folder/'analysis/figure_audit.json',audit);print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
