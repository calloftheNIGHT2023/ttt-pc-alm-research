"""Scientific figures for253/255; frozen measured data, no illustrative data."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cold_stagnation_switch as cold
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/local_dual_jump';out=base/'figures';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    screen=json.loads((base/'screen_v2/summary.json').read_text());atomic=json.loads((base/'transfer/summary.json').read_text());continuation=json.loads((base/'continuation/summary.json').read_text())
    proofs=json.loads((base/'credit_proof/witnesses.json').read_text());w=proofs[0];rows=json.loads((base/'transfer/support_rows.json').read_text());lookup={(r['seed'],r['restart'],r['method']):r for r in rows}
    plt.rcParams.update({'font.size':10,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(13,8));fig.subplots_adjust(top=.85,bottom=.11,hspace=.48,wspace=.30)
    labels=list(screen['selected_tau_counts']);values=list(screen['selected_tau_counts'].values());axs[0,0].bar(labels,values,color='#16838a')
    axs[0,0].set(title='A. Certified at step16: 260 /272 states',xlabel='Selected multiplier jump scale',ylabel='State count')
    methods=['reorder_no_dual','activity_only','dual_jump','alm1','pc1','branch_probe'];labels=['No-dual\nreorder','Same new\nactivity','Dual\njump','ALM1','PC1','Branch\nprobe']
    values=[atomic['methods'][m]['forward_branch_changed'] for m in methods];bars=axs[0,1].bar(labels,values,color=['#9aaab7','#cb9b64','#16838a','#9aaab7','#9aaab7','#617b97'])
    for b,v in zip(bars,values):axs[0,1].text(b.get_x()+b.get_width()/2,v+3,str(v),ha='center')
    axs[0,1].set(title='B. Actual selected-parameter branch changes',ylabel='States /272',ylim=(0,240))
    names=['unchanged','branch_probe','activity_only','dual_jump'];labels=['Old best','Best finite\nbranch probe','Same activity\nwithout dual','Dual jump']
    values=[lookup[w['seed'],w['restart'],m]['best_error'] for m in names];bars=axs[1,0].bar(labels,values,color=['#9aaab7','#617b97','#cb9b64','#16838a'])
    for b,v in zip(bars,values):axs[1,0].text(b.get_x()+b.get_width()/2,v+.001,f'{v:.5f}',ha='center',fontsize=9)
    axs[1,0].set(title=f'C. Support-selected exact witness: {w["seed"]}, r{w["restart"]}',ylabel='Maximum observed-support error',ylim=(0,max(values)*1.2))
    categories=['Dual lower','Same','Probe lower'];values=[6,13,253];bars=axs[1,1].barh(categories,values,color=['#16838a','#c4cbd0','#617b97'])
    for b,v in zip(bars,values):axs[1,1].text(v+3,b.get_y()+b.get_height()/2,str(v),va='center')
    axs[1,1].set(title='D. All paired support errors, not only witnesses',xlabel='States /272',xlim=(0,290));axs[1,1].invert_yaxis()
    fig.suptitle('Local branch switching reaches real parameters; dual credit changes their location',fontsize=15,y=.97)
    fig.text(.5,.91,'Exact activity certificates +9 atomic controls; all16 old tasks and17 starts retained',ha='center',color='#485662')
    fig.text(.5,.025,'No query targets read. Probe coverage contains all dual feasible modes. Six strict finite-probe support witnesses do not imply overall superiority.',ha='center',fontsize=9)
    file=out/'253_atomic_transfer.png';fig.savefig(file,dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,6));fig.subplots_adjust(top=.80,bottom=.18,wspace=.42,left=.13)
    names=list(continuation['methods']);labels=['Dual +ALM64','Activity +ALM64','Reorder +ALM64','Probe +ALM64','Dual reset +ALM64','ALM65','No dual65','PC65','Adam60','Adam240','GN20','GN40','All probes +ALM64']
    values=[continuation['methods'][m]['feasible_tasks'] for m in names];colors=['#16838a' if m=='dual_alm64' else '#617b97' for m in names]
    bars=axs[0].barh(labels,values,color=colors)
    for b,v in zip(bars,values):axs[0].text(v+.2,b.get_y()+b.get_height()/2,str(v),va='center',fontsize=9)
    axs[0].invert_yaxis();axs[0].set(title='A. Support-feasible tasks',xlabel='Tasks /16',xlim=(0,17))
    rows=json.loads((base/'continuation/support_rows.json').read_text());lookup={(r['seed'],r['method']):r for r in rows}
    for ax,seed in zip(axs[1:],[5900000,5900003]):
        for name,label,color,style in [('dual_alm64','Keep jump dual','#16838a','-'),('dual_reset_alm64','Reset jump dual','#c76338','--'),('activity_alm64','Activity only','#617b97',':'),('alm65','Ordinary ALM','#888888','-.')]:
            with np.load(base/'continuation'/lookup[seed,name]['file']) as a:x=a['x'];v=a['v'];best=a['best']
            curve=[]
            for bank in best:
                _,bb=cold.select(bank,x,v);curve.append(cold.base.score(bb[None],x,v,np.zeros(4))[0][0])
            ax.plot(range(len(curve)),curve,label=label,color=color,ls=style,lw=1.8)
        ax.axhline(.001001,c='black',ls='--',lw=.8);ax.set_yscale('log');ax.set(title=f'{"B" if seed==5900000 else "C"}. Task {seed}',xlabel='Continuation sweeps',ylabel='Selected support max error')
    axs[2].legend(fontsize=8,loc='upper right')
    fig.suptitle('Keeping multiplier state helps two tasks; strong BP remains stronger in support fitting',fontsize=15,y=.96)
    fig.text(.5,.875,'208 fixed continuations; same nodual16 preparation; candidate64 sweeps, ordinary controls65',ha='center',color='#485662')
    fig.text(.5,.045,'8 /16 vs6 /16 is an ablation result, not query generalization. Adam240 /GN reach15 /16. All-probe continuation uses629 starts/task, not17.',ha='center',fontsize=9)
    file=out/'255_continuation.png';fig.savefig(file,dpi=170);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),figures={p.name:sha(p) for p in out.glob('*.png')},inputs={str(p.relative_to(base)):sha(p) for p in [base/'screen_v2/summary.json',base/'transfer/summary.json',base/'continuation/summary.json',base/'credit_proof/witnesses.json']})
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
