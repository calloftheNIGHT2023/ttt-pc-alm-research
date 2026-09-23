"""257 measured query/cost and full paired dual-state ablation figure."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/dual_jump_query';out=base/'figures';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    methods={r['method']:r for r in json.loads((base/'cost_analysis/methods.json').read_text())};paired=json.loads((base/'analysis/paired.json').read_text());timings=json.loads((base/'resources/timings.json').read_text())
    fig,ax=plt.subplots(1,3,figsize=(15,5.8));fig.subplots_adjust(left=.065,right=.98,top=.77,bottom=.27,wspace=.38)
    names=['dual_alm64','alm65','cold__alm16','cold__alm128','probe_all_alm64','adam240','gn20','cold__prior4096_ridge','cold__meta_ridge128','cold__meta_shallow64_20']
    labels=['Dual jump','Ordinary ALM65','Cold ALM16','Cold ALM128','All probes','Adam240','GN20','Prior ridge','Meta ridge128','Shallow20']
    offsets=[(4,10),(-36,-25),(-63,-14),(-11,-38),(-46,-15),(5,2),(-8,10),(-35,9),(6,3),(6,-3)]
    for name,label,offset in zip(names,labels,offsets):
        r=methods[name];color='#c76338' if name=='dual_alm64' else '#16838a' if name=='cold__alm16' else '#65788e'
        ax[0].scatter(r['mean_seconds']*1000,r['mean_mse'],color=color,s=38,zorder=3)
        ax[0].annotate(label,(r['mean_seconds']*1000,r['mean_mse']),xytext=offset,textcoords='offset points',fontsize=8,
            arrowprops=dict(arrowstyle='-',lw=.5,color='#aab1b7') if name in ['cold__alm128','alm65'] else None)
    ax[0].set_xscale('log');ax[0].set(title='A. Complete query-quality / cost',xlabel='Fresh complete call (ms)',ylabel='16-task mean query MSE',xlim=(1.5,1800),ylim=(.035,.102))
    pair=next(r for r in paired if r['primary_readout']=='mode' and r['control']=='dual_reset_alm64');delta=np.array(pair['differences'])
    ax[1].bar(range(16),delta,color=np.where(delta<0,'#16838a','#c76338'));ax[1].axhline(0,c='#58636e',lw=.8)
    ax[1].set(title='B. Keep dual minus reset dual',xlabel='Old task index (5900000 +index)',ylabel='Query MSE difference',xticks=range(0,16,2),ylim=(-.027,.028))
    ax[1].text(.5,.94,'4 better /6 equal /6 worse',transform=ax[1].transAxes,ha='center',fontsize=9)
    colors=['#bbc6cd','#c76338','#6b8199','#16838a','#93bdb4'];categories=['Preparation','Certificate scan','Atomic /optimizer','Geometry','Sampling /read']
    cost=[]
    for name in ['dual_alm64','cold__alm16']:
        rr=[r['metadata'] for r in timings if r['method']==name]
        mean=lambda key:float(np.mean([r.get(key,0.) for r in rr]))*1000
        cost.append([mean('preparation_seconds'),mean('search_seconds'),mean('atomic_seconds')+mean('continuation_collection_seconds')+mean('discovery_seconds'),mean('geometry_seconds'),mean('sampling_seconds')+mean('read_seconds')])
    bottom=np.zeros(2)
    for j,(label,color) in enumerate(zip(categories,colors)):
        value=np.array(cost)[:,j];ax[2].bar([0,1],value,bottom=bottom,color=color,label=label);bottom+=value
    ax[2].set(title='C. Full measured cost components',ylabel='Milliseconds /task',xticks=[0,1],xticklabels=['Dual jump','Cold ALM16'],ylim=(0,870))
    for x,value in enumerate(bottom):ax[2].text(x,value+12,f'{value:.0f} ms',ha='center')
    ax[2].legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,fontsize=8,frameon=False)
    for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.14);a.set_axisbelow(True)
    fig.suptitle('Query evaluation: shared readout helps, but the additional dual jump is not independently ahead',fontsize=15,y=.96)
    fig.text(.5,.855,'1248 new +1120 frozen-reference predictions;528 fresh timings +66 separate memory runs; no retuning',ha='center',fontsize=10,color='#485662')
    fig.text(.5,.025,'Mode readout uses2048 particles,5 MC repeats per task. Point-only heads retain their frozen predictions.\n16 old tasks are not an independent confirmation. Cost includes preparation, full scan, optimization and own geometry; model loading is separate.',ha='center',fontsize=9,color='#485662')
    file=out/'257_query_and_cost.png';fig.savefig(file,dpi=170);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),figure_sha256=sha(file),quality_sha256=sha(base/'analysis/summary.json'),cost_sha256=sha(base/'cost_analysis/summary.json'))
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
