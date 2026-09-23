"""261 query changes and actual cost, with task-paired and MC boundaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/minimum_dual_query';out=base/'figures_v2';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    costs={r['method']:r for r in json.loads((base/'cost_analysis/methods.json').read_text())};paired=json.loads((base/'analysis/paired.json').read_text());quality={(r['method'],r['readout']):r for r in json.loads((base/'analysis/methods.json').read_text())};timings=json.loads((base/'resources/timings.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(2,2,figsize=(13,8.9),layout='constrained');teal='#16838a';orange='#c76338';gray='#6b8199'
    ax=axes[0,0];names=['minimum_dual_alm64','minimum_activity_alm64','minimum_reset_alm64','dual_alm64','alm65','cold__alm16','cold__alm128','probe_all_alm64'];labels=['Minimum dual','Same activity, no dual','Reset minimum dual','Old grid dual','Ordinary ALM65','Cold ALM16','Cold ALM128','All direct probes']
    for i,(name,label) in enumerate(zip(names,labels)):
        row=quality[name,'mode'];rr=row['replicate_task_mean_mses'];y=row['mean_mse'];ax.errorbar(y,i,xerr=[[y-min(rr)],[max(rr)-y]],fmt='o',color=teal if i==0 else gray,capsize=3)
    ax.set_yticks(range(len(names)),labels);ax.invert_yaxis();ax.set_xlabel('Mean unseen-query MSE (lower is better)');ax.set_title('A. Mean and range over five MC repeats',loc='left',fontweight='bold');ax.grid(axis='x',alpha=.15)
    ax=axes[0,1];pair=next(r for r in paired if r['primary_readout']=='mode' and r['control']=='alm65');delta=np.array(pair['differences']);ax.bar(range(16),delta,color=np.where(delta<0,teal,orange));ax.axhline(0,c=gray,lw=1);ax.set_xticks(range(0,16,2));ax.set_xlabel('Old task index (5900000 + index)');ax.set_ylabel('Query MSE difference');ax.set_title('B. Minimum dual minus ordinary ALM65',loc='left',fontweight='bold');ax.text(.97,.05,f"{pair['lower']} better / {pair['same']} equal / {pair['higher']} worse\nMean: {pair['mean_difference']:+.6f}",transform=ax.transAxes,ha='right',fontsize=9);ax.grid(axis='y',alpha=.15)
    ax=axes[1,0];names=['minimum_dual_alm64','dual_alm64','alm65','cold__alm16','cold__alm128','adam240','cold__prior4096_ridge','cold__meta_ridge128','cold__meta_shallow64_20','probe_all_alm64'];labels=['Minimum dual','Old grid','ALM65','ALM16','ALM128','Adam240','Prior ridge','Meta ridge','Shallow20','All probes'];offsets=[(-70,-37),(38,31),(-60,17),(-45,-19),(5,-20),(5,5),(-35,8),(5,5),(5,-6),(10,-10)]
    for name,label,offset in zip(names,labels,offsets):
        r=costs[name];ax.scatter(r['mean_seconds']*1000,r['mean_mse'],s=40,color=teal if name=='minimum_dual_alm64' else gray);ax.annotate(label,(r['mean_seconds']*1000,r['mean_mse']),xytext=offset,textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='-',color='#AAAAAA',lw=.5))
    ax.set_xscale('log');ax.set_xlim(1.4,1800);ax.set_ylim(.03,.105);ax.set_xlabel('Fresh complete call (ms)');ax.set_ylabel('Mean unseen-query MSE');ax.set_title('C. Full cost: all preprocessing is charged',loc='left',fontweight='bold');ax.grid(alpha=.15)
    ax=axes[1,1];names=['minimum_dual_alm64','dual_alm64','alm65','cold__alm16'];labels=['Minimum\ndual','Old grid\n(short scan)','Ordinary\nALM65','Cold\nALM16'];data=[]
    for name in names:
        rr=[r['metadata'] for r in timings if r['method']==name];mean=lambda k:float(np.mean([r.get(k,0.) for r in rr]))*1000
        data.append([mean('preparation_seconds'),mean('search_seconds')-mean('first_crossing_seconds'),mean('first_crossing_seconds'),mean('atomic_seconds')+mean('continuation_collection_seconds')+mean('discovery_seconds'),mean('geometry_seconds'),mean('sampling_seconds')+mean('read_seconds')])
    bottom=np.zeros(4)
    for j,(label,color) in enumerate(zip(['Preparation','Grid search','Exact threshold','Atomic / optimizer','Geometry','Sample / read'],['#bdc7ce','#c98358','#e5b068','#6b8199','#16838a','#93bdb4'])):
        values=np.array(data)[:,j];ax.bar(range(4),values,bottom=bottom,color=color,label=label);bottom+=values
    ax.set_xticks(range(4),labels);ax.set_ylabel('Milliseconds / task');ax.set_ylim(0,max(bottom)*1.2);ax.set_title('D. Measured components; threshold not free',loc='left',fontweight='bold')
    for i,value in enumerate(bottom):ax.text(i,value+max(bottom)*.025,f'{value:.0f}',ha='center',fontsize=9)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,fontsize=8,frameon=False)
    fig.suptitle('Minimum-sufficient dual: small development-query gain; no independent confirmation yet',fontweight='bold',fontsize=14);path=out/'261_minimum_dual_query_v2.png';fig.savefig(path,dpi=175);plt.close(fig)
    dump(out/'summary.json',dict(passed=True,source_sha256=sha(Path(__file__)),figure_sha256=sha(path),quality_sha256=sha(base/'analysis/summary.json'),cost_sha256=sha(base/'cost_analysis/summary.json')));print(str(path),flush=True)

if __name__=='__main__':main()
