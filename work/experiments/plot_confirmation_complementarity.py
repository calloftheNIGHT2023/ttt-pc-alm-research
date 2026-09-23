"""Presentation of the audited union accounting, explicitly not a method result."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/confirmation_conditional_risk';inp=base/'complementarity';out=base/'complementarity_figures';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    audit=base/'complementarity_audit';s=json.loads((audit/'summary.json').read_text());assert s['passed'];p=json.loads((audit/'protocol.json').read_text());assert sha(inp/'summary.json')==p['input_summary_sha256']
    hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    summary=json.loads((inp/'summary.json').read_text())
    for n,h in summary['outputs_sha256'].items():assert sha(inp/n)==h
    selected=[('minimum_activity_alm64','Same activity'),('minimum_reset_alm64','Reset dual'),('alm65','Warm ALM65'),('cold__alm16','Cold ALM16'),('cold__alm128','Cold ALM128'),('adam960','Warm Adam960'),('cold__gn40_r33','Cold GN33'),('cold__adam240_r33','Cold Adam33')]
    dump(out/'protocol.json',dict(source_sha256=hashes,audit_sha256=sha(audit/'summary.json'),selected=selected,scope='Descriptive diagnostic only; union acquisition is not free and no joint runtime was measured'))
    methods=json.loads((inp/'methods.json').read_text());paired=json.loads((inp/'paired.json').read_text());mm={(r['control'],r['grid']):r for r in methods};cc={(r['control'],r['grid'],r['metric']):r for r in paired}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13.8,6.5),layout='constrained');yy=np.arange(len(selected));labels=[l for _,l in selected]
    for shift,field,color,label in [(-.2,'addition','#257f81','Add proposed-only explanations'),(.2,'discarding','#ca772b','Do not retain control-only explanations')]:
        axes[0].barh(yy+shift,[mm[n,257][field] for n,_ in selected],height=.34,color=color,label=label)
    axes[0].scatter([mm[n,257]['difference'] for n,_ in selected],yy,color='#163d53',marker='D',s=24,label='Net proposed minus control',zorder=3)
    axes[0].set_yticks(yy,labels);axes[0].invert_yaxis();axes[0].axvline(0,color='#8a959c',linewidth=.8)
    axes[0].set_xlabel('Mean ideal-policy excess-risk difference');axes[0].set_title('A. Net difference = addition + discarding',loc='left',fontweight='bold')
    axes[0].legend(loc='lower left',fontsize=8,bbox_to_anchor=(0,-.28));axes[0].grid(axis='x',alpha=.2)
    for i,(name,_) in enumerate(selected):
        r=cc[name,257,'addition'];mean=r['mean_difference'];lo,hi=r['descriptive_ci95']
        axes[1].errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',color='#257f81',capsize=3)
    axes[1].set_yticks(yy,labels);axes[1].invert_yaxis();axes[1].axvline(0,color='#8a959c',linewidth=.8);axes[1].grid(axis='x',alpha=.2)
    axes[1].set_xlabel('Union minus control (negative = complementary value)');axes[1].set_title('B. Addition: descriptive paired 95% CIs',loc='left',fontweight='bold')
    fig.suptitle('Useful complementary explanations can coexist with a worse standalone pool',fontweight='bold',fontsize=14)
    fig.supxlabel('64 old tasks; observation-only posterior diagnostic. The union is NOT a matched-cost online method.',fontsize=10)
    fig.savefig(out/'270_complementarity.png',dpi=175);plt.close(fig)
    lines=['# All35 internal optimizers: complementary-mode accounting','',
        'Fixed64 tasks. A=E(union)-E(control); L=E(primary)-E(union); D=A+L. These are ideal policies, not new online predictors. All intervals are unadjusted descriptive task-bootstrap intervals.','']
    for grid in [257,129]:
        lines.extend([f'## {grid}-point arithmetic grid','',
            '| Control | Control risk | Union risk | A: addition | L: discarding | D: net |','|---|---:|---:|---:|---:|---:|'])
        for r in methods:
            if r['grid']==grid:lines.append('| '+r['control']+' | '+' | '.join(f'{r[f]:.10g}' for f in ['control_excess','union_excess','addition','discarding','difference'])+' |')
        lines.extend(['','| Control | Term | Mean | CI lower | CI upper | Lower/equal/higher tasks |','|---|---|---:|---:|---:|---|'])
        for r in paired:
            if r['grid']==grid:
                lo,hi=r['descriptive_ci95'];lines.append(f"| {r['control']} | {r['metric']} | {r['mean_difference']:.10g} | {lo:.10g} | {hi:.10g} | {r['lower_tasks']}/{r['equal_tasks']}/{r['higher_tasks']} |")
        lines.append('')
    (out/'all_methods.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    ans=dict(passed=True,method_grid_tables=70,paired_intervals=210,protocol_sha256=sha(out/'protocol.json'),outputs_sha256={n:sha(out/n) for n in ['270_complementarity.png','all_methods.md']})
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
