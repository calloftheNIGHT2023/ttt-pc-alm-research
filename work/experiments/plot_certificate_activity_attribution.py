"""Audited development result figure and complete65-method tables."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/certificate_activity_attribution';inp=base/'evaluation';audit=base/'evaluation_audit';out=base/'figures';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    aa=json.loads((audit/'summary.json').read_text());assert aa['passed'];ap0=json.loads((audit/'protocol.json').read_text());assert sha(inp/'summary.json')==ap0['evaluation_summary_sha256']
    hashes=dict(ap0['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    p=json.loads((inp/'protocol.json').read_text());ss=json.loads((inp/'summary.json').read_text())
    for n,h in ss['outputs_sha256'].items():assert sha(inp/n)==h
    selected=[('anchor_A33_32_A64','Post-result hypothesis: certified A33'),('credit_control_bias33','Bias-only atom + ALM'),('credit_control_zero_activity33','Same site, zero-dual activity + ALM'),('credit_control_alm33','Plain ALM atom + ALM'),('credit_control_pc33','Plain PC atom + PC'),('credit_control_nodual33','No-dual atom + no-dual'),('credit_control_probe33','All zero-dual branch trials + ALM'),('distinct_D32_A64','D32 plus A anchor'),('cold__alm256','Cold ALM256, 17 starts'),('adam960','Warm Adam960'),('cold__adam240_r33','Cold Adam240, 33 starts'),('cold__prior16384_ridge','Prior ridge, 16384'),('cold__meta_shallow64_20','Meta-shallow, 20 steps')]
    dump(out/'protocol.json',dict(source_sha256=hashes,evaluation_audit_sha256=sha(audit/'summary.json'),selected=selected,scope='Development only; same1088 restart-sweeps is not a wall-clock or memory match'))
    methods=json.loads((inp/'methods.json').read_text());paired=json.loads((inp/'paired.json').read_text());mm={(r['method'],r['grid']):r for r in methods}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,7.5),layout='constrained');yy=np.arange(len(selected));labels=[l for _,l in selected];colors=['#c7742e']+['#86abad']*6+['#c9dadd']*(len(selected)-7)
    for ax,metric,title in zip(axes,['mse','actual_excess'],['A. Observed query MSE','B. Conditional excess risk above Bayes']):
        ax.barh(yy,[mm[n,257][metric] for n,_ in selected],color=colors,height=.67)
        ax.set_yticks(yy,labels if ax is axes[0] else ['']*len(labels));ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Lower is better')
    fig.suptitle('Certificate-selected activity: six full33-start controls, all methods retained',fontweight='bold',fontsize=14)
    fig.supxlabel('64 previously evaluated tasks, not a new blind confirmation. New traced timings are not compared with old untraced timings.',fontsize=9)
    fig.savefig(out/'282_certificate_activity_development.png',dpi=175);plt.close(fig)
    lines=['# All65 methods and all512 comparisons','',p['scope'],'']
    for grid in [257,129]:
        lines.extend([f'## {grid}-point grid','',
            '| Method | Query MSE | Point MSE | Actual excess | Ideal policy excess | Expected MC | Empty pools |','|---|---:|---:|---:|---:|---:|---:|'])
        for name in p['methods']:
            r=mm[name,grid];lines.append('| '+name+' | '+' | '.join(f'{r[f]:.10g}' for f in ['mse','point_mse','actual_excess','policy_excess','expected_mc'])+f" | {r['empty_pool_tasks']} |")
        lines.extend(['','| Control | Metric | Primary minus control | CI lower | CI upper | Lower/equal/higher tasks |','|---|---|---:|---:|---:|---|'])
        for r in paired:
            if r['grid']==grid:
                lo,hi=r['descriptive_ci95'];lines.append(f"| {r['control']} | {r['metric']} | {r['mean_difference']:.10g} | {lo:.10g} | {hi:.10g} | {r['lower_tasks']}/{r['equal_tasks']}/{r['higher_tasks']} |")
        lines.append('')
    (out/'all_methods.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    ans=dict(passed=True,method_grid_tables=130,paired_intervals=512,protocol_sha256=sha(out/'protocol.json'),outputs_sha256={n:sha(out/n) for n in ['282_certificate_activity_development.png','all_methods.md']})
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
