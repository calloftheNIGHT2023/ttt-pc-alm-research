"""Complete seven-learner pool comparison and predeclared strong contrasts."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();out=root/'results/band_conditioned_online/analysis'
    s=read(out/'summary.json');assert s['passed'] and s['failed_episodes']==0;r={r['method']:r for r in s['summaries']};paired={r['right']:r for r in read(out/'primary_contrasts.json')}
    learners=['alm','adam60','adam240','gn20','pc','nodual','direct64'];labels=['ALM16','Adam60','Adam240','GN20','PC80','No dual16','Direct64'];x=np.arange(7)
    fig,axes=plt.subplots(2,2,figsize=(14,9.5),layout='constrained')
    pools=[('prior256','#a5b0c0'),('prior1280','#428ab5'),('band_inverse','#158674')]
    for (key,title,ylabel),ax in zip([('mse','Full-stream unseen-query risk','Mean across four prefixes'),('first_mse','First-write unseen-query risk','n = 4 observed pairs'),('time','Full-stream computation','New measured seconds, including initialization')],[axes[0,0],axes[0,1],axes[1,0]]):
        for j,(pool,color) in enumerate(pools):
            values=[r[f'{pool}_{learner}_c5'][key] for learner in learners];ax.bar(x+(j-1)*.24,values,width=.23,color=color,label=pool)
        ax.set(xticks=x,xticklabels=labels,ylabel=ylabel,title=title);ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0,0].legend(frameon=False,fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.1))
    names=['band_inverse_adam60_c5','band_inverse_adam240_c5','band_inverse_gn20_c5','band_inverse_pc_c5','band_inverse_nodual_c5','band_inverse_direct64_c5','prior256_alm_native_gate','prior1280_alm_native_gate']
    titlelabels=['Same starts: Adam60','Same starts: Adam240','Same starts: GN20','Same starts: PC80','Same starts: no dual','Same starts: direct','ALM + prior256','ALM + prior1280']
    ax=axes[1,1];y=np.arange(len(names));mean=np.array([paired[n]['mse']['mean'] for n in names]);bounds=np.array([paired[n]['mse']['ci95'] for n in names])
    ax.errorbar(mean,y,xerr=np.vstack([np.maximum(0,mean-bounds[:,0]),np.maximum(0,bounds[:,1]-mean)]),fmt='o',color='#158674',capsize=3)
    ax.axvline(0,color='#333c48',lw=1,ls='--');ax.set(yticks=y,yticklabels=titlelabels,xlabel='Primary risk minus comparator (negative favors primary)',title='Paired descriptive 95% intervals: 16 old tasks');ax.invert_yaxis();ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Observation-conditioned starts: separate initialization benefit from optimizer benefit\n39 configurations, 1,248 streams; two repetitions averaged within each task; no new confirmation',fontsize=11)
    target=out/'band_conditioned_online.png';assert not target.exists();fig.savefig(target,dpi=175);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),analysis_sha256=sha(out/'summary.json'),contrast_sha256=sha(out/'primary_contrasts.json'),figure_sha256=sha(target))
    (out/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
