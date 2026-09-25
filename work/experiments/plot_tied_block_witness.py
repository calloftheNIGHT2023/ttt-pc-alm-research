"""Static illustration of the exact repair witness, including simple control."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args();out=args.project/'results/tied_local_block';data=json.loads((out/'constructive_witness.json').read_text())
    q=np.linspace(0,1,1025)
    def predict(b):
        value=q.copy()
        for shift in b:value=np.maximum(0,1-np.abs(2*(value+shift)-1))
        return value
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));ax=axes[0]
    ax.plot(q,predict(data['before_b']),color='#D79434',ls='--',label='Before / blocked scalar repair')
    ax.plot(q,predict(data['tied_b']),color='#177E89',lw=2,label='Teacher = tied repair = simple inverse')
    ax.axhline(.75,color='#748299',ls=':',label='Fitted affine head');ax.scatter(data['x'],data['target'],s=45,c='black',marker='D',label='Only observed targets',zorder=5)
    ax.set(xlabel='Unseen query x',ylabel='Prediction',xlim=(0,1),ylim=(-.02,1.07));ax.legend(fontsize=8,loc='upper right');ax.grid(alpha=.15)
    keys=['before','fitted_affine_head','oracle_best_affine','same_pool_energy','tied'];labels=['Scalar blocked','Fitted affine head','Oracle best affine','Same-pool energy','Tied + exact cuts'];values=[data['exact_uniform_risks'][k]['float'] for k in keys]
    keys.append('simple_inverse');labels.append('Simple internal inverse');values.append(0.)
    y=np.arange(len(keys));axes[1].barh(y,values,color=['#D79434','#748299','#748299','#B76859','#177E89','#177E89']);axes[1].set_xscale('symlog',linthresh=1e-7)
    axes[1].set_yticks(y,labels);axes[1].invert_yaxis();axes[1].set_xlabel('Exact uniform-query squared risk (symlog)');axes[1].set_xlim(0,.5)
    for i,val in enumerate(values):axes[1].text(val*1.25 if val else 1e-8,i,f'{val:.4g}' if val else 'exactly 0',va='center',fontsize=8)
    axes[1].grid(axis='x',alpha=.15);fig.suptitle('Constructive witness: one scalar cannot repair; a tied local block can',fontsize=12)
    fig.text(.5,.015,'The simple parametric inverse also solves this case. This proves a repair-family separation, not unique PC-ALM superiority.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(out/'tied_block_witness.png',dpi=170);plt.close(fig)
    result=dict(source_sha256=sha(Path(__file__)),input_sha256=sha(out/'constructive_witness.json'),figure_sha256=sha(out/'tied_block_witness.png'))
    (out/'witness_figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
