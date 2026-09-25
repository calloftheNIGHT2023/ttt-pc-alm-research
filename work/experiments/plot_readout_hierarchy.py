"""Static scientific figures for the sampling bottleneck and exact witness."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args()
    group=args.project/'results/readout_hierarchy/diagnostic';witness=args.project/'results/conditional_fiber/primitive'
    data=json.loads((group/'summary.json').read_text());rows={r['method']:r for r in data['summary']}
    names=['alm_retained_full','alm_tied_forward_full','alm_bp_tied_full','adam16_tied_full','adam60_h2_full','direct4096_h2_full']
    labels=['Original ALM','ALM tied forward','Same-path BP credit','Adam16 tied','Adam60-H2','Direct4096-H2']
    within=np.array([rows[n]['within'] for n in names]);between=np.array([rows[n]['between'] for n in names])
    fig,ax=plt.subplots(figsize=(10,4.5));y=np.arange(len(names))
    ax.barh(y,within,color='#188487',label='Within-region parameter sampling')
    ax.barh(y,between,left=within,color='#dfaa48',label='Random region labels')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Expected 512-particle contribution to first-write excess risk')
    ax.set_title('Both kinds of sampling noise remain after region discovery')
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.legend(loc='lower center',bbox_to_anchor=(.5,-.32),ncol=2)
    fig.tight_layout();fig.savefig(group/'readout_hierarchy.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));a=1/16;e=1/64
    polygon=np.array([[-a,2*a],[a,-2*a],[a,e-2*a],[-a,e+2*a],[-a,2*a]])
    axes[0].fill(polygon[:,0],polygon[:,1],color='#188487',alpha=.3,label='Uniform parameter cell')
    u=np.linspace(-a,a,301);z=e/2
    axes[0].plot(u,z-2*u,color='#188487',lw=2.5,label='Joint fiber: direction (1,-2)')
    axes[0].plot([-e/4,e/4],[z,z],color='#cc5144',lw=3,label='Coordinate fiber')
    axes[0].scatter([0],[z],c='black',s=20,zorder=5);axes[0].set(xlabel='First bias',ylabel='Second bias',title='Narrow coordinates, long joint direction')
    axes[0].legend(fontsize=8,loc='upper right');axes[0].grid(alpha=.15)
    prediction=1-2*np.abs(z-4*u);mean=.75-4*z*z
    axes[1].plot(u,prediction,color='#188487',label='Query prediction along joint fiber')
    axes[1].axhline(mean,color='#cc5144',ls='--',label='Exact conditional average')
    axes[1].set(xlabel='First bias along the same fiber',ylabel='Prediction at unseen input q=3/4',title='Integrate uncertainty; keep the target')
    axes[1].legend(fontsize=8,loc='lower center');axes[1].grid(alpha=.15)
    eps=2.**np.linspace(-10,-4,101)
    original=1/48+2*eps**2/3-16*eps**4/9
    coordinate_lower=original-eps**2/3;joint=64*eps**4/45
    axes[2].loglog(eps,original,color='black',label='Original particle variance')
    axes[2].loglog(eps,coordinate_lower,color='#cc5144',ls='--',label='Coordinate: remaining variance lower bound')
    axes[2].loglog(eps,joint,color='#188487',label='Joint: exact remaining variance')
    axes[2].axvline(e,color='gray',ls=':',alpha=.7)
    axes[2].set(xlabel='Width parameter e (fixed example: 1/64)',ylabel='Single-particle estimator variance',title='A strict joint-versus-coordinate gap')
    axes[2].legend(fontsize=8,loc='lower right');axes[2].grid(alpha=.15)
    fig.suptitle('Constructive integration example — geometric control, not PC-specific or a task-risk win',fontsize=13)
    fig.tight_layout();fig.savefig(witness/'conditional_fiber_witness.png',dpi=160,bbox_inches='tight');plt.close(fig)
    audit=dict(source_sha256=sha(Path(__file__)),hierarchy_summary_sha256=sha(group/'summary.json'),
               witness_result_sha256=sha(witness/'result.json'),figures={
                   'results/readout_hierarchy/diagnostic/readout_hierarchy.png':sha(group/'readout_hierarchy.png'),
                   'results/conditional_fiber/primitive/conditional_fiber_witness.png':sha(witness/'conditional_fiber_witness.png')})
    (group/'figure_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
