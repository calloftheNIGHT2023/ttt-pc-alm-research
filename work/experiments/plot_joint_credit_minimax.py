"""Full-hull information result, with same-trajectory and cost boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/joint_credit_minimax/development';s=json.loads((inp/'summary.json').read_text());audit=json.loads((root/'results/joint_credit_minimax/audit/summary.json').read_text());assert audit['passed']
    order=['alm_native','alm_residual','adam60_native','adam60_residual','pc_native','nodual_native'];labels=['ALM joint','Same-ALM residual','Adam joint','Adam residual','Ordinary PC','No dual']
    lookup={r['method']:r for r in s['summaries']};rr=[lookup[n] for n in order];y=np.arange(len(order))
    old=np.array([r['old_proofs'] for r in rr]);new=np.array([r['new_proofs'] for r in rr]);total=old+new
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.4),layout='constrained')
    axes[0].barh(y,old,color='#8b99aa',label='Existing single-direction proofs');axes[0].barh(y,new,left=old,color='#2b9e91',label='New exact convex-mixture proofs')
    for i,r in enumerate(rr):axes[0].text(total[i]+6,i,f'{total[i]} / {r["regions"]}',va='center',fontsize=9)
    axes[0].set(yticks=y,yticklabels=labels,xlim=(0,640),xlabel='Certified infeasible regions',title='1,802 additional exact proofs');axes[0].invert_yaxis();axes[0].legend(loc='lower right',fontsize=8)
    values=np.array([r['new_from_old_gate_skip'] for r in rr]);axes[1].barh(y,values,color='#286f9c')
    for i,v in enumerate(values):axes[1].text(v+4,i,str(v),va='center',fontsize=9)
    axes[1].set(yticks=y,yticklabels=labels,xlim=(0,455),xlabel='New proofs where every old direction was nonpositive',title='1,218 exact witnesses of the quantifier gap');axes[1].invert_yaxis()
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Full convex-hull diagnostic, not a deployed algorithm | ALM joint and same-trajectory residual sets are identical',fontsize=10.5)
    out=root/'results/joint_credit_minimax/analysis';out.mkdir(parents=True,exist_ok=True);path=out/'joint_credit_minimax.png';fig.savefig(path,dpi=180);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),summary_sha256=sha(inp/'summary.json'),audit_sha256=sha(root/'results/joint_credit_minimax/audit/summary.json'),figure_sha256=sha(path))
    (out/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
