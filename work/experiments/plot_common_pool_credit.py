"""Total coverage and symmetric set differences on exactly the same regions."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/common_pool_credit'
    s=json.loads((base/'development/summary.json').read_text());a=json.loads((base/'audit/summary.json').read_text());assert a['passed']
    ss=s['summaries'];names=['ALM native','Same-ALM residual','Adam native','Adam residual','Ordinary PC','No dual'];y=np.arange(6)
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.4),layout='constrained');old=np.array([r['old_count'] for r in ss]);new=np.array([r['positive'] for r in ss])
    axes[0].barh(y,old,color='#8b9aa9',label='Individual-credit certificates');axes[0].barh(y,new,left=old,color='#148e88',label='New joint-credit certificates')
    for i,v in enumerate(old+new):axes[0].text(v+3,i,str(v),va='center',fontsize=9)
    axes[0].set(yticks=y,yticklabels=names,xlim=(0,380),xlabel='Total exact exclusions (same 700 task-region pairs)',title='Common pool removes different-denominator confounding')
    axes[0].invert_yaxis();axes[0].legend(fontsize=8,loc='lower right')
    pairs=[r for r in a['comparisons'] if r['method_a']=='alm_native' and r['method_b']!='adam60_residual']
    labels={'alm_residual':'Same-ALM residual','adam60_native':'Adam native','pc_native':'Ordinary PC','nodual_native':'No dual'};y=np.arange(len(pairs))
    left=np.array([r['only_a'] for r in pairs]);right=np.array([r['only_b'] for r in pairs])
    axes[1].barh(y,-left,color='#148e88',label='Only ALM native');axes[1].barh(y,right,color='#d59b4e',label='Only comparator')
    for i,(l,r) in enumerate(zip(left,right)):
        axes[1].text(-l-3,i,str(l),ha='right',va='center');axes[1].text(r+3,i,str(r),va='center')
    axes[1].set(yticks=y,yticklabels=[labels[r['method_b']] for r in pairs],xlim=(-145,140),xlabel='Symmetric exclusive counts (each side must be reported)',title='ALM is complementary, not a superset of strong controls')
    axes[1].invert_yaxis();axes[1].axvline(0,color='#5c6771',lw=.8);axes[1].legend(fontsize=8,loc='lower right')
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f"16 tasks; same support / regions / 128-response solver. ALM-only vs non-ALM union: {a['grouped']['alm_only_vs_nonalm']}; vs all other banks: {a['grouped']['alm_only_vs_all_other']}.\nFinite-path evidence, NOT a proof that comparator convex hulls cannot certify those regions.",fontsize=10.5)
    out=base/'analysis';out.mkdir(parents=True,exist_ok=True);path=out/'common_pool_credit.png';assert not path.exists();fig.savefig(path,dpi=175);plt.close(fig)
    result=dict(passed=True,source_sha256=sha(Path(__file__)),main_sha256=sha(base/'development/summary.json'),audit_sha256=sha(base/'audit/summary.json'),figure_sha256=sha(path))
    (out/'figure_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
