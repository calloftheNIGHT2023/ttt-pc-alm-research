"""278 audited transfer figure and full predefined count cross-tab."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/forward_credit_transfer';out=base/'figures';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    audit=base/'audit';aa=json.loads((audit/'summary.json').read_text());assert aa['passed'];ap0=json.loads((audit/'protocol.json').read_text());assert sha(base/'diagnosis/summary.json')==ap0['diagnosis_summary_sha256']
    hashes=dict(ap0['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    ss=json.loads((base/'diagnosis/summary.json').read_text());assert sha(base/'diagnosis/rows.json')==ss['outputs_sha256']['rows.json']
    rows=json.loads((base/'diagnosis/rows.json').read_text());cross=Counter()
    for row in rows:
        for action,other in [('D','A'),('A','D')]:
            if row[action]['positive_mode'] and not row[other]['positive_mode']:
                category='no_event' if row['event'] is None else 'target_hit' if row[action]['branch_event']['actual_target_hit'] else 'target_miss';cross[action+'_'+category]+=1
    assert sum(v for k,v in cross.items() if k.startswith('D_'))==aa['groups']['positive_mode_only_D']
    assert sum(v for k,v in cross.items() if k.startswith('A_'))==aa['groups']['positive_mode_only_A']
    dump(out/'protocol.json',dict(source_sha256=hashes,audit_summary_sha256=sha(audit/'summary.json'),scope='Support-only2112-state diagnosis; local-event denominators1921; counts are states, not distinct modes'))
    dump(out/'cross_counts.json',dict(cross))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained');labels=['Split-activity\ntarget hit','True-forward\ntarget hit','Residual bound\ncertificate'];x=np.arange(3)
    for action,offset,color in [('D',-.18,'#c7742e'),('A',.18,'#729faa')]:
        values=[aa['counts'][action+'_'+field] for field in ['split_target_hit','actual_target_hit','strict_certificate']]
        bars=axes[0].bar(x+offset,values,width=.36,color=color,label=action);axes[0].bar_label(bars,padding=3)
    axes[0].set_xticks(x,labels);axes[0].set_ylim(0,2150);axes[0].set_title('A. Local target vs. complete forward path');axes[0].set_ylabel('States, out of1921 with a local event');axes[0].legend()
    labels=['D only','Both D and A','A only'];values=[aa['groups']['positive_mode_'+k] for k in ['only_D','both','only_A']]
    bars=axes[1].bar(np.arange(3),values,color=['#c7742e','#b7c9c9','#729faa']);axes[1].bar_label(bars,padding=3);axes[1].set_xticks(np.arange(3),labels);axes[1].set_ylim(0,520)
    axes[1].set_title('B. Branch contains a support-feasible region');axes[1].set_ylabel('States, out of2112; neither =1466')
    fig.suptitle('Local credit transfer: target preservation and useful branch changes differ',fontweight='bold')
    fig.supxlabel('Support-only development diagnosis. Positive branch does not mean the current parameters fit support; no query benefit claimed.',fontsize=9)
    fig.savefig(out/'278_forward_credit_transfer.png',dpi=175);plt.close(fig)
    ans=dict(passed=True,protocol_sha256=sha(out/'protocol.json'),cross_counts=dict(cross),outputs_sha256={n:sha(out/n) for n in ['cross_counts.json','278_forward_credit_transfer.png']})
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
