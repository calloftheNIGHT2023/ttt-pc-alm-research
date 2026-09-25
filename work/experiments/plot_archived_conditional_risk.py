import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.input/'protocol.json').read_text());rows=json.loads((a.input/'risks.json').read_text());summary=json.loads((a.input/'summary.json').read_text())
    assert len(rows)==96
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h
    for row in rows:
        assert hashlib.sha256((a.input/row['state_file']).read_bytes()).hexdigest()==row['state_sha256']
    s={r['method']:r for r in summary['summary']};fig,axes=plt.subplots(1,2,figsize=(11,4.3));x=np.arange(3)
    for suffix,shift,color,label in [('original',-.17,'#9aa6b2','Best-point patterns only'),('archive',.17,'#16857f','All visited patterns')]:
        items=[s[f'{f}_{suffix}'] for f in ['alm20','adam60','adam240']]
        axes[0].bar(x+shift,[r['actual_excess'] for r in items],width=.32,color=color,label=label)
        axes[1].bar(x+shift,[r['mass_fraction'] for r in items],width=.32,color=color,label=label)
    for ax in axes:ax.set_xticks(x,['ALM20','Adam60','Adam240']);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Conditional excess risk of frozen 512-sample state');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Discovered posterior mass fraction');axes[1].set_ylim(0,1.05)
    fig.suptitle('Retaining visited modes repairs missing-explanation bias (16 old tasks)');fig.tight_layout();fig.savefig(a.input/'archive_conditional_risk.png',dpi=165);plt.close(fig)
    result=dict(source_hashes=len(protocol['source_sha256']),state_hashes=len(rows),passed=True)
    (a.input/'final_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
