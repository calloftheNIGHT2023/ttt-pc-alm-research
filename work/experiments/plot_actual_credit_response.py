"""Audited279 coverage comparison; no hidden-query or online-speed claim."""
import argparse,json
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import actual_credit_response as exact
from run_multiplier_fixed_point_screen import sha,dump


def root_float(record):
    r=exact.decode_root(record)
    if r.value is not None:return float(r.value)
    for _ in range(100):r.refine()
    return float((r.lo+r.hi)/2)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/actual_credit_response';inp=base/'primitive';audit=base/'primitive_audit';out=base/'figures';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    aa=json.loads((audit/'summary.json').read_text());assert aa['passed'];ap0=json.loads((audit/'protocol.json').read_text());assert sha(inp/'summary.json')==ap0['primitive_summary_sha256']
    hashes=dict(ap0['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    rows=json.loads((inp/'rows.json').read_text());counts=[]
    for seed in sorted({r['seed'] for r in rows}):
        rr=[r for r in rows if r['seed']==seed];sets={key:set().union(*(set(r[key]) for r in rr)) for key in ['endpoint_positive','response_positive','chord_positive']}
        counts.append(dict(seed=seed,**{key:len(value) for key,value in sets.items()},response_only=len(sets['response_positive']-sets['chord_positive']),chord_only=len(sets['chord_positive']-sets['response_positive'])))
    # First counterexample to response-only necessity, disclosed as such;
    # not selected by any hidden-query or posterior-risk value.
    witness=next(r for r in rows if r['chord_only_positive']);assert sha(inp/witness['file'])==witness['sha256'];data=json.loads((inp/witness['file']).read_text())
    dump(out/'protocol.json',dict(source_sha256=hashes,audit_summary_sha256=sha(audit/'summary.json'),illustration=dict(seed=witness['seed'],restart=witness['restart'],selection='first chord-only positive-state counterexample, no query information')))
    dump(out/'task_coverage.json',counts)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained');x=np.arange(4)
    for offset,field,color,label in [(-.25,'endpoint_positive','#b7c9c9','Endpoints'),(0,'response_positive','#c7742e','Credit response'),(.25,'chord_positive','#557c9c','Simple chord')]:
        bars=axes[0].bar(x+offset,[r[field] for r in counts],width=.24,color=color,label=label);axes[0].bar_label(bars,padding=2)
    axes[0].set_xticks(x,[str(r['seed']) for r in counts]);axes[0].set_ylabel('Distinct positive modes across33 starts');axes[0].set_ylim(0,11);axes[0].legend();axes[0].set_title('A. Equal task-level coverage')
    ends=np.array([[float(F(v)) for v in b] for b in data['mathematical_endpoints']]);dims=np.where(np.any(np.array(data['dual'])!=0,axis=1))[0];assert len(dims)==2
    first=True
    for segment in data['response_segments']:
        lo,hi=root_float(segment['lo']),root_float(segment['hi']);b=np.array([[float(F(v)) for v in pair] for pair in segment['bias']]);points=b[:,0,None]+b[:,1,None]*np.array([lo,hi])
        axes[1].plot(points[dims[0]],points[dims[1]],color='#c7742e',linewidth=2.4,label='Actual credit response' if first else None);first=False
    axes[1].plot(ends[:,dims[0]],ends[:,dims[1]],'--',color='#557c9c',linewidth=2,label='Simple parameter chord')
    axes[1].scatter(ends[:,dims[0]],ends[:,dims[1]],c='#243746',s=42,zorder=5)
    for name,point in zip(['A: t=0','D: t=1'],ends):axes[1].annotate(name,(point[dims[0]],point[dims[1]]),xytext=(5,7),textcoords='offset points')
    for segment in data['chord_segments']:
        if segment['mode'] in witness['chord_only_positive']:
            t=float(F(segment['representative']));point=(1-t)*ends[0]+t*ends[1];axes[1].scatter(point[dims[0]],point[dims[1]],c='#31966a',s=65,marker='D',label='Chord-only feasible mode',zorder=6)
    axes[1].set_xlabel(f'bias[{dims[0]}]');axes[1].set_ylabel(f'bias[{dims[1]}]');axes[1].legend(fontsize=9);axes[1].set_title(f'B. Counterexample: {witness["seed"]}, start{witness["restart"]}')
    fig.suptitle('Exact credit-response paths:132 fixed states, strong interpolation control',fontweight='bold')
    fig.supxlabel('Certified open segments plus common endpoints; isolated breakpoints not claimed complete. No query-risk or wall-time advantage tested.',fontsize=9)
    fig.savefig(out/'280_actual_credit_response.png',dpi=175);plt.close(fig)
    ans=dict(passed=True,protocol_sha256=sha(out/'protocol.json'),task_coverage=counts,outputs_sha256={n:sha(out/n) for n in ['task_coverage.json','280_actual_credit_response.png']})
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
