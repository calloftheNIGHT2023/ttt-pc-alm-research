"""Independent full-region LP audit and source/figure delivery for round193."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import factorized_credit_bank as factor
from audit_light_h2_certificates_v2 import relaxed_lp


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/light_h2_credit/full_bank_ceiling';bench=root/'results/factorized_credit_bank/development'
    out=root/'results/factorized_credit_bank/analysis';out.mkdir(parents=True,exist_ok=True)
    p=json.loads((bench/'protocol.json').read_text());summary=json.loads((bench/'summary.json').read_text())
    assert summary['passed'] and summary['rows']==576 and summary['banks']==96
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert p['parent_protocol_sha256']==sha(inp/'protocol.json')
    records=json.loads((inp/'records.json').read_text());count=0;maxerror=0.;values_count=0
    for record in records:
        path=inp/record['data_file'];assert sha(path)==record['data_sha256']
        with np.load(path) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
        solver=factor.Bank(x,v,bank);_,proofs,_=solver.screen(regs);assert proofs==record['proofs'];values_count+=len(regs)*len(bank)
        for proof in proofs:
            reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,len(x));a=bank[proof['direction']]
            exact=factor.original.exact_optimum(x,v,reg,a);assert exact==proof['exact'] and exact['positive']
            # Use the original full parameter-space region constraints, not
            # the factorized relaxation, for a separate feasibility check.
            import light_h2_credit as light
            _,_,matrix,rhs=light.neighbor.pattern_matrix(x,v,reg)
            lp=linprog(np.zeros(4),A_ub=matrix,b_ub=rhs,bounds=[(-.12,.12)]*4,options={'primal_feasibility_tolerance':1e-9})
            assert lp.status==2,(record['seed'],record['method'],lp.status)
            lp,value=relaxed_lp(x,v,reg,a)
            if 'empty_layer' in exact:assert lp.status==2
            else:
                assert lp.success;error=abs(exact['value']-value);assert error<1e-8
                maxerror=max(maxerror,error)
            count+=1
    assert count==summary['proof_records_per_variant']==732
    assert values_count==summary['total_float_values_bitwise']
    coverage=json.loads((inp/'summary.json').read_text())['summaries'];names=p['methods'];xaxis=np.arange(len(names));width=.36
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.3),layout='constrained')
    labels=['ALM\njoint','ALM\nresidual','Adam\njoint','Adam\nresidual','PC','No dual']
    axes[0].bar(xaxis-width/2,[r['compressed'] for r in coverage],width,label='32-direction compression',color='#89949e')
    axes[0].bar(xaxis+width/2,[r['full'] for r in coverage],width,label='Full existing checkpoints',color='#237f99')
    axes[0].set_ylabel('Certified empty regions (16 tasks)');axes[0].set_title('Information retained, own pools per learner')
    ss=summary['summaries']
    axes[1].bar(xaxis-width/2,[r['mean_seconds']['dense']*1000 for r in ss],width,label='Dense + exact',color='#bc7a41')
    axes[1].bar(xaxis+width/2,[r['mean_seconds']['factorized']*1000 for r in ss],width,label='Layer reuse + exact',color='#348d77')
    axes[1].set_ylabel('Cold component time (ms)');axes[1].set_title('Same full bank, same exact proofs')
    for ax in axes:ax.set_xticks(xaxis,labels);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Development diagnostic: 96 banks, 576 timings; not end-to-end online speedup')
    image=out/'factorized_credit_bank.png';fig.savefig(image,dpi=165);plt.close(fig)
    doc=root/'outputs/ttt-pc-alm-research/193_factorized_credit_bank_results.md';links=0
    for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        if '://' in target or target.startswith('#'):continue
        assert (doc.parent/target).resolve().exists(),target;links+=1
    result=dict(passed=True,sources=len(p['source_sha256']),banks=len(records),full_certificate_replays=count,
        independent_full_region_lp=count,independent_relaxed_lp=count,maximum_relaxed_lp_error=maxerror,
        float_values_bitwise_from_frozen_benchmark=values_count,links=links,figure_sha256=sha(image),
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(bench/'protocol.json'),
        scope='component identities and independent proof checks; no new online efficacy claim')
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
