"""Every causal-wave skip is replayed by independent exact integer sums."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import audit_primal_gate_integer as exact
import bounded_error_primal_gate as candidate


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/online_primal_gate/development_v2';p=json.loads((inp/'protocol.json').read_text())
    assert json.loads((inp/'run_audit.json').read_text())['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=json.loads((inp/'rows.json').read_text());lookup={(r['seed'],r['method'],r['repetition'],r['n']):r for r in rows}
    checks=dict(banks=0,causal_waves=0,skipped_occurrences=0,unique_skipped_regions=0,exact_direction_checks=0,
        strict_upper_enclosures=0,duplicate_repetition_waves=0,preserved_proofs=0)
    details=[]
    for cfg in [c for c in p['configs'] if c.get('primal_upper_gate')]:
        for seed in p['seeds']:
            row=lookup[seed,cfg['name'],0,4];path=inp/row['detail_file'];assert sha(path)==row['detail_sha256'];meta=json.loads(path.read_text())
            assert sha(inp/row['state_file'])==row['state_sha256']
            with np.load(inp/row['state_file']) as z:x=z['x'];v=z['v']
            bank=np.array(meta['credit_bank']);ae=exact.exponent(bank);ai=exact.integers(bank,ae);solver=candidate.Bank(x,v,bank)
            checked={};start=dict(checks);proofs={pp['pattern'] for pp in meta['credit_proofs']}
            other=lookup[seed,cfg['name'],1,4]['screen_calls']
            assert len(other)==len(row['screen_calls'])
            for call,again in zip(row['screen_calls'],other):
                g=call['credit']['primal_gate'];gg=again['credit']['primal_gate'];patterns=g['skipped_patterns']
                assert call['input_pattern_hash']==again['input_pattern_hash'] and call['screened_pattern_hash']==again['screened_pattern_hash']
                assert patterns==gg['skipped_patterns'] and g['skipped']==len(patterns)
                assert all(key not in proofs for key in patterns)
                checks['causal_waves']+=1;checks['duplicate_repetition_waves']+=1;checks['skipped_occurrences']+=len(patterns)
                fresh=list(dict.fromkeys(key for key in patterns if key not in checked))
                if fresh:
                    regs=np.array([np.frombuffer(bytes.fromhex(key),np.uint8).reshape(bank.shape[1:]) for key in fresh])
                    upper=solver.values(regs)
                    assert np.all(np.isfinite(upper)) and np.all(upper<=0)
                    for key,reg,bounds in zip(fresh,regs,upper):
                        vals,den=exact.exact_values(x,v,reg,ai,ae);assert vals is not None
                        for value,bound in zip(vals,bounds):
                            num,bd=float(bound).as_integer_ratio();assert value<=0 and value*bd<=num*den
                            checks['exact_direction_checks']+=1;checks['strict_upper_enclosures']+=1
                        checked[key]=True;checks['unique_skipped_regions']+=1
            checks['preserved_proofs']+=len(proofs);checks['banks']+=1
            details.append(dict(seed=seed,method=cfg['name'],checks={k:checks[k]-start[k] for k in checks}))
            print(json.dumps(dict(banks=checks['banks'],exact_direction_checks=checks['exact_direction_checks'])),flush=True)
    result=dict(passed=True,checks=checks,details=details,source_sha256=sha(Path(__file__)),
        input_sha256={name:sha(inp/name) for name in ['protocol.json','rows.json','run_audit.json']},
        scope='All actual first-stage causal skips, unique patterns evaluated exactly; repeated sample RNG does not enter deterministic credit discovery; both repetition wave hashes and skipped patterns equal')
    out=root/'results/online_primal_gate/skip_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=True,checks=checks)),flush=True)


if __name__=='__main__':main()
