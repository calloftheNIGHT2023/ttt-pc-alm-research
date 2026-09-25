"""Set-level strengthening of the same-trajectory credit attribution audit."""
import argparse,hashlib,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();root=a.project/'results'
    paths=dict(cross=root/'cross_mode_credit_bank/diagnostic/proofs.json',bp_audit=root/'bp_credit_control/diagnostic/audits.json',
        local=root/'reused_local_credit/diagnostic/proofs.json')
    data={k:json.loads(v.read_text()) for k,v in paths.items()};rows=[];before=0;after_bp=0
    for old in data['bp_audit']:
        if old['method']!='alm16':continue
        seed=old['seed'];bp={r['pattern'] for r in data['cross'] if r['seed']==seed and r['task']=='alm16' and r['bank']=='bp32'}
        residual={r['pattern'] for r in data['cross'] if r['seed']==seed and r['task']=='alm16' and r['bank']=='residual32'}
        original=set(old['remaining_local_patterns']);before+=len(original);after_bp+=len(original-bp)
        for key in sorted(original-bp-residual):
            witness=next(r for r in data['local'] if r['seed']==seed and r['method']=='alm16' and r['pattern']==key)
            assert witness['exact']['positive'] and witness['lp_status']==2
            rows.append(dict(seed=seed,pattern=key,witness_credit=witness['credit'],exact=witness['exact'],lower=witness['lower']))
    result=dict(before_cross_mode_controls=before,after_bp_bank=after_bp,after_bp_and_residual_banks=len(rows),
        tasks_with_remaining_local_increment=len({r['seed'] for r in rows}),remaining=rows,
        input_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='same 16 old ALM trajectories; witnesses beyond contractor20, residual, current/history BP and fixed32 BP/residual cross banks; not all possible BP algorithms')
    out=root/'cross_mode_credit_bank/diagnostic/strong_overlap.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['remaining','input_sha256','audit_source_sha256']},indent=2))


if __name__=='__main__':main()
