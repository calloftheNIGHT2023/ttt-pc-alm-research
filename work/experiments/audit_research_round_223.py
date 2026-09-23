"""Frozen finite-credit run, all evidence hashes, resources and report links."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/finite_credit_game';inp=base/'development';p=json.loads((inp/'protocol.json').read_text());s=json.loads((inp/'summary.json').read_text())
    assert s['execution_complete'] and s['counts']==dict(banks=96,regions=3998,old_proofs=732,attempted=3266,new_proofs=28,oracle_pairs=417197,exact_calls=28)
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert s['source_sha256']==sha(Path(__file__).with_name('run_finite_credit_game.py'))
    assert s['protocol_sha256']==sha(inp/'protocol.json') and s['banks_sha256']==sha(inp/'banks.json')
    assert p['primitive_sha256']==sha(base/'primitive/summary.json')
    docs=root/'outputs/ttt-pc-alm-research';assert p['design_sha256']==sha(docs/'222_finite_credit_game_design.md')
    banks=json.loads((inp/'banks.json').read_text());assert len(banks)==96
    for r in banks:
        assert sha(inp/r['arrays_file'])==r['arrays_sha256'] and sha(inp/r['meta_file'])==r['meta_sha256']
        assert sha(root/'results/light_h2_credit/full_bank_ceiling'/r['source_data_file'])==r['source_data_sha256']
    a=json.loads((base/'audit/summary.json').read_text());assert a['passed'] and a['checks']['strict_synergy']==22
    assert a['source_sha256']==sha(Path(__file__).with_name('audit_finite_credit_game.py'))
    assert a['protocol_sha256']==sha(inp/'protocol.json') and a['banks_sha256']==sha(inp/'banks.json')
    assert a['main_summary_sha256']==sha(inp/'summary.json') and a['proofs_sha256']==sha(base/'audit/proofs.json')
    assert a['ceiling_audit_sha256']==sha(root/'results/joint_credit_minimax/audit/summary.json')
    f=json.loads((base/'analysis/figure_audit.json').read_text());assert f['passed']
    assert f['source_sha256']==sha(Path(__file__).with_name('plot_finite_credit_game.py'))
    assert f['main_sha256']==sha(inp/'summary.json') and f['audit_sha256']==sha(base/'audit/summary.json')
    assert f['figure_sha256']==sha(base/'analysis/finite_credit_game.png')
    resources=[(seed,method) for seed in [5920000,5920015] for method in p['methods']]
    resources += [(5920004,'alm_native'),(5920004,'alm_residual')]
    replays=0;positive=0
    for seed,method in resources:
        r=json.loads((base/f'resources/{seed}_{method}.json').read_text());assert r['passed']
        assert r['source_sha256']==sha(Path(__file__).with_name('audit_finite_credit_resources.py'))
        assert r['protocol_sha256']==sha(inp/'protocol.json');replays+=r['replayed_regions'];positive+=r['positive']
    assert replays==429 and positive==6
    links=0;figures=0
    for name in ['222_finite_credit_game_design.md','223_finite_credit_game_results.md','224_credit_halfspace_projection_design.md']:
        path=docs/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            resolved=(path.parent/target).resolve();assert resolved.exists(),resolved
            links+=1;figures+=int(resolved.suffix=='.png')
    assert figures==1
    result=dict(passed=True,frozen_sources=len(p['source_sha256']),counts=s['counts'],independent_checks=a['checks'],
        fresh_processes=len(resources),resource_region_replays=replays,resource_positive_proofs=positive,links=links,figures=figures,
        source_sha256=sha(Path(__file__)),scope='Finite joint-credit algorithm and safety result, not PC-ALM-independent task advantage')
    out=root/'results/round_223_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
