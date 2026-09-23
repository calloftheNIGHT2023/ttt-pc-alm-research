"""Full chain audit for 224/225 projection certificates and resource records."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/credit_halfspace_projection';inp=base/'development'
    p=json.loads((inp/'protocol.json').read_text());s=json.loads((inp/'summary.json').read_text())
    assert s['execution_complete'] and s['counts']==dict(regions=3998,old_proofs=732,attempted=3266,new_proofs=524,
        oracle_pairs=384739,mean_evaluations=19618161,exact_calls=524,target_limited=0,bracket_failures=0,nonfinite_failures=0,banks=96)
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert s['source_sha256']==sha(Path(__file__).with_name('run_credit_halfspace_projection.py'))
    assert s['protocol_sha256']==sha(inp/'protocol.json') and s['banks_sha256']==sha(inp/'banks.json')
    assert p['primitive_sha256']==sha(base/'primitive/summary.json')
    docs=root/'outputs/ttt-pc-alm-research';assert p['design_sha256']==sha(docs/'224_credit_halfspace_projection_design.md')
    banks=json.loads((inp/'banks.json').read_text());assert len(banks)==96
    for r in banks:
        assert sha(inp/r['arrays_file'])==r['arrays_sha256'] and sha(inp/r['meta_file'])==r['meta_sha256']
        assert sha(root/'results/light_h2_credit/full_bank_ceiling'/r['source_data_file'])==r['source_data_sha256']
    a=json.loads((base/'audit/summary.json').read_text());assert a['passed']
    assert a['checks']['new_exact']==524 and a['checks']['strict_synergy']==377 and a['checks']['previous_retained']==28 and a['checks']['previous_lost']==0
    assert a['projection_checks']['rows']==384215
    assert a['source_sha256']==sha(Path(__file__).with_name('audit_credit_halfspace_projection.py'))
    for name,value in a['input_sha256'].items():assert sha(inp/name)==value
    assert a['proofs_sha256']==sha(base/'audit/proofs.json') and a['projection_checks_sha256']==sha(base/'audit/projection_checks.json')
    assert a['ceiling_audit_sha256']==sha(root/'results/joint_credit_minimax/audit/summary.json')
    assert sum(r['native_total'] for r in a['alm_paired_sets'])==270 and sum(r['residual_total'] for r in a['alm_paired_sets'])==252
    assert sum(r['native_only'] for r in a['alm_paired_sets'])==54 and sum(r['residual_only'] for r in a['alm_paired_sets'])==36
    f=json.loads((base/'analysis/figure_audit.json').read_text());assert f['passed']
    assert f['source_sha256']==sha(Path(__file__).with_name('plot_credit_halfspace_projection.py'))
    assert f['main_sha256']==sha(inp/'summary.json') and f['audit_sha256']==sha(base/'audit/summary.json')
    assert f['baseline_sha256']==sha(root/'results/finite_credit_game/development/summary.json')
    assert f['figure_sha256']==sha(base/'analysis/credit_halfspace_projection.png')
    replays=0;positive=0;resources=0
    for seed in [5920000,5920015]:
        for method in p['methods']:
            r=json.loads((base/f'resources/{seed}_{method}.json').read_text());assert r['passed']
            assert r['source_sha256']==sha(Path(__file__).with_name('audit_credit_halfspace_resources.py')) and r['protocol_sha256']==sha(inp/'protocol.json')
            replays+=r['replayed_regions'];positive+=r['positive'];resources+=1
    assert replays==307 and positive==27 and resources==12
    links=0;figures=0
    for name in ['224_credit_halfspace_projection_design.md','225_credit_halfspace_projection_results.md','226_credit_moment_step_design.md']:
        path=docs/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            resolved=(path.parent/target).resolve();assert resolved.exists(),resolved;links+=1;figures+=int(resolved.suffix=='.png')
    assert figures==1
    result=dict(passed=True,frozen_sources=len(p['source_sha256']),counts=s['counts'],independent_checks=a['checks'],projection_checks=a['projection_checks'],
        fresh_processes=resources,resource_region_replays=replays,resource_positive_proofs=positive,links=links,figures=figures,
        source_sha256=sha(Path(__file__)),scope='Validated larger finite joint-credit coverage; no claim of online cost advantage or PC-ALM-specific task gain')
    out=root/'results/round_225_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
