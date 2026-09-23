"""Artifact chain and independent recomputation of the interleaved benchmark."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/credit_moment_step';inp=base/'development';docs=root/'outputs/ttt-pc-alm-research'
    p=read(inp/'protocol.json');s=read(inp/'summary.json');a=read(base/'audit/summary.json')
    assert s['execution_complete'] and a['passed']
    assert s['counts']==dict(regions=3998,old_proofs=732,attempted=3266,new_proofs=453,oracle_pairs=391007,mean_evaluations=781108,
        variance_evaluations=390554,exact_calls=453,target_limited=0,zero_variance=0,nonfinite_failures=0,already_at_target=0,banks=96)
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert s['source_sha256']==sha(Path(__file__).with_name('run_credit_moment_step.py'))
    assert s['protocol_sha256']==sha(inp/'protocol.json') and s['banks_sha256']==sha(inp/'banks.json')
    assert p['primitive_sha256']==sha(base/'primitive/summary.json') and p['design_sha256']==sha(docs/'226_credit_moment_step_design.md')
    banks=read(inp/'banks.json');assert len(banks)==96
    for r in banks:
        assert sha(inp/r['arrays_file'])==r['arrays_sha256'] and sha(inp/r['meta_file'])==r['meta_sha256']
        assert sha(root/'results/light_h2_credit/full_bank_ceiling'/r['source_data_file'])==r['source_data_sha256']
    assert a['checks']==dict(banks=96,regions=3998,replayed_regions=3266,replayed_arrays=1152,old_proofs=732,new_exact=453,
        exact_convex_mixtures=453,exact_rounding_bounds=453,full_region_infeasible=453,strict_synergy=330)
    assert a['moment_checks']['rows']==390554 and a['moment_checks']['max_post_minus_target']<0
    assert a['source_sha256']==sha(Path(__file__).with_name('audit_credit_moment_step.py'))
    for name,value in a['input_sha256'].items():assert sha(inp/name)==value
    assert a['proofs_sha256']==sha(base/'audit/proofs.json') and a['moment_checks_sha256']==sha(base/'audit/moment_checks.json')
    assert a['ceiling_audit_sha256']==sha(root/'results/joint_credit_minimax/audit/summary.json')
    changed={key:sum(r['comparison']['projection'][key] for r in a['summaries']) for key in ['retained','lost','new']}
    assert changed==dict(retained=441,lost=83,new=12)
    assert sum(r['comparison']['fixed']['retained'] for r in a['summaries'])==28
    assert sum(r['comparison']['fixed']['lost'] for r in a['summaries'])==0
    paired={key:sum(r[key] for r in a['alm_paired_sets']) for key in ['native_total','residual_total','native_only','residual_only']}
    assert paired['native_total']==256 and paired['residual_total']==244
    timing=base/'timing';tp=read(timing/'protocol.json');ts=read(timing/'summary.json');tr=read(timing/'rows.json');assert ts['passed']
    for name,value in tp['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert ts['source_sha256']==sha(Path(__file__).with_name('benchmark_credit_solvers.py'))
    assert tp['design_sha256']==sha(docs/'227_credit_solver_timing_protocol.md') and tp['audit_sha256']==sha(base/'audit/summary.json')
    assert ts['protocol_sha256']==sha(timing/'protocol.json') and ts['rows_sha256']==sha(timing/'rows.json')
    assert len(tr)==ts['solves']==576 and ts['proof_checks']==2*(28+524+453) and ts['array_checks']==5760
    assert Counter((r['variant'],r['rank']) for r in tr)==Counter({(v,k):64 for v in ['fixed','projection','moment'] for k in range(3)})
    variants={'fixed':'finite_credit_game','projection':'credit_halfspace_projection','moment':'credit_moment_step'};gold={}
    for variant,folder in variants.items():
        path=root/f'results/{folder}/development';assert tp['input_hashes'][variant]==sha(path/'banks.json')
        for r in read(path/'banks.json'):gold[variant,r['seed'],r['method']]=r
    lookup={(r['seed'],r['method'],r['repeat'],r['variant']):r for r in tr};assert len(lookup)==576
    for r in tr:
        assert r['positive']==gold[r['variant'],r['seed'],r['method']]['new_proofs']
        assert r['seconds']>0 and r['external_guard_seconds']>=r['seconds']
        assert [t['step'] for t in r['traces']]==[1,4,16,64,128]
        for t in r['traces']:assert t['positive']==gold[r['variant'],r['seed'],r['method']]['prefix_proofs'][str(t['step'])]
    rng=np.random.default_rng(tp['bootstrap_seed']);boot=rng.integers(0,len(p['seeds']),(tp['bootstrap_samples'],len(p['seeds'])))
    for r in ts['summaries']:
        method=r['method'];values={v:np.array([np.mean([lookup[seed,method,rep,v]['seconds'] for rep in range(2)]) for seed in p['seeds']]) for v in variants}
        for v in variants:
            assert r['mean_seconds'][v]==float(values[v].mean())
            assert r['full_128_new'][v]==sum(gold[v,seed,method]['new_proofs'] for seed in p['seeds'])
        for v in ['fixed','projection']:
            diff=values['moment']-values[v];q=r['comparisons'][v]
            assert q['moment_minus_baseline_mean_seconds']==float(diff.mean())
            assert np.array_equal(q['paired_bootstrap95'],np.quantile(diff[boot].mean(1),[.025,.975]))
            assert q['baseline_mean_div_moment_mean']==float(values[v].mean()/values['moment'].mean())
            envelope=[]
            for seed in p['seeds']:
                for rep in range(2):
                    m=lookup[seed,method,rep,'moment'];affordable=[t for t in lookup[seed,method,rep,v]['traces'] if t['elapsed_seconds']<=m['seconds']]
                    last=affordable[-1] if affordable else dict(step=0,positive=0)
                    envelope.append(dict(seed=seed,repeat=rep,step=last['step'],positive=last['positive']))
            assert r['prefix_envelope'][v]==envelope
            assert r['prefix_envelope_mean_count_per_repeat'][v]==sum(t['positive'] for t in envelope)/2
    resources=[]
    for seed in [5920000,5920015]:
        for method in p['methods']:
            r=read(base/f'resources/{seed}_{method}.json');assert r['passed']
            assert r['source_sha256']==sha(Path(__file__).with_name('audit_credit_moment_resources.py')) and r['protocol_sha256']==sha(inp/'protocol.json')
            assert r['positive']==gold['moment',seed,method]['new_proofs'];resources.append(r)
    assert sum(r['replayed_regions'] for r in resources)==307 and sum(r['positive'] for r in resources)==28
    f=read(base/'analysis/figure_audit.json');assert f['passed']
    assert f['source_sha256']==sha(Path(__file__).with_name('plot_credit_moment_step.py'))
    assert f['timing_sha256']==sha(timing/'summary.json') and f['audit_sha256']==sha(base/'audit/summary.json')
    assert f['figure_sha256']==sha(base/'analysis/credit_moment_step.png')
    links=0;figures=0
    for name in ['226_credit_moment_step_design.md','227_credit_solver_timing_protocol.md','228_credit_moment_step_results.md','229_common_pool_credit_protocol.md']:
        path=docs/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            resolved=(path.parent/target).resolve();assert resolved.exists(),resolved;links+=1;figures+=int(resolved.suffix=='.png')
    assert figures==1
    result=dict(passed=True,frozen_sources=len(p['source_sha256']),timing_sources=len(tp['source_sha256']),counts=s['counts'],checks=a['checks'],
        moment_checks=a['moment_checks'],changed_sets=changed,alm_paired_sets=paired,timing_solves=576,timing_array_checks=ts['array_checks'],
        timing_proof_checks=ts['proof_checks'],timing_summaries_recomputed=6,fresh_processes=12,resource_region_replays=307,resource_positive_proofs=28,
        links=links,figures=figures,source_sha256=sha(Path(__file__)),scope='Analytic component cost-coverage result; no independent PC-ALM task-superiority claim')
    out=root/'results/round_228_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
