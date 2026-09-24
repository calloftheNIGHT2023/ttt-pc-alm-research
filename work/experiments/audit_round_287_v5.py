"""DRAFT: link all287 evidence after numerical audits AND actual visual review.

Defaults to two old tasks. No algorithm or teacher is executed by this file.
This seals provenance/census, not another independent statistical calculation.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(directory, relative):
    path = (directory / str(relative).replace('\\', '/')).resolve()
    require(path.is_relative_to(directory.resolve()), f'Escaping path: {relative}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--stage', choices=['preflight', 'confirmation'], default='preflight')
    args = parser.parse_args()
    root = args.project.resolve()
    base = root / 'results/probe_credit_confirmation'
    require(not (base / 'predictions/RUNNING.lock').exists(), 'Wait for live prediction timing')
    preflight = args.stage == 'preflight'
    suffix = '_preflight' if preflight else ''
    output = root / ('results/round_287_preflight_audit_v5.json' if preflight else 'results/round_287_audit_v5.json')
    require(not output.exists(), 'Preserve existing completed audit')
    parent = root / 'results/round_286_audit.json'
    old = read(parent)
    require(old['passed'], 'Parent audit failed')
    hashes = dict(old['source_sha256'])
    reports = dict(old['report_sha256'])
    summaries = {}
    counts = Counter()

    def check_files(directory, entries):
        for name, digest in entries.items():
            require(sha(inside(directory, name)) == digest, f'Hash differs: {directory}/{name}')
            counts['hashed_manifest_entries'] += 1

    def load_stage(name):
        directory = base / name
        require(not (directory / 'RUNNING.lock').exists(), f'Stage still active: {name}')
        summary = read(directory / 'summary.json')
        protocol = read(directory / 'protocol.json')
        require(summary['passed'], f'Stage failed: {name}')
        check_files(directory, summary.get('outputs_sha256', {}))
        if 'protocol_sha256' in summary:
            require(summary['protocol_sha256'] == sha(directory / 'protocol.json'), name)
        for filename, digest in protocol['source_sha256'].items():
            require(filename not in hashes or hashes[filename] == digest, f'Source lineage differs: {filename}')
            hashes[filename] = digest
        summaries[name] = sha(directory / 'summary.json')
        return directory, summary, protocol

    _, plan, _ = load_stage('planning')
    require(plan['planned_tasks'] == 8192, 'Planning task count differs')
    _, heads, _ = load_stage('head_preflight')
    _, head_audit, _ = load_stage('head_audit')
    require(heads['timing_calls'] == 192 and heads['memory_calls'] == 16, 'Extra head resource census')
    require(head_audit['counts']['predictors'] == 32, 'Extra head predictor census')
    _, resume, _ = load_stage('runner_preflight_v2')
    _, resume_audit, _ = load_stage('runner_audit_v2')
    require(resume['actual_resume_test_passed'] and resume['predictors'] == 54, 'Resume preflight')
    require(resume_audit['runner_preflight_summary_sha256'] == summaries['runner_preflight_v2'], 'Resume audit link')
    pred, ps, pp = load_stage('runner_preflight_v2' if preflight else 'predictions')
    n = 2 if preflight else 8192
    seeds = [5910000, 5910063] if preflight else list(range(5500000, 5508192))
    names = pp['methods']
    require(pp['seeds'] == seeds and len(names) == len(set(names)) == 27, 'Frozen data/method list differs')
    require(ps['tasks'] == n and ps['predictors'] == n * 27 and ps['checkpoint_replay_verified'], 'Prediction census')
    require(ps['query_targets_accessed'] is False, 'Prediction phase accessed queries')
    before = read(pred / 'before_query_manifest.json')
    require(before['tasks'] == n and before['predictors'] == n * 27, 'Before-query census')
    require(before['protocol_sha256'] == sha(pred / 'protocol.json') and before['source_sha256'] == pp['source_sha256'], 'Before-query lineage')
    require(before['query_targets_accessed'] is False and [r['seed'] for r in before['task_commits']] == seeds, 'Before-query seed order')
    commits = {}
    failures = 0
    for seed, entry in zip(seeds, before['task_commits']):
        path = inside(pred, entry['file'])
        require(path == (pred / 'tasks' / str(seed) / 'commit.json').resolve() and sha(path) == entry['sha256'], 'Task commit link')
        commit = read(path)
        require(commit['seed'] == seed and commit['methods'] == names and commit['protocol_sha256'] == before['protocol_sha256'], 'Task identity')
        require(commit['query_targets_accessed'] is False and len(commit['files']) == 82, 'Task file census')
        check_files(pred, commit['files'])
        rows = read(inside(pred, commit['rows_file']))
        require(len(rows) == 27 and {r['method'] for r in rows} == set(names), 'Method census in task')
        require(sum(r['metadata']['execution_failed'] for r in rows) == commit['failures'] == entry['failures'], 'Failure census')
        for attempt in commit['interrupted_attempt_costs']['attempts']:
            check_files(pred, attempt['files'])
        failures += commit['failures']
        commits[seed] = entry['sha256']
    require(failures == ps['numerical_failures'], 'Global failure census')
    pa, pas, pap = load_stage('prediction_audit' + suffix + '_v2')
    ta, tas, tap = load_stage('path_audit' + suffix + '_v5')
    pred_digest = sha(pred / 'summary.json')
    for directory, summary, protocol, expected in [(pa, pas, pap, n), (ta, tas, tap, min(n, 64))]:
        require(summary['counts']['tasks'] == expected and summary['counts']['predictors'] == 27 * expected, 'Independent audit census')
        require(summary['query_targets_accessed'] is False and protocol['prediction_summary_sha256'] == pred_digest, 'Audit prediction link')
        index = read(directory / 'files.json')
        require(len(index) == expected, 'Independent audit task file count')
        check_files(directory, index)
        seen = []
        for relative in index:
            record = read(inside(directory, relative))
            seen.append(record['seed'])
            require(record['protocol_sha256'] == sha(directory / 'protocol.json'), 'Audit task protocol link')
            require(record['prediction_commit_sha256'] == commits[record['seed']] and not record['query_targets_accessed'], 'Audit task prediction link')
            check_files(directory, record.get('files', {}))
        require(sorted(seen) == seeds[:expected], 'Independent audit fixed task census')
    require(tap['all_prediction_audit_sha256'] == sha(pa / 'summary.json'), 'Path/all-prediction link')
    ev, es, ep = load_stage('evaluation' + suffix + '_v5')
    ea, eas, eap = load_stage('evaluation_audit' + suffix + '_v5')
    fig, fs, fp = load_stage('figures' + suffix + '_v6')
    require(ep['prediction_summary_sha256'] == pred_digest and ep['prediction_audit_sha256'] == sha(pa / 'summary.json'), 'Evaluation gates')
    require(ep['path_audit_sha256'] == sha(ta / 'summary.json'), 'Evaluation path gate')
    require(eap['evaluation_summary_sha256'] == eas['evaluation_summary_sha256'] == sha(ev / 'summary.json'), 'Independent risk link')
    require(fp['evaluation_summary_sha256'] == sha(ev / 'summary.json') and fp['evaluation_audit_sha256'] == sha(ea / 'summary.json'), 'Figure audit links')
    require(ep['seeds'] == seeds and ep['methods'] == names and ep['primary'] == pp['primary'], 'Evaluation method/data identity')
    require(ep['metrics'] == ['mse257', 'mse129', 'point_mse257', 'point_mse129'] and ep['primary_metric'] == 'mse257', 'Frozen metrics')
    require(ep['bootstrap']['repetitions'] == 100000 and ep['bootstrap']['seed'] == 287193, 'Frozen resampling')
    require(ep['bootstrap']['primary_adjusted_upper_quantile'] == 1 - .05 / 25, 'Frozen comparison threshold')
    controls = [name for name in names if name != pp['primary']]
    family = [name for name in controls if name != 'probe_all_alm64']
    require(ep['controls'] == controls and ep['primary_family'] == family and len(family) == 25, 'Frozen main family')
    methods = read(ev / 'methods.json')
    comparisons = read(ev / 'comparisons.json')
    require([r['method'] for r in methods] == names, 'All method tables required')
    require(len(comparisons) == 104 and {(r['control'], r['metric']) for r in comparisons} == {(name, metric) for name in controls for metric in ep['metrics']}, 'All contrasts required')
    passed = 0
    for row in comparisons:
        main = row['control'] in family and row['metric'] == 'mse257'
        require(row['tasks'] == n and row['primary'] == pp['primary'] and row['predeclared_main_comparison'] == main, 'Contrast identity')
        if main:
            decision = row['bonferroni_one_sided_upper'] < 0
            require(row['adjusted_upper_below_zero'] == decision, 'Frozen decision differs')
            passed += decision
        else:
            require(row['bonferroni_one_sided_upper'] is None and row['adjusted_upper_below_zero'] is None, 'Secondary promoted to main')
    for summary in [es, eas]:
        require(summary['tasks'] == n and summary['predictors'] == n * 27 and summary['comparisons'] == 104, 'Statistical audit census')
        require(summary['main_adjusted_negative_upper_bounds'] == passed and summary['core_research_goal_complete'] is False, 'Scientific status differs')
    require(es['functional_preflight_only'] == preflight and es['all25_adjusted_negative_upper_bounds'] == (passed == 25), 'Evaluation scope')
    require(es['numerical_failures'] == failures and eas['independent_task_losses'] == n * 27 * 4, 'Independent risk census')
    require(eas['bootstrap_replicates'] == 100000 and eas['bootstrap_scalar_means'] == 10400000 and eas['old_and_actual_resource_classes'] == 54, 'Statistical/resource coverage')
    require(fs['tasks'] == n and fs['method_tables'] == 27 and fs['contrast_rows'] == 104 and fs['all_methods_retained'], 'Figure coverage')
    require(fs['functional_preflight_only'] == preflight and fs['core_research_goal_complete'] is False, 'Figure scientific scope')
    require(fp['method_order'] == names and fp['contrast_order'] == controls, 'Figure order')
    qa_path = fig / 'visual_review.json'
    qa = read(qa_path)
    reviewed = {name: sha(fig / name) for name in ['287_all_methods.png', '287_all_contrasts.png', 'report.md']}
    require(qa['passed'] is True and qa['stage'] == args.stage and qa['reviewed_files'] == reviewed, 'Actual visual review required')
    require(qa.get('reviewer') and qa.get('reviewed_utc'), 'Visual review attestation identity')
    for key in ['layout_readable', 'no_clipping', 'all_methods_retained', 'interval_legend_correct', 'preflight_or_confirmation_label_correct', 'resource_and_evidence_limits_present']:
        require(qa['checks'].get(key) is True, 'Visual review check missing: ' + key)
    docs = root / 'outputs/ttt-pc-alm-research'
    for name, digest in reports.items():
        require(sha(inside(docs, name)) == digest, 'Historical report changed: ' + name)
    moment = docs / '287_ols_moment_addendum.md'
    require(fp['moment_note_sha256'] == sha(moment), 'Moment qualification changed')
    reports[moment.name] = sha(moment)
    geometry_note = docs / '287_exact_geometry_certificate.md'
    reports[geometry_note.name] = sha(geometry_note)
    pool_note = docs / '287_pool_budget_scope_correction.md'
    require(tap['component_acceptance_revision_explicit'] is True, 'Pool acceptance correction must be explicit')
    require(tap['pool_budget_note_sha256'] == fp['pool_budget_note_sha256'] == sha(pool_note), 'Pool budget qualification changed')
    require(tap['tolerances']['nominal_pool_readout_budget'] == 1e-12, 'Final pool budget changed')
    reports[pool_note.name] = sha(pool_note)
    collapse_note = docs / '287_collapsed_vertex_certificate.md'
    require(tap['vertex_collapse_revision_explicit'] is True, 'Vertex collapse certificate extension must be explicit')
    require(tap['vertex_collapse_note_sha256'] == fp['vertex_collapse_note_sha256'] == sha(collapse_note), 'Vertex collapse note changed')
    reports[collapse_note.name] = sha(collapse_note)
    head_note = docs / '287_head_reference_diagnosis.md'
    require(fp['head_reference_note_sha256'] == sha(head_note), 'Head arithmetic qualification changed')
    reports[head_note.name] = sha(head_note)
    for relative, digest in pap['precise_reference_prerequisites'].items():
        require(sha(inside(base, relative)) == digest, 'Precise-reference prerequisite changed: ' + relative)
    runtime_note = docs / '287_runtime_resource_observation_20260924.md'
    require(fp['runtime_interference_note_sha256'] == sha(runtime_note), 'Runtime qualification changed')
    require(fp['publication_interference_note_sha256'] == sha(root / 'release/SCOPE.md'), 'Publication qualification changed')
    reports[runtime_note.name] = sha(runtime_note)
    for relative, digest in tap['certificate_prerequisites'].items():
        require(sha(inside(base, relative)) == digest, 'Certificate prerequisite changed: ' + relative)
    audit_plan = docs / '287_final_audit_plan.md'
    reports[audit_plan.name] = sha(audit_plan)
    hashes[Path(__file__).name] = sha(Path(__file__))
    for name, digest in hashes.items():
        require(sha(inside(root / 'work/experiments', name)) == digest, 'Frozen source changed: ' + name)
    if not preflight:
        gate = read(root / 'results/round_287_preflight_audit_v5.json')
        require(gate['passed'] and gate['functional_preflight_only'] and gate['source_sha256'] == hashes, 'Same-source old final-audit preflight required')
        for name in ['prediction_audit', 'path_audit', 'evaluation', 'evaluation_audit', 'figures']:
            version = '_v2' if name == 'prediction_audit' else '_v6' if name == 'figures' else '_v5'
            _, old_summary, old_protocol = load_stage(name + '_preflight' + version)
            require(old_protocol['source_sha256'] == read(base / (name + version) / 'protocol.json')['source_sha256'], 'Old/new stage sources differ: ' + name)
    result = dict(passed=True, stage=args.stage, functional_preflight_only=preflight,
                  parent_audit_sha256=sha(parent), source_count=len(hashes), source_sha256=hashes,
                  report_sha256=reports, summary_sha256=summaries, checks=dict(counts),
                  tasks=n, predictors=n * 27, fixed_path_audit_tasks=min(n, 64), comparisons=104,
                  main_family_size=25, main_adjusted_negative_upper_bounds=passed,
                  all25_adjusted_negative_upper_bounds=passed == 25,
                  visual_review_sha256=sha(qa_path),
                  core_research_goal_complete=False,
                  scope='Provenance and full census seal, not a second numerical audit or universal superiority theorem',
                  resource_interference_note_sha256=sha(root / 'release/SCOPE.md') if (root / 'release/SCOPE.md').exists() else None,
                  next='Report every frozen control and resource gap; assess general-MLP/official-TTT bridge separately')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({k: v for k, v in result.items() if k not in ['source_sha256', 'report_sha256', 'summary_sha256']}), flush=True)


if __name__ == '__main__':
    main()
