"""One-shot orchestration of already old-preflighted confirmation stages.

Waits for the SAME live all-predictor audit, never restarts it. Each subsequent
stage retains its own full evidence gates. Any failure stops the pipeline.
Final visual inspection/sealing remains an actual human/model viewing step.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def now(): return datetime.now(timezone.utc).isoformat()


def verify_sources(root, seal):
    for name, digest in seal['source_sha256'].items():
        assert sha(root/'work/experiments'/name) == digest, name


def complete(folder):
    assert not (folder/'RUNNING.lock').exists(), ('Stage still locked', str(folder))
    summary = read(folder/'summary.json'); assert summary['passed']
    for name, digest in summary['outputs_sha256'].items(): assert sha(folder/name) == digest
    return summary


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve(); base = root/'results/probe_credit_confirmation'
    out = base/'pipeline_execution_v4'; assert not out.exists(), 'Inspect existing execution; no automatic retry'
    seal_path = root/'results/round_287_preflight_audit_v4.json'; seal = read(seal_path)
    assert seal['passed'] and seal['functional_preflight_only'] and seal['tasks'] == 2
    assert not seal['core_research_goal_complete']
    verify_sources(root, seal)
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    pred = base/'prediction_audit_v2'; identity = None
    if not (pred/'summary.json').exists():
        identity = read(pred/'RUNNING.lock'); process = psutil.Process(identity['pid'])
        assert process.create_time() == identity['create_time'] and process.is_running()
        command = process.cmdline()
        assert any(Path(arg).name == 'audit_probe_confirmation_predictions_v2.py' for arg in command)
        assert '--stage' in command and command[command.index('--stage')+1] == 'confirmation'
        assert Path(process.cwd()).resolve() == root
    stages = [
        ('paths', 'audit_probe_confirmation_paths_v4.py', 'path_audit_v4'),
        ('evaluation', 'evaluate_probe_credit_confirmation_v4.py', 'evaluation_v4'),
        ('statistics', 'audit_probe_credit_confirmation_evaluation_v4.py', 'evaluation_audit_v4'),
        ('figures', 'plot_probe_credit_confirmation_v5.py', 'figures_v5'),
    ]
    for _, script, folder in stages:
        assert script in seal['source_sha256'] and not (base/folder).exists(), folder
    out.mkdir(); exclusive_json(out/'protocol.json', dict(
        created_utc=now(), driver_source_sha256=sha(Path(__file__)),
        old_complete_preflight_sha256=sha(seal_path), existing_prediction_audit_identity=identity,
        stages=stages, no_restart=True, no_algorithm_changes=True, component_gate_scope_changed=True,
        pool_budget_scope_note_sha256=sha(root/'outputs/ttt-pc-alm-research/287_pool_budget_scope_correction.md'),
        query_access_policy='Only the frozen evaluator after all-prediction and fixed-first64 path gates; any stage failure stops',
        final_visual_review_required=True))
    print(json.dumps(dict(stage='waiting_existing_all_prediction_audit', identity=identity, utc=now())), flush=True)
    try:
        while identity is not None:
            try:
                process = psutil.Process(identity['pid'])
                live = process.create_time() == identity['create_time'] and process.is_running()
            except psutil.NoSuchProcess: live = False
            if not live: break
            time.sleep(10)
        audited = complete(pred)
        assert audited['counts']['tasks'] == 8192 and audited['counts']['predictors'] == 221184
        assert not audited['query_targets_accessed']
        exclusive_json(out/'all_prediction_complete.json', dict(utc=now(), summary_sha256=sha(pred/'summary.json')))
        for label, script, folder in stages:
            verify_sources(root, seal)
            command = [sys.executable, '-u', str(root/'work/experiments'/script), '--project', str(root), '--stage', 'confirmation']
            child = subprocess.Popen(command, cwd=root, env=os.environ.copy())
            try: creation = psutil.Process(child.pid).create_time()
            except psutil.NoSuchProcess: creation = None
            exclusive_json(out/(label+'_start.json'), dict(utc=now(), pid=child.pid, create_time=creation,
                script=script, script_sha256=sha(root/'work/experiments'/script)))
            print(json.dumps(dict(stage=label, event='started', pid=child.pid, utc=now())), flush=True)
            code = child.wait()
            exclusive_json(out/(label+'_exit.json'), dict(utc=now(), returncode=code))
            assert code == 0, (label, code, 'Preserve failure and stop without retry')
            complete(base/folder)
            exclusive_json(out/(label+'_complete.json'), dict(utc=now(), summary_sha256=sha(base/folder/'summary.json')))
        verify_sources(root, seal)
        answer = dict(passed=True, utc=now(), stages_completed=4, final_visual_review_required=True,
            core_research_goal_complete=False, next='Open both confirmation figures and read report; only then write actual visual review and run audit_round_287_v4.py --stage confirmation')
        exclusive_json(out/'summary.json', answer); print(json.dumps(answer), flush=True)
    except BaseException as exc:
        exclusive_json(out/'failure.json', dict(utc=now(), error_type=type(exc).__name__, message=str(exc),
            no_restart=True, next='Inspect exact failed stage, preserve evidence, no ungated query access'))
        raise


if __name__ == '__main__': main()
