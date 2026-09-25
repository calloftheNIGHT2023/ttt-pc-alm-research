"""390 matched-cohort official TTT and strong meta controls; training only.

All eight models consume the exact same materialized outer-training tensors.
No held-out research query targets are read or generated. Run only after 387
and the independent official-update preflight have passed; no auto dispatch.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(out, device_name):
    import numpy as np
    import torch
    import matched_official_ttt_v1 as bridge
    import matched_shifted_meta_models as population
    import official_ttt_meta_bridge as old
    import test_matched_official_ttt_v1 as preflight

    pre = preflight.OUT
    checked = read(pre / 'summary.json')
    assert checked['passed'] and not (pre / 'failure.json').exists()
    for relative, digest in checked['source_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    for relative, digest in checked['outputs_sha256'].items():
        assert sha(pre / relative) == digest, relative
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; no silent CPU fallback')
    original_protocol = ROOT / 'results/scalar_matched_batched/meta_prior/protocol.json'
    prior = read(original_protocol)
    cfg = {name:prior[name] for name in ('train_seed', 'initial_seed', 'steps', 'batch_size',
            'train_queries', 'validation_seeds', 'validation_queries', 'validation_interval')}
    configurations = [dict(kind='official_ttt', **c) for c in bridge.configs()]
    configurations.extend([
        dict(name='cohort_meta_ridge128', kind='ridge', rank=128, learning_rate=.001),
        dict(name='cohort_meta_shallow64_20', kind='shallow', width=64, inner_steps=20, learning_rate=.001)])
    source_files = [Path(__file__), Path(bridge.__file__), Path(population.__file__),
                    Path(old.__file__), Path(preflight.__file__),
                    ROOT / 'work/third_party/ttt-lm-pytorch/ttt.py',
                    ROOT / 'outputs/ttt-pc-alm-research/390_matched_official_ttt_design_v1.md']
    official_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=old.OFFICIAL_REPO, text=True).strip()
    assert official_commit == 'cd831db10c8c9a0f6340f02da5613316a8a92b67'
    protocol = dict(**cfg, configs=configurations, primary=bridge.PRIMARY, dtype='float64',
                    stages=list(bridge.STAGES), warm=True, device=str(device),
                    cpu_threads=1, torch_version=torch.__version__, platform=platform.platform(),
                    gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    official_commit=official_commit,
                    source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source_files},
                    parent_training_protocol_sha256=sha(original_protocol),
                    preflight_summary_sha256=sha(pre / 'summary.json'),
                    cohort_generation_device='cpu',
                    exact_training_tensors_shared_by_all_eight_models=True,
                    exact_training_tensors_equal_historical_cuda_cohort=False,
                    selection='minimum old-validation four-prefix raw MSE; earlier step wins ties',
                    no_test_evaluation=True, new_task_query_targets_accessed=False,
                    outer_bp_used=True, full_research_goal_complete=False)
    save(out / 'protocol.json', protocol)
    start_all = time.perf_counter()
    generator = torch.Generator(device='cpu').manual_seed(cfg['train_seed'])
    batches = [population.training_batch(cfg['batch_size'], cfg['train_queries'], generator, 'cpu')
               for _ in range(cfg['steps'])]
    # Store one cohort, not eight independently sampled histories with matching
    # seed labels. All arrays are on CPU and immutable during subsequent fits.
    cohort = tuple(torch.stack([batch[j] for batch in batches]) for j in range(4))
    del batches

    def batch_digest(batch):
        digest = hashlib.sha256()
        for tensor in batch:
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(str((array.shape, array.dtype.str)).encode('ascii'))
            digest.update(array.tobytes())
        return digest.hexdigest()

    fingerprints = [batch_digest(tuple(values[i] for values in cohort)) for i in range(cfg['steps'])]
    torch.save(dict(zip(('support_x', 'support_v', 'query_x', 'query_target'), cohort)), out / 'training_cohort.pt')
    save(out / 'cohort_manifest.json', dict(checkpoint_sha256=sha(out / 'training_cohort.pt'),
        batch_sha256=fingerprints, shape=[list(a.shape) for a in cohort],
        unique_tasks=cfg['steps'] * cfg['batch_size'],
        unique_generated_query_values=cfg['steps'] * cfg['batch_size'] * cfg['train_queries'],
        generation_and_archive_seconds=time.perf_counter() - start_all))
    validation = population.fixed_batch(cfg['validation_seeds'], cfg['validation_queries'], device)
    logs = []; summaries = []

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    for configuration in configurations:
        torch.manual_seed(cfg['initial_seed'])
        if configuration['kind'] == 'official_ttt':
            model = bridge.MatchedOfficialTTT(configuration['head_dim'], configuration['inner_passes'], configuration['key_basis'], configuration['rate_mode']).double()
            objective = bridge.trajectory_loss
        else:
            model = population.make_model(configuration)
            objective = population.trajectory_loss
        model.to(device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=configuration['learning_rate'])
        best = float('inf'); selected = None; selected_step = None
        consumed = []; validation_events = 0
        synchronize()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        for step in range(cfg['steps'] + 1):
            if step:
                cpu_batch = tuple(values[step - 1] for values in cohort)
                digest = batch_digest(cpu_batch)
                assert digest == fingerprints[step - 1]
                consumed.append(digest)
                batch = tuple(value.to(device) for value in cpu_batch)
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = objective(model, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(('non-finite training loss', configuration['name'], step))
                loss.backward()
                if not all(p.grad is not None for p in parameters):
                    raise RuntimeError(('missing gradient', configuration['name'], step))
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
            if step % cfg['validation_interval'] == 0 or step == cfg['steps']:
                model.eval()
                with torch.no_grad():
                    value = float(objective(model, validation).cpu())
                if not np.isfinite(value):
                    raise RuntimeError(('non-finite validation', configuration['name'], step))
                validation_events += 1
                improved = value < best
                if improved:
                    best = value; selected_step = step
                    selected = {name:t.detach().cpu().clone() for name, t in model.state_dict().items()}
                event = dict(method=configuration['name'], step=step, validation_raw_mse=value,
                             best_validation_raw_mse=best, selected_as_best=improved,
                             train_loss=float(loss.detach().cpu()) if step else None,
                             gradient_norm_before_clip=float(norm.detach().cpu()) if step else None)
                logs.append(event); save(out / 'training_log.json', logs)
                print(json.dumps(event), flush=True)
        synchronize()
        seconds = time.perf_counter() - begin
        assert selected is not None and consumed == fingerprints
        checkpoint = out / (configuration['name'] + '.pt')
        torch.save(selected, checkpoint)
        manifest_path = out / (configuration['name'] + '_consumed_batches.json')
        save(manifest_path, consumed)
        record = dict(method=configuration['name'], outer_training_seconds=seconds,
                      outer_tasks=cfg['steps'] * cfg['batch_size'],
                      unique_outer_query_values=cfg['steps'] * cfg['batch_size'] * cfg['train_queries'],
                      outer_query_loss_exposures=cfg['steps'] * cfg['batch_size'] * cfg['train_queries'] * len(bridge.STAGES),
                      validation_events=validation_events,
                      validation_query_loss_exposures=validation_events * len(cfg['validation_seeds']) * cfg['validation_queries'] * len(bridge.STAGES),
                      trainable_parameters=sum(p.numel() for p in parameters),
                      selected_step=selected_step, best_validation_raw_mse=best,
                      checkpoint_sha256=sha(checkpoint), consumed_batches_sha256=sha(manifest_path),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
                      memory_scope='CUDA allocator process-wide; not total driver/host memory',
                      new_task_query_targets_accessed=False)
        summaries.append(record); save(out / 'training_summary.json', summaries)
        print(json.dumps(dict(completed_method=configuration['name'], **record)), flush=True)
        # Discard model-dependent state before constructing the next method.
        del model, optimizer, parameters, selected
        if step:
            del loss, norm, batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        for relative, digest in protocol['source_sha256'].items():
            assert sha(ROOT / relative) == digest, relative
    save(out / 'summary.json', dict(passed=True, trained_models=len(summaries),
        exact_cohort_identity_verified=True, training_only=True, new_task_query_targets_accessed=False,
        seconds=time.perf_counter() - start_all,
        outputs_sha256={name:sha(out / name) for name in ('protocol.json', 'cohort_manifest.json', 'training_cohort.pt', 'training_summary.json', 'training_log.json')},
        full_research_goal_complete=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    # This stdlib-only guard runs before a heavy runtime import.
    from test_matched_official_ttt_v1 import require_finished_deadline, OUT as PREFLIGHT
    require_finished_deadline()
    assert (PREFLIGHT / 'summary.json').exists() and read(PREFLIGHT / 'summary.json')['passed']
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args.out, args.device)
    except Exception:
        save(args.out / 'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
