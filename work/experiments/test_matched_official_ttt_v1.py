"""390 independent fast-update/stream isolation checks, no new test queries.

Waits for the ongoing formal deadline experiment to finish BEFORE importing
Torch; this preflight must not compete with formal online timing.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import inspect
import io
import json
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/matched_official_ttt/preflight_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def require_finished_deadline():
    directory = ROOT / 'results/new_task_deadline_risk'
    for relative in ('development_v1/summary.json', 'audit_v1/summary.json', 'evaluation_v1/summary.json'):
        path = directory / relative
        if not path.exists() or not json.loads(path.read_text(encoding='utf-8'))['passed']:
            raise RuntimeError('Do not run Torch preflight during formal timing: ' + relative)


def run():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matched_official_ttt_v1 as model_source
    import matched_shifted_meta_models as population
    import official_ttt_meta_bridge as old
    import streaming_branch_projection as base

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    official = ROOT / 'work/third_party/ttt-lm-pytorch'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=official, text=True).strip()
    assert commit == 'cd831db10c8c9a0f6340f02da5613316a8a92b67', commit
    subprocess.check_call(['git', 'diff', '--exit-code', 'HEAD', '--', 'ttt.py'], cwd=official)
    files = [Path(__file__), Path(model_source.__file__), Path(population.__file__),
             Path(old.__file__), Path(base.__file__), Path(inspect.getfile(base.forward)), official / 'ttt.py']
    protocol = dict(source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
                    official_commit=commit, torch_version=torch.__version__,
                    transformers_version=importlib.metadata.version('transformers'),
                    task_seeds=[5920000, 5920001], new_task_queries_accessed=False,
                    role='baseline interface preflight; no training or efficacy claim')
    save(OUT / 'protocol.json', protocol)
    data = population.fixed_batch(protocol['task_seeds'], 11, 'cpu')
    x, v, q, target = data
    teacher_gap = 0.
    for index, seed in enumerate(protocol['task_seeds']):
        truth = np.random.default_rng(seed).uniform(-.12, .12, 4)
        teacher_gap = max(teacher_gap, float(np.max(abs(base.forward(q[index].numpy(), truth) - target[index].numpy()))))
    assert teacher_gap < 1e-14

    # Independent autograd reference: one minibatch is a simultaneous gradient
    # step at the current weights, not sequential SGD at every support token.
    def reference_adapt(model, sx, sv, state=None):
        with torch.no_grad():
            key = model.key_features(sx)[:, None]
            value = key + model.value_code(sv)[:, None]
            reconstruction_target = value - key
            if model.rate_mode == 'native':
                gate = torch.sigmoid(F.linear(key[:, 0], model.learner.learnable_ttt_lr_weight[0],
                                             model.learner.learnable_ttt_lr_bias[0]))
                scale = (model.learner.token_idx[sx.shape[1]-1] + model.learner.learnable_token_idx[sx.shape[1]-1]).clamp_min(0)
                weights = gate[:, None] * (model.learner.config.ttt_base_lr / model.head_dim) * scale
            else:
                weights = torch.sigmoid(model.inner_lr_logit) / model.head_dim
            initial = model.initial_fast_state(len(sx)) if state is None else state
        parameters = [initial[name].detach().clone().requires_grad_() for name in model_source.FAST_NAMES]
        gamma = model.learner.ttt_norm_weight.detach().reshape(model.head_dim)
        beta = model.learner.ttt_norm_bias.detach().reshape(model.head_dim)
        for _ in range(model.inner_passes):
            w1, b1, w2, b2 = parameters
            z2 = F.gelu(key @ w1 + b1, approximate='tanh') @ w2 + b2
            code = F.layer_norm(z2, (model.head_dim,), gamma, beta, eps=1e-6)
            loss = .5 * ((code - reconstruction_target).square() * weights).sum()
            gradients = torch.autograd.grad(loss, parameters)
            parameters = [(p - g).detach().requires_grad_() for p, g in zip(parameters, gradients)]
        result = {name:value.detach() for name, value in zip(model_source.FAST_NAMES, parameters)}
        result.update({name.replace('_states', '_grad'):torch.zeros_like(value) for name, value in list(result.items())})
        return result

    def gap(a, b):
        assert set(a) == set(b)
        return max(float((a[k] - b[k]).abs().max().detach()) for k in a)

    results = []
    for config in model_source.configs():
        torch.manual_seed(874601)
        model = model_source.MatchedOfficialTTT(config['head_dim'], config['inner_passes'], config['key_basis'], config['rate_mode']).double().eval()
        if config['key_basis'] == 'prior256':
            prior_reference = population.PriorMetaRidge(8)
            assert torch.equal(model.key_prior_bank, prior_reference.bank)
            with torch.no_grad():
                features = torch.stack([torch.as_tensor(base.forward(x.numpy(), b.numpy())) for b in model.key_prior_bank], dim=-1)
                expected_key = model.key_encoder(torch.cat((x[..., None], features), dim=-1))
                assert torch.allclose(model.key_features(x), expected_key, atol=1e-12, rtol=1e-12)
        before = {k:value.detach().clone() for k, value in model.state_dict().items()}
        warm = reference = None
        maximum = 0.
        for count in model_source.STAGES:
            with torch.no_grad():
                cold = model.adapt(x[:, :count], v[:, :count])
                warm = model.adapt(x[:, :count], v[:, :count], warm)
            cold_reference = reference_adapt(model, x[:, :count], v[:, :count])
            reference = reference_adapt(model, x[:, :count], v[:, :count], reference)
            maximum = max(maximum, gap(cold, cold_reference), gap(warm, reference))
            assert torch.isfinite(model.predict(warm, q)).all()
        assert maximum < 1e-7, (config, maximum)
        with torch.no_grad():
            a = model.adapt(x[:1], v[:1]); saved_a = {k:t.clone() for k, t in a.items()}
            model.adapt(x[1:], v[1:]); a_again = model.adapt(x[:1], v[:1])
            assert gap(a, a_again) == 0.
            p = model.predict(a, q[:1]); reverse = model.predict(a, q[:1].flip(1))
            assert torch.allclose(p, reverse.flip(1), atol=1e-12, rtol=1e-12)
            assert gap(a, saved_a) == 0.
            changed = model.adapt(x[:1], v[:1] + .03)
            assert gap(a, changed) > 1e-10
            one_by_one = torch.cat([model.predict(model.adapt(x[i:i+1], v[i:i+1]), q[i:i+1]) for i in range(2)])
            joint = model.predict(model.adapt(x, v), q)
            batch_gap = float((joint - one_by_one).abs().max())
            assert batch_gap < 1e-9, batch_gap
            assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())

        model.train(); model.zero_grad(set_to_none=True)
        loss = model_source.trajectory_loss(model, data)
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        optimizer = torch.optim.Adam(trainable, lr=.001)
        torch.nn.utils.clip_grad_norm_(trainable, 1.)
        optimizer.step()
        assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items())
        model.eval()
        with torch.no_grad():
            updated = model_source.trajectory_predictions(model, x, v, q)
            assert torch.isfinite(updated).all()
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = model_source.MatchedOfficialTTT(config['head_dim'], config['inner_passes'], config['key_basis'], config['rate_mode']).double().eval()
        restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True))
        with torch.no_grad():
            recovered = model_source.trajectory_predictions(restored, x, v, q)
        assert torch.equal(updated, recovered)
        results.append(dict(config=config, gradient_reference_max_error=maximum,
                            batch_vs_individual_max_error=batch_gap,
                            outer_gradients_finite=True, query_read_only=True,
                            task_reset_identical=True, inference_slow_weights_unchanged=True,
                            outer_optimizer_step_finite=True, checkpoint_roundtrip_identical=True,
                            resources=model_source.state_resources(model, a)))

    # Original float32 cold-n24 bridge must remain exactly reproduced. In
    # float64 the new driver deliberately creates rates in model dtype; the old
    # driver created some rate tensors in default float32, so use the independent
    # float64 gradient reference above instead of claiming bitwise equivalence.
    torch.manual_seed(874601)
    original = old.OfficialTTTBridge(16, 24, 1).eval()
    wrapped = model_source.MatchedOfficialTTT(16, 1).eval()
    wrapped.load_state_dict(original.state_dict())
    with torch.no_grad():
        legacy = original.adapt(x.float(), v.float())
        current = wrapped.adapt(x.float(), v.float())
    assert gap(legacy, current) == 0.
    for relative, digest in protocol['source_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    save(OUT / 'checks.json', results)
    summary = dict(passed=True, configurations=len(results), teacher_max_error=teacher_gap,
                   original_float32_cold24_fast_state_max_error=gap(legacy, current),
                   support_prefixes=list(model_source.STAGES),
                   independent_reference='Torch autograd of per-token-weighted half-sum squared LN reconstruction; native factors independently recomputed',
                   formal_training_started=False, optimizer_smoke_steps=len(results), new_task_queries_accessed=False,
                   source_sha256=protocol['source_sha256'],
                   outputs_sha256={name:sha(OUT / name) for name in ('protocol.json', 'checks.json')})
    save(OUT / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    require_finished_deadline()
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        run()
    except Exception:
        save(OUT / 'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
