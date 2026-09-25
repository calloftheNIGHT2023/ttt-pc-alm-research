"""390 official TTT-MLP bridge for the matched shifted-tent population.

Task representations and the support-prefix driver are adapters. The pinned
official TTTMLP.ttt routine performs every fast update. This is a BP baseline, not the
local-credit candidate. No query target is accepted by adapt or predict.
"""
import numpy as np
import torch
import official_ttt_meta_bridge as old
import matched_shifted_meta_models as population

STAGES = population.STAGES
FAST_NAMES = ('W1_states', 'b1_states', 'W2_states', 'b2_states')
PRIMARY = 'official_ttt_native_prior256_32_p1'


def configs():
    return [dict(name=f'official_ttt_native_prior256_{width}_p{passes}', head_dim=width,
                 inner_passes=passes, key_basis='prior256', rate_mode='native', learning_rate=.001)
            for width in (16, 32) for passes in (1, 4)] + [
                dict(name='official_ttt_native_scalar32_p1', head_dim=32,
                     inner_passes=1, key_basis='scalar', rate_mode='native', learning_rate=.001),
                dict(name='official_ttt_legacy_prior256_32_p1', head_dim=32,
                     inner_passes=1, key_basis='prior256', rate_mode='legacy', learning_rate=.001)]


class MatchedOfficialTTT(old.OfficialTTTBridge):
    def __init__(self, head_dim=16, inner_passes=1, key_basis='scalar', rate_mode='legacy'):
        if head_dim < 2 or inner_passes < 1:
            raise ValueError('positive passes and at least two channels required')
        super().__init__(head_dim, max(STAGES), inner_passes)
        if key_basis not in ('scalar', 'prior256'):
            raise ValueError('unknown key basis')
        self.key_basis = key_basis
        if rate_mode not in ('native', 'legacy'):
            raise ValueError('unknown learning-rate mode')
        self.rate_mode = rate_mode
        if rate_mode == 'native':
            self.inner_lr_logit.requires_grad_(False)
            for name in ('learnable_token_idx', 'learnable_ttt_lr_weight', 'learnable_ttt_lr_bias'):
                getattr(self.learner, name).requires_grad_(True)
        if key_basis == 'prior256':
            # Same full-precision prior bank as the matched meta-ridge controls.
            self.register_buffer('key_prior_bank', torch.as_tensor(
                np.random.default_rng(731).uniform(-.12, .12, (256, 4)), dtype=torch.float64))
            self.key_encoder = torch.nn.Sequential(
                torch.nn.Linear(257, 64), torch.nn.GELU(),
                torch.nn.Linear(64, head_dim), torch.nn.LayerNorm(head_dim))

    def key_features(self, x):
        if self.key_basis == 'scalar':
            return super().key_features(x)
        bank = self.key_prior_bank.to(dtype=x.dtype)
        h = x[..., None].expand(*x.shape, len(bank))
        for layer in range(4):
            h = (1 - torch.abs(2 * (h + bank[:, layer]) - 1)).clamp_min(0)
        return self.key_encoder(torch.cat((x[..., None], h), dim=-1))

    def initial_fast_state(self, batch_size):
        state = {}
        for name in FAST_NAMES:
            value = getattr(self.learner, name.removesuffix('_states'))
            state[name] = value.unsqueeze(0).expand(batch_size, -1, -1, -1)
            state[name.replace('_states', '_grad')] = torch.zeros_like(state[name])
        return state

    def adapt(self, context_x, context_y, state=None):
        if context_x.ndim != 2 or context_x.shape != context_y.shape:
            raise ValueError('matching [batch, support] tensors required')
        if context_x.shape[1] not in STAGES or not context_x.shape[0]:
            raise ValueError(f'support prefixes must be one of {STAGES}')
        parameter = self.learner.W1
        for value in (context_x, context_y):
            if value.device != parameter.device or value.dtype != parameter.dtype:
                raise ValueError('input dtype/device must match model')
        batch, count = context_x.shape
        key = self.key_features(context_x)
        value = key + self.value_code(context_y)
        if self.rate_mode == 'native':
            token_eta, ttt_lr_eta = self.learner.get_eta(key[:, None], 0, count)
        else:
            token_eta = context_x.new_ones((batch, 1, 1, count, 1))
            lr = torch.sigmoid(self.inner_lr_logit) / self.head_dim
            ttt_lr_eta = lr * context_x.new_ones((batch, 1, 1, 1, count))
        key = key[:, None, None]
        value = value[:, None, None]
        inputs = dict(XQ=key, XK=key, XV=value, token_eta=token_eta,
                      ttt_lr_eta=ttt_lr_eta, eta=token_eta * ttt_lr_eta)
        current = state
        for _ in range(self.inner_passes):
            # No cache: the official implementation explicitly chooses its dual
            # form even when this prefix is shorter than configured max context.
            _, current = self.learner.ttt(
                inputs, mini_batch_size=count,
                last_mini_batch_params_dict=current, cache_params=None)
        return current

    def predict(self, state, query_x):
        return self.predict_from_fast(state, query_x)


def trajectory_predictions(model, x, v, q, warm=True):
    state = None
    predictions = []
    for count in STAGES:
        state = model.adapt(x[:, :count], v[:, :count], state if warm else None)
        predictions.append(model.predict(state, q))
    return torch.stack(predictions)


def trajectory_loss(model, batch, warm=True):
    x, v, q, target = batch
    return (trajectory_predictions(model, x, v, q, warm) - target[None]).square().mean()


def state_resources(model, state):
    return dict(
        shared_model_bytes=sum(p.numel() * p.element_size() for p in model.state_dict().values()),
        fast_parameter_bytes=sum(state[n].numel() * state[n].element_size() for n in FAST_NAMES),
        gradient_carry_bytes=sum(v.numel() * v.element_size() for n, v in state.items() if n.endswith('_grad')),
        fast_state_bytes=sum(v.numel() * v.element_size() for v in state.values()),
        scope='named tensors only; transient tensors and runtime memory measured separately',
        rate_mode=model.rate_mode,
        inner_update='pinned official TTTMLP.ttt; analytic global chain derivatives',
        query_targets_accessed=False)
