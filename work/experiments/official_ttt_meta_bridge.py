"""Meta-train the pinned official TTT-MLP update on causal tent episodes.

Context values are visible only through the V view. Query values are absent from
the adaptation and prediction APIs. A separately meta-trained closed-form ridge
feature model receives the same outer-training episodes and is the primary
regression baseline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = ROOT / "work" / "third_party" / "ttt-lm-pytorch"
sys.path.insert(0, str(OFFICIAL_REPO))
import ttt  # type: ignore  # noqa: E402


def compose_torch(x: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    output = x
    for index in range(parameters.shape[-1]):
        threshold = parameters[:, index, None]
        output = torch.where(
            output <= threshold,
            output / threshold,
            (1.0 - output) / (1.0 - threshold),
        )
    return output


def compose_numpy(x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    output = np.asarray(x, dtype=np.float64)
    for threshold in parameters:
        output = np.where(
            output <= threshold,
            output / threshold,
            (1.0 - output) / (1.0 - threshold),
        )
    return output


def sample_training_batch(
    batch_size: int,
    depth: int,
    n_context: int,
    n_query: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    parameters = 0.40 + 0.20 * torch.rand(
        batch_size, depth, generator=generator, device=device
    )
    bins = torch.arange(n_context, device=device, dtype=torch.float32)[None, :]
    context_x = (bins + torch.rand(
        batch_size, n_context, generator=generator, device=device
    )) / n_context
    order = torch.rand(
        batch_size, n_context, generator=generator, device=device
    ).argsort(dim=1)
    context_x = context_x.gather(1, order)
    query_x = torch.rand(
        batch_size, n_query, generator=generator, device=device
    )
    context_y = compose_torch(context_x, parameters)
    query_y = compose_torch(query_x, parameters)
    return context_x, context_y, query_x, query_y


def fixed_episode_batch(
    seed_offset: int,
    episodes: int,
    depth: int,
    n_context: int,
    n_query: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    contexts_x = []
    contexts_y = []
    queries_x = []
    queries_y = []
    seeds = []
    for episode in range(episodes):
        seed = seed_offset + episode
        rng = np.random.default_rng(seed)
        parameters = rng.uniform(0.40, 0.60, size=depth)
        context_x = (np.arange(n_context) + rng.random(n_context)) / n_context
        rng.shuffle(context_x)
        query_x = rng.random(n_query)
        contexts_x.append(context_x)
        contexts_y.append(compose_numpy(context_x, parameters))
        queries_x.append(query_x)
        queries_y.append(compose_numpy(query_x, parameters))
        seeds.append(seed)
    convert = lambda values: torch.as_tensor(  # noqa: E731
        np.stack(values), dtype=torch.float32, device=device
    )
    return convert(contexts_x), convert(contexts_y), convert(queries_x), convert(queries_y), seeds


def scalar_basis(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            x,
            torch.sin(2.0 * math.pi * x),
            torch.cos(2.0 * math.pi * x),
            torch.sin(4.0 * math.pi * x),
            torch.cos(4.0 * math.pi * x),
        ],
        dim=-1,
    )


class OfficialTTTBridge(torch.nn.Module):
    """Causal K/V write and Q read using the official TTTMLP fast update."""

    def __init__(self, head_dim: int, n_context: int, inner_passes: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.n_context = n_context
        self.inner_passes = inner_passes
        config = ttt.TTTConfig(
            hidden_size=head_dim,
            intermediate_size=4 * head_dim,
            num_hidden_layers=1,
            num_attention_heads=1,
            mini_batch_size=n_context,
            ttt_layer_type="mlp",
            share_qk=False,
            use_cache=False,
            scan_checkpoint_group_size=0,
        )
        self.learner = ttt.TTTMLP(config, layer_idx=0)
        used_learner_parameters = {
            "W1", "b1", "W2", "b2", "ttt_norm_weight", "ttt_norm_bias"
        }
        for name, parameter in self.learner.named_parameters():
            parameter.requires_grad_(name in used_learner_parameters)
        self.key_encoder = torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, head_dim),
            torch.nn.LayerNorm(head_dim),
        )
        self.value_encoder = torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, head_dim),
            torch.nn.LayerNorm(head_dim),
        )
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(2 * head_dim, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 1),
        )
        # Official TTT scales the learned rate by head_dim. This logit keeps the
        # same 0..1/head_dim range rather than introducing an unconstrained rate.
        self.inner_lr_logit = torch.nn.Parameter(torch.tensor(0.0))

    def key_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.key_encoder(scalar_basis(x))

    def value_code(self, y: torch.Tensor) -> torch.Tensor:
        features = torch.stack(
            [
                y,
                y.square(),
                torch.sin(math.pi * y),
                torch.cos(math.pi * y),
                torch.ones_like(y),
            ],
            dim=-1,
        )
        return self.value_encoder(features)

    def initial_fast_state(self, batch_size: int) -> dict[str, torch.Tensor]:
        return {
            "W1_states": self.learner.W1.unsqueeze(0).expand(batch_size, -1, -1, -1),
            "b1_states": self.learner.b1.unsqueeze(0).expand(batch_size, -1, -1, -1),
            "W2_states": self.learner.W2.unsqueeze(0).expand(batch_size, -1, -1, -1),
            "b2_states": self.learner.b2.unsqueeze(0).expand(batch_size, -1, -1, -1),
            "W1_grad": torch.zeros(
                batch_size, *self.learner.W1.shape, device=self.learner.W1.device
            ),
            "b1_grad": torch.zeros(
                batch_size, *self.learner.b1.shape, device=self.learner.b1.device
            ),
            "W2_grad": torch.zeros(
                batch_size, *self.learner.W2.shape, device=self.learner.W2.device
            ),
            "b2_grad": torch.zeros(
                batch_size, *self.learner.b2.shape, device=self.learner.b2.device
            ),
        }

    def adapt(self, context_x: torch.Tensor, context_y: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, n_context = context_x.shape
        if n_context != self.n_context:
            raise ValueError(f"expected {self.n_context} context tokens, received {n_context}")
        xk = self.key_features(context_x)
        # The observed value is present only in V, and the official target is V-K.
        xv = xk + self.value_code(context_y)
        xk = xk[:, None, None, :, :]
        xv = xv[:, None, None, :, :]
        token_eta = torch.ones(
            batch_size, 1, 1, n_context, 1, device=context_x.device
        )
        inner_lr = torch.sigmoid(self.inner_lr_logit) / self.head_dim
        ttt_lr_eta = inner_lr * torch.ones(
            batch_size, 1, 1, 1, n_context, device=context_x.device
        )
        eta = token_eta * ttt_lr_eta
        inputs = {
            "XQ": xk,
            "XK": xk,
            "XV": xv,
            "eta": eta,
            "token_eta": token_eta,
            "ttt_lr_eta": ttt_lr_eta,
        }
        fast_state = None
        for _ in range(self.inner_passes):
            _, fast_state = self.learner.ttt(
                inputs,
                mini_batch_size=n_context,
                last_mini_batch_params_dict=fast_state,
                cache_params=None,
            )
        assert fast_state is not None
        return fast_state

    def predict_from_fast(
        self, fast_state: dict[str, torch.Tensor], query_x: torch.Tensor
    ) -> torch.Tensor:
        xq = self.key_features(query_x)
        query = xq[:, None, :, :]
        z1 = query @ fast_state["W1_states"] + fast_state["b1_states"]
        hidden = F.gelu(z1, approximate="tanh")
        z2 = hidden @ fast_state["W2_states"] + fast_state["b2_states"]
        gamma = self.learner.ttt_norm_weight.reshape(1, 1, 1, self.head_dim)
        beta = self.learner.ttt_norm_bias.reshape(1, 1, 1, self.head_dim)
        code = ttt.ln_fwd(z2, gamma, beta).squeeze(1)
        return torch.sigmoid(self.readout(torch.cat([xq, code], dim=-1))).squeeze(-1)

    def forward(
        self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_from_fast(self.adapt(context_x, context_y), query_x)

    def predict_without_adaptation(self, query_x: torch.Tensor) -> torch.Tensor:
        return self.predict_from_fast(self.initial_fast_state(query_x.shape[0]), query_x)

    @property
    def inner_lr(self) -> float:
        return float(torch.sigmoid(self.inner_lr_logit).detach() / self.head_dim)


class MetaRidge(torch.nn.Module):
    """Meta-trained frozen nonlinear features with a closed-form episode head."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        if feature_dim < 2:
            raise ValueError("feature_dim must be at least two")
        self.feature_dim = feature_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(5, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, feature_dim - 1),
            torch.nn.Tanh(),
        )
        self.log_ridge = torch.nn.Parameter(torch.tensor(math.log(1e-3)))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        learned = self.encoder(scalar_basis(x))
        return torch.cat([learned, torch.ones_like(x[..., None])], dim=-1)

    def solve(self, context_x: torch.Tensor, context_y: torch.Tensor) -> torch.Tensor:
        phi = self.features(context_x)
        gram = phi.transpose(-2, -1) @ phi
        ridge = self.log_ridge.exp().clamp(1e-7, 1.0)
        identity = torch.eye(
            self.feature_dim, device=phi.device, dtype=phi.dtype
        )[None, :, :]
        penalty = ridge * identity
        penalty[:, -1, -1] = 0.0
        right = phi.transpose(-2, -1) @ context_y[..., None]
        return torch.linalg.solve(gram + penalty, right)

    def forward(
        self, context_x: torch.Tensor, context_y: torch.Tensor, query_x: torch.Tensor
    ) -> torch.Tensor:
        coefficients = self.solve(context_x, context_y)
        return (self.features(query_x) @ coefficients).squeeze(-1)

    @property
    def ridge(self) -> float:
        return float(self.log_ridge.exp().detach().clamp(1e-7, 1.0))


def train_meta_model(
    name: str,
    model: torch.nn.Module,
    steps: int,
    learning_rate: float,
    batch_size: int,
    depth: int,
    n_context: int,
    train_queries: int,
    train_seed: int,
    validation: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    validation_interval: int,
    device: torch.device,
) -> tuple[list[dict[str, float | int | str]], int, float]:
    generator = torch.Generator(device=device).manual_seed(train_seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-5,
    )
    logs: list[dict[str, float | int | str]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_step = 0
    best_validation = float("inf")
    validation_x, validation_y, validation_qx, validation_qy = validation
    for step in range(1, steps + 1):
        model.train()
        context_x, context_y, query_x, query_y = sample_training_batch(
            batch_size, depth, n_context, train_queries, generator, device
        )
        prediction = model(context_x, context_y, query_x)
        loss = torch.mean((prediction - query_y).square())
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced a non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step == 1 or step % validation_interval == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                validation_prediction = model(validation_x, validation_y, validation_qx)
                validation_loss = float(
                    torch.mean((validation_prediction - validation_qy).square())
                )
            is_best = validation_loss < best_validation
            if is_best:
                best_validation = validation_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            logs.append(
                {
                    "method": name,
                    "step": step,
                    "train_query_mse": float(loss.detach()),
                    "validation_query_mse": validation_loss,
                    "gradient_norm_before_clip": gradient_norm,
                    "selected_as_best": int(is_best),
                }
            )
            print(
                f"{name:18s} step={step:5d} train={float(loss.detach()):.5f} "
                f"validation={validation_loss:.5f} best={best_validation:.5f}"
            )
    model.load_state_dict(best_state)
    return logs, best_step, best_validation


@torch.no_grad()
def predict_in_query_chunks(
    model: OfficialTTTBridge | MetaRidge,
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    query_x: torch.Tensor,
    chunk_size: int,
    no_adaptation: bool = False,
) -> torch.Tensor:
    predictions = []
    if isinstance(model, OfficialTTTBridge):
        fast_state = None if no_adaptation else model.adapt(context_x, context_y)
        if no_adaptation:
            fast_state = model.initial_fast_state(context_x.shape[0])
        assert fast_state is not None
        for start in range(0, query_x.shape[1], chunk_size):
            predictions.append(
                model.predict_from_fast(fast_state, query_x[:, start : start + chunk_size])
            )
    else:
        coefficients = model.solve(context_x, context_y)
        for start in range(0, query_x.shape[1], chunk_size):
            features = model.features(query_x[:, start : start + chunk_size])
            predictions.append((features @ coefficients).squeeze(-1))
    return torch.cat(predictions, dim=1)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    summaries = []
    for method in sorted({str(row["method"]) for row in rows}):
        values = np.asarray(
            [float(row["query_mse"]) for row in rows if row["method"] == method]
        )
        summaries.append(
            {
                "method": method,
                "episodes": values.size,
                "mean_query_mse": float(np.mean(values)),
                "median_query_mse": float(np.median(values)),
                "p90_query_mse": float(np.quantile(values, 0.9)),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-context", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--inner-passes", type=int, default=1)
    parser.add_argument("--ridge-features", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-queries", type=int, default=64)
    parser.add_argument("--ttt-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ridge-learning-rate", type=float, default=2e-3)
    parser.add_argument("--validation-episodes", type=int, default=16)
    parser.add_argument("--validation-queries", type=int, default=512)
    parser.add_argument("--validation-seed-offset", type=int, default=980000)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=64)
    parser.add_argument("--test-queries", type=int, default=4096)
    parser.add_argument("--test-seed-offset", type=int, default=1000000)
    parser.add_argument("--train-seed", type=int, default=20260921)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.train_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.train_seed)

    validation_full = fixed_episode_batch(
        args.validation_seed_offset,
        args.validation_episodes,
        args.depth,
        args.n_context,
        args.validation_queries,
        device,
    )
    validation = validation_full[:4]

    ttt_model = OfficialTTTBridge(args.head_dim, args.n_context, args.inner_passes).to(device)
    ridge_model = MetaRidge(args.ridge_features).to(device)
    ttt_logs, ttt_best_step, ttt_best_validation = train_meta_model(
        "official_ttt_meta",
        ttt_model,
        args.steps,
        args.ttt_learning_rate,
        args.batch_size,
        args.depth,
        args.n_context,
        args.train_queries,
        args.train_seed,
        validation,
        args.validation_interval,
        device,
    )
    ridge_logs, ridge_best_step, ridge_best_validation = train_meta_model(
        "meta_feature_ridge",
        ridge_model,
        args.steps,
        args.ridge_learning_rate,
        args.batch_size,
        args.depth,
        args.n_context,
        args.train_queries,
        args.train_seed,
        validation,
        args.validation_interval,
        device,
    )
    torch.save(ttt_model.state_dict(), args.output_dir / "official_ttt_meta_best.pt")
    torch.save(ridge_model.state_dict(), args.output_dir / "meta_feature_ridge_best.pt")
    write_csv(args.output_dir / "meta_bridge_training.csv", ttt_logs + ridge_logs)

    test_x, test_y, test_qx, test_qy, test_seeds = fixed_episode_batch(
        args.test_seed_offset,
        args.test_episodes,
        args.depth,
        args.n_context,
        args.test_queries,
        device,
    )
    ttt_model.eval()
    ridge_model.eval()
    predictions = {
        "official_ttt_meta": predict_in_query_chunks(
            ttt_model, test_x, test_y, test_qx, args.query_chunk_size
        ),
        "official_ttt_no_adapt": predict_in_query_chunks(
            ttt_model, test_x, test_y, test_qx, args.query_chunk_size, no_adaptation=True
        ),
        "meta_feature_ridge": predict_in_query_chunks(
            ridge_model, test_x, test_y, test_qx, args.query_chunk_size
        ),
    }
    rows: list[dict[str, float | int | str]] = []
    for method, prediction in predictions.items():
        losses = torch.mean((prediction - test_qy).square(), dim=1).cpu().numpy()
        for seed, loss in zip(test_seeds, losses):
            rows.append({"seed": seed, "method": method, "query_mse": float(loss)})
    write_csv(args.output_dir / "meta_bridge_episodes.csv", rows)
    summaries = summarize(rows)
    write_csv(args.output_dir / "meta_bridge_summary.csv", summaries)

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=OFFICIAL_REPO, text=True
    ).strip()
    metadata = vars(args) | {
        "output_dir": str(args.output_dir),
        "device_resolved": str(device),
        "official_ttt_commit": source_commit,
        "official_inner_target": "XV-XK; context V is K plus an encoding of observed y",
        "query_token_value": 0,
        "adaptation_or_prediction_api_accepts_query_targets": False,
        "outer_training_uses_training_episode_query_targets": True,
        "validation_query_targets_used_only_for_checkpoint_selection": True,
        "test_query_targets_used_only_for_final_evaluation": True,
        "ttt_best_step": ttt_best_step,
        "ttt_best_validation_mse": ttt_best_validation,
        "ttt_inner_lr_selected": ttt_model.inner_lr,
        "ttt_episode_fast_parameters": args.head_dim * (8 * args.head_dim + 5),
        "ttt_outer_trainable_parameters": count_trainable_parameters(ttt_model),
        "ridge_best_step": ridge_best_step,
        "ridge_best_validation_mse": ridge_best_validation,
        "ridge_selected_regularization": ridge_model.ridge,
        "ridge_episode_coefficients": args.ridge_features,
        "ridge_outer_trainable_parameters": count_trainable_parameters(ridge_model),
        "warning": "The bridge uses the official fast-update kernel but task-specific learned K/V/Q encoders; it is not a pretrained language model result.",
    }
    with (args.output_dir / "meta_bridge_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
