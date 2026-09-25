"""Pre-registered real-data domain-shift experiment on sklearn digits.

The online candidate estimates a six-parameter inverse affine transform in front
of a frozen convolutional classifier. Query labels are never used by fitting,
risk weighting, early stopping, or hyper-parameter selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


SPLIT_SEED = 20260920
TRAIN_SEED = 20260921
VALIDATION_SEED_OFFSET = 4_100_000
CONFIRMATION_SEED_OFFSET = 4_200_000
N_VALIDATION_EPISODES = 32
N_CONFIRMATION_EPISODES = 128
IMAGE_SIZE = 16
N_CLASSES = 10
FEATURE_DIM = 64
CONTEXT_ADAPT_PER_CLASS = 2
CONTEXT_CALIBRATION_PER_CLASS = 1
QUERY_PER_CLASS = 10
INTERNAL_STAGE_STEPS = 80
SHALLOW_STEPS = 160


@dataclass(frozen=True)
class ShiftRanges:
    angle_deg: float = 35.0
    translation: float = 0.20
    log_scale: float = 0.18
    shear: float = 0.18


@dataclass(frozen=True)
class SelectedConfig:
    internal_lr: float
    internal_regularization: float
    ridge_alpha: float
    rbf_alpha: float
    rbf_gamma_multiplier: float
    shallow_lr: float
    shallow_weight_decay: float
    tangent_alpha: float
    geometric_bank_size: int
    geometric_bank_alpha: float
    internal_stage_steps: int = INTERNAL_STAGE_STEPS
    shallow_steps: int = SHALLOW_STEPS
    shallow_width: int = 32


class DigitBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.project = nn.Linear(64 * 4 * 4, FEATURE_DIM)
        self.classifier = nn.Linear(FEATURE_DIM, N_CLASSES)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).flatten(1)
        return F.relu(self.project(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResidualShallowHead(nn.Module):
    def __init__(self, feature_dim: int, width: int) -> None:
        super().__init__()
        self.first = nn.Linear(feature_dim, width)
        self.second = nn.Linear(width, N_CLASSES)
        nn.init.kaiming_uniform_(self.first.weight, a=math.sqrt(5))
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)

    def forward(self, features: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        return base_logits + self.second(F.relu(self.first(features)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_arrays() -> tuple[torch.Tensor, torch.Tensor]:
    digits = load_digits()
    images = torch.as_tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
    images = F.interpolate(
        images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    labels = torch.as_tensor(digits.target, dtype=torch.long)
    return images, labels


def fixed_split(labels: torch.Tensor) -> dict[str, np.ndarray]:
    indices = np.arange(labels.numel())
    train, remainder = train_test_split(
        indices,
        train_size=0.60,
        stratify=labels.numpy(),
        random_state=SPLIT_SEED,
    )
    validation, confirmation = train_test_split(
        remainder,
        train_size=0.50,
        stratify=labels.numpy()[remainder],
        random_state=SPLIT_SEED + 1,
    )
    return {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "confirmation": np.sort(confirmation),
    }


def physical_to_theta(parameters: torch.Tensor) -> torch.Tensor:
    """Convert angle, tx, ty, log-scale, shear-x/y to affine_grid theta."""
    angle, tx, ty, log_scale, shear_x, shear_y = parameters.unbind(-1)
    scale = torch.exp(log_scale)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    base = torch.stack(
        [scale, shear_x, shear_y, scale], dim=-1
    ).reshape(parameters.shape[:-1] + (2, 2))
    rotation = torch.stack(
        [cosine, -sine, sine, cosine], dim=-1
    ).reshape(parameters.shape[:-1] + (2, 2))
    linear = rotation @ base
    translation = torch.stack([tx, ty], dim=-1).unsqueeze(-1)
    return torch.cat([linear, translation], dim=-1)


def inverse_theta(theta: torch.Tensor) -> torch.Tensor:
    leading = theta.shape[:-2]
    bottom = torch.zeros(leading + (1, 3), dtype=theta.dtype, device=theta.device)
    bottom[..., 0, 2] = 1.0
    homogeneous = torch.cat([theta, bottom], dim=-2)
    return torch.linalg.inv(homogeneous)[..., :2, :]


def correction_theta(raw: torch.Tensor) -> torch.Tensor:
    bounded = torch.tanh(raw)
    row_1 = torch.stack(
        [1.0 + 0.55 * bounded[0], 0.55 * bounded[1], 0.45 * bounded[2]]
    )
    row_2 = torch.stack(
        [0.55 * bounded[3], 1.0 + 0.55 * bounded[4], 0.45 * bounded[5]]
    )
    return torch.stack([row_1, row_2])


def warp(images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    if theta.ndim == 2:
        theta = theta.unsqueeze(0).expand(images.shape[0], -1, -1)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def sample_physical_parameters(
    rng: np.random.Generator, ranges: ShiftRanges, count: int = 1
) -> torch.Tensor:
    values = np.column_stack(
        [
            rng.uniform(-ranges.angle_deg, ranges.angle_deg, count) * math.pi / 180.0,
            rng.uniform(-ranges.translation, ranges.translation, count),
            rng.uniform(-ranges.translation, ranges.translation, count),
            rng.uniform(-ranges.log_scale, ranges.log_scale, count),
            rng.uniform(-ranges.shear, ranges.shear, count),
            rng.uniform(-ranges.shear, ranges.shear, count),
        ]
    )
    return torch.as_tensor(values, dtype=torch.float32)


def random_training_theta(batch_size: int, device: torch.device) -> torch.Tensor:
    ranges = torch.tensor(
        [20.0 * math.pi / 180.0, 0.10, 0.10, 0.10, 0.10, 0.10],
        dtype=torch.float32,
        device=device,
    )
    parameters = (2.0 * torch.rand(batch_size, 6, device=device) - 1.0) * ranges
    identity_mask = torch.rand(batch_size, device=device) < 0.20
    parameters[identity_mask] = 0.0
    return physical_to_theta(parameters)


@torch.no_grad()
def batched_features_and_logits(
    model: DigitBackbone, images: torch.Tensor, batch_size: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_parts = []
    logit_parts = []
    for start in range(0, images.shape[0], batch_size):
        features = model.features(images[start : start + batch_size])
        feature_parts.append(features)
        logit_parts.append(model.classifier(features))
    return torch.cat(feature_parts), torch.cat(logit_parts)


def probabilities(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


def project_to_probability_simplex(scores: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of every row onto the probability simplex.

    Ridge is trained directly against one-hot probability vectors. Projecting
    its unconstrained output is therefore the metric-compatible conversion:
    because every target lies in the simplex, projection cannot increase its
    squared distance to any such target.
    """
    sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_scores, dim=-1) - 1.0
    ranks = torch.arange(
        1, scores.shape[-1] + 1, dtype=scores.dtype, device=scores.device
    )
    positive = sorted_scores - cumulative / ranks > 0.0
    rho = positive.sum(dim=-1).clamp_min(1) - 1
    threshold = cumulative.gather(-1, rho.unsqueeze(-1)).squeeze(-1) / (
        rho.to(scores.dtype) + 1.0
    )
    return (scores - threshold.unsqueeze(-1)).clamp_min(0.0)


def metric_values(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    one_hot = F.one_hot(labels, N_CLASSES).to(probs.dtype)
    brier = (probs - one_hot).square().sum(dim=-1).mean()
    error = (probs.argmax(dim=-1) != labels).to(torch.float32).mean()
    nll = -torch.log(probs[torch.arange(labels.numel(), device=labels.device), labels].clamp_min(1e-9)).mean()
    return {
        "query_brier": float(brier.detach().cpu()),
        "query_error": float(error.detach().cpu()),
        "query_nll": float(nll.detach().cpu()),
    }


def brier_value(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return metric_values(probs, labels)["query_brier"]


def ridge_probabilities(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    x_mean = train_features.mean(dim=0, keepdim=True)
    y = F.one_hot(train_labels, N_CLASSES).to(train_features.dtype)
    y_mean = y.mean(dim=0, keepdim=True)
    centered_x = train_features - x_mean
    centered_y = y - y_mean
    gram = centered_x @ centered_x.T
    dual = torch.linalg.solve(
        gram + alpha * torch.eye(gram.shape[0], device=gram.device), centered_y
    )
    scores = (test_features - x_mean) @ centered_x.T @ dual + y_mean
    return project_to_probability_simplex(scores)


def rbf_probabilities(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    alpha: float,
    gamma_multiplier: float,
) -> torch.Tensor:
    train_features = F.normalize(train_features, dim=-1)
    test_features = F.normalize(test_features, dim=-1)
    train_distances = torch.cdist(train_features, train_features).square()
    nonzero = train_distances[train_distances > 1e-10]
    median = nonzero.median() if nonzero.numel() else torch.tensor(1.0, device=train_features.device)
    gamma = gamma_multiplier / median.clamp_min(1e-6)
    kernel = torch.exp(-gamma * train_distances)
    y = F.one_hot(train_labels, N_CLASSES).to(train_features.dtype)
    dual = torch.linalg.solve(
        kernel + alpha * torch.eye(kernel.shape[0], device=kernel.device), y
    )
    test_kernel = torch.exp(-gamma * torch.cdist(test_features, train_features).square())
    return project_to_probability_simplex(test_kernel @ dual)


def fit_internal(
    model: DigitBackbone,
    images: torch.Tensor,
    labels: torch.Tensor,
    learning_rate: float,
    regularization: float,
    steps: int,
    raw: torch.Tensor | None = None,
    optimizer_state: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    raw_parameter = nn.Parameter(
        torch.zeros(6, device=images.device) if raw is None else raw.detach().clone()
    )
    optimizer = torch.optim.Adam([raw_parameter], lr=learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        theta = correction_theta(raw_parameter)
        logits = model(warp(images, theta))
        determinant = torch.linalg.det(theta[:, :2])
        barrier = F.relu(0.20 - determinant).square()
        loss = (
            F.cross_entropy(logits, labels)
            + regularization * raw_parameter.square().mean()
            + 10.0 * barrier
        )
        loss.backward()
        optimizer.step()
    return raw_parameter.detach(), optimizer.state_dict()


@torch.no_grad()
def internal_probabilities(
    model: DigitBackbone, images: torch.Tensor, raw: torch.Tensor
) -> torch.Tensor:
    return probabilities(model(warp(images, correction_theta(raw))))


def fit_shallow_head(
    train_features: torch.Tensor,
    train_logits: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_logits: torch.Tensor,
    learning_rate: float,
    weight_decay: float,
    width: int,
    steps: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    head = ResidualShallowHead(train_features.shape[1], width).to(train_features.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_features, train_logits)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return probabilities(head(test_features, test_logits))


def tangent_jacobian(
    model: DigitBackbone, images: torch.Tensor, difference: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference probability Jacobian at the identity correction."""
    with torch.no_grad():
        raw_zero = torch.zeros(6, device=images.device)
        base = internal_probabilities(model, images, raw_zero)
        columns = []
        for parameter in range(6):
            plus = raw_zero.clone()
            minus = raw_zero.clone()
            plus[parameter] = difference
            minus[parameter] = -difference
            derivative = (
                internal_probabilities(model, images, plus)
                - internal_probabilities(model, images, minus)
            ) / (2.0 * difference)
            columns.append(derivative)
    return base, torch.stack(columns, dim=-1)


def tangent_raw_solution(
    base_probabilities: torch.Tensor,
    jacobian: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    design = jacobian.reshape(-1, 6)
    target = (F.one_hot(labels, N_CLASSES).to(base_probabilities.dtype) - base_probabilities).reshape(-1)
    normal = design.T @ design + alpha * torch.eye(6, device=design.device)
    raw = torch.linalg.solve(normal, design.T @ target)
    return raw.clamp(-2.0, 2.0)


def fixed_geometric_bank(size: int, device: torch.device) -> torch.Tensor:
    if size < 1:
        raise ValueError("bank must include at least the identity")
    generator = np.random.default_rng(20260922)
    sampled = sample_physical_parameters(generator, ShiftRanges(), size - 1)
    correction = inverse_theta(physical_to_theta(sampled))
    identity = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    return torch.cat([identity, correction], dim=0).to(device)


@torch.no_grad()
def bank_features(
    model: DigitBackbone, images: torch.Tensor, bank: torch.Tensor
) -> torch.Tensor:
    count = images.shape[0]
    repeated = images.unsqueeze(0).expand(bank.shape[0], -1, -1, -1, -1)
    flattened = repeated.reshape(-1, *images.shape[1:])
    theta = bank[:, None].expand(-1, count, -1, -1).reshape(-1, 2, 3)
    transformed = warp(flattened, theta)
    features, _ = batched_features_and_logits(model, transformed)
    return features.reshape(bank.shape[0], count, -1).permute(1, 0, 2).reshape(count, -1)


def episode_tensors(
    images: torch.Tensor,
    labels: torch.Tensor,
    split_indices: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor | np.ndarray]:
    rng = np.random.default_rng(seed)
    adapt_indices: list[int] = []
    calibration_indices: list[int] = []
    query_indices: list[int] = []
    labels_numpy = labels.numpy()
    for digit in range(N_CLASSES):
        candidates = split_indices[labels_numpy[split_indices] == digit]
        required = CONTEXT_ADAPT_PER_CLASS + CONTEXT_CALIBRATION_PER_CLASS + QUERY_PER_CLASS
        selected = rng.choice(candidates, size=required, replace=False)
        adapt_indices.extend(selected[:CONTEXT_ADAPT_PER_CLASS])
        calibration_indices.append(selected[CONTEXT_ADAPT_PER_CLASS])
        query_indices.extend(selected[CONTEXT_ADAPT_PER_CLASS + CONTEXT_CALIBRATION_PER_CLASS :])
    adapt_indices = np.asarray(adapt_indices)
    calibration_indices = np.asarray(calibration_indices)
    query_indices = np.asarray(query_indices)
    context_indices = np.concatenate([adapt_indices, calibration_indices])
    corruption_parameters = sample_physical_parameters(rng, ShiftRanges(), 1)
    corruption_theta = physical_to_theta(corruption_parameters).to(device)[0]
    all_indices = np.concatenate([context_indices, query_indices])
    clean = images[all_indices].to(device)
    corrupted = warp(clean, corruption_theta)
    n_context = context_indices.size
    return {
        "adapt_images": corrupted[: adapt_indices.size],
        "adapt_labels": labels[adapt_indices].to(device),
        "calibration_images": corrupted[adapt_indices.size:n_context],
        "calibration_labels": labels[calibration_indices].to(device),
        "context_images": corrupted[:n_context],
        "context_labels": labels[context_indices].to(device),
        "query_images": corrupted[n_context:],
        "query_labels": labels[query_indices].to(device),
        "corruption_parameters": corruption_parameters.numpy()[0],
        "corruption_theta": corruption_theta.detach().cpu().numpy(),
        "context_indices": context_indices,
        "query_indices": query_indices,
    }


def load_model_and_data(
    checkpoint_dir: Path, device: torch.device
) -> tuple[DigitBackbone, torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
    images, labels = load_arrays()
    split_file = checkpoint_dir / "split_indices.npz"
    checkpoint = checkpoint_dir / "backbone.pt"
    if not split_file.exists() or not checkpoint.exists():
        raise FileNotFoundError("run the train command before validation or confirmation")
    loaded = np.load(split_file)
    splits = {name: loaded[name] for name in ("train", "validation", "confirmation")}
    expected = fixed_split(labels)
    if any(not np.array_equal(splits[name], expected[name]) for name in expected):
        raise RuntimeError("saved split does not match the pre-registered deterministic split")
    model = DigitBackbone().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, images, labels, splits


def train_backbone(args: argparse.Namespace) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    set_seed(TRAIN_SEED)
    device = torch.device(args.device)
    images, labels = load_arrays()
    splits = fixed_split(labels)
    np.savez(output / "split_indices.npz", **splits)
    dataset = TensorDataset(images[splits["train"]], labels[splits["train"]])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(TRAIN_SEED),
    )
    model = DigitBackbone().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history: list[dict] = []
    validation_images = images[splits["validation"]].to(device)
    validation_labels = labels[splits["validation"]].to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            augmented = warp(batch_images, random_training_theta(batch_images.shape[0], device))
            optimizer.zero_grad(set_to_none=True)
            logits = model(augmented)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch_images.shape[0]
            total_correct += int((logits.argmax(-1) == batch_labels).sum())
            total_seen += batch_images.shape[0]
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_images)
            validation_accuracy = float(
                (validation_logits.argmax(-1) == validation_labels).to(torch.float32).mean()
            )
        row = {
            "epoch": epoch,
            "train_augmented_loss": total_loss / total_seen,
            "train_augmented_accuracy": total_correct / total_seen,
            "validation_clean_accuracy": validation_accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(row, flush=True)
    torch.save(model.state_dict(), output / "backbone.pt")
    write_csv(output / "train_history.csv", history)
    metadata = {
        "split_seed": SPLIT_SEED,
        "train_seed": TRAIN_SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "training_augmentation": {
            "angle_degrees": 20.0,
            "translation": 0.10,
            "log_scale": 0.10,
            "shear": 0.10,
            "identity_probability": 0.20,
        },
        "split_sizes": {key: int(value.size) for key, value in splits.items()},
        "checkpoint_sha256": sha256(output / "backbone.pt"),
        "confirmation_images_evaluated_during_training": False,
    }
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def validation_rows_for_episode(
    model: DigitBackbone,
    episode: dict,
    seed: int,
    bank_25: torch.Tensor,
) -> list[dict]:
    context_images = episode["context_images"]
    context_labels = episode["context_labels"]
    query_images = episode["query_images"]
    query_labels = episode["query_labels"]
    adapt_images = episode["adapt_images"]
    adapt_labels = episode["adapt_labels"]
    context_features, context_logits = batched_features_and_logits(model, context_images)
    query_features, query_logits = batched_features_and_logits(model, query_images)
    rows: list[dict] = []

    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        probs = ridge_probabilities(context_features, context_labels, query_features, alpha)
        rows.append({"seed": seed, "family": "ridge", "config": json.dumps({"alpha": alpha}, sort_keys=True), "query_brier": brier_value(probs, query_labels)})
        for gamma in (0.25, 1.0, 4.0):
            probs = rbf_probabilities(context_features, context_labels, query_features, alpha, gamma)
            rows.append({"seed": seed, "family": "rbf", "config": json.dumps({"alpha": alpha, "gamma_multiplier": gamma}, sort_keys=True), "query_brier": brier_value(probs, query_labels)})

    for learning_rate in (0.02, 0.05, 0.10):
        for regularization in (1e-4, 1e-3):
            raw, state = fit_internal(model, adapt_images, adapt_labels, learning_rate, regularization, INTERNAL_STAGE_STEPS)
            raw, _ = fit_internal(model, context_images, context_labels, learning_rate, regularization, INTERNAL_STAGE_STEPS, raw, state)
            probs = internal_probabilities(model, query_images, raw)
            config = {"learning_rate": learning_rate, "regularization": regularization}
            rows.append({"seed": seed, "family": "internal", "config": json.dumps(config, sort_keys=True), "query_brier": brier_value(probs, query_labels)})

    for learning_rate in (0.003, 0.01, 0.03):
        for weight_decay in (1e-4, 1e-3):
            probs = fit_shallow_head(
                context_features,
                context_logits,
                context_labels,
                query_features,
                query_logits,
                learning_rate,
                weight_decay,
                32,
                SHALLOW_STEPS,
                seed,
            )
            config = {"learning_rate": learning_rate, "weight_decay": weight_decay}
            rows.append({"seed": seed, "family": "shallow", "config": json.dumps(config, sort_keys=True), "query_brier": brier_value(probs, query_labels)})

    context_bank = bank_features(model, context_images, bank_25)
    query_bank = bank_features(model, query_images, bank_25)
    for size in (13, 25):
        context_view = context_bank[:, : size * FEATURE_DIM]
        query_view = query_bank[:, : size * FEATURE_DIM]
        for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            probs = ridge_probabilities(context_view, context_labels, query_view, alpha)
            config = {"size": size, "alpha": alpha}
            rows.append({"seed": seed, "family": "geometric_bank", "config": json.dumps(config, sort_keys=True), "query_brier": brier_value(probs, query_labels)})

    base, jacobian = tangent_jacobian(model, context_images)
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        raw = tangent_raw_solution(base, jacobian, context_labels, alpha)
        probs = internal_probabilities(model, query_images, raw)
        rows.append({"seed": seed, "family": "tangent", "config": json.dumps({"alpha": alpha}, sort_keys=True), "query_brier": brier_value(probs, query_labels)})
    return rows


def validate_hyperparameters(args: argparse.Namespace) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, images, labels, splits = load_model_and_data(args.checkpoint_dir, device)
    bank = fixed_geometric_bank(25, device)
    rows: list[dict] = []
    for episode_index in range(args.episodes):
        seed = VALIDATION_SEED_OFFSET + episode_index
        episode = episode_tensors(images, labels, splits["validation"], seed, device)
        rows.extend(validation_rows_for_episode(model, episode, seed, bank))
        print(f"validation episode {episode_index + 1:3d}/{args.episodes}", flush=True)
    write_csv(output / "validation_grid.csv", rows)
    summaries: list[dict] = []
    for family in sorted({row["family"] for row in rows}):
        configs = sorted({row["config"] for row in rows if row["family"] == family})
        for config in configs:
            values = np.asarray([float(row["query_brier"]) for row in rows if row["family"] == family and row["config"] == config])
            summaries.append({
                "family": family,
                "config": config,
                "episodes": values.size,
                "mean_query_brier": float(values.mean()),
                "median_query_brier": float(np.median(values)),
            })
    write_csv(output / "validation_summary.csv", summaries)
    best = {}
    for family in sorted({row["family"] for row in summaries}):
        candidates = [row for row in summaries if row["family"] == family]
        winner = min(candidates, key=lambda row: (float(row["mean_query_brier"]), row["config"]))
        best[family] = json.loads(winner["config"])
    selected = SelectedConfig(
        internal_lr=best["internal"]["learning_rate"],
        internal_regularization=best["internal"]["regularization"],
        ridge_alpha=best["ridge"]["alpha"],
        rbf_alpha=best["rbf"]["alpha"],
        rbf_gamma_multiplier=best["rbf"]["gamma_multiplier"],
        shallow_lr=best["shallow"]["learning_rate"],
        shallow_weight_decay=best["shallow"]["weight_decay"],
        tangent_alpha=best["tangent"]["alpha"],
        geometric_bank_size=best["geometric_bank"]["size"],
        geometric_bank_alpha=best["geometric_bank"]["alpha"],
    )
    selection = {
        "selected": asdict(selected),
        "validation_seed_range": [VALIDATION_SEED_OFFSET, VALIDATION_SEED_OFFSET + args.episodes - 1],
        "selection_metric": "mean_query_brier",
        "confirmation_split_evaluated": False,
        "checkpoint_sha256": sha256(args.checkpoint_dir / "backbone.pt"),
    }
    (output / "selected_config.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)


def evaluate_confirmation_episode(
    model: DigitBackbone,
    episode: dict,
    seed: int,
    config: SelectedConfig,
    bank: torch.Tensor,
) -> tuple[list[dict], dict]:
    adapt_images = episode["adapt_images"]
    adapt_labels = episode["adapt_labels"]
    calibration_images = episode["calibration_images"]
    calibration_labels = episode["calibration_labels"]
    context_images = episode["context_images"]
    context_labels = episode["context_labels"]
    query_images = episode["query_images"]
    query_labels = episode["query_labels"]
    context_features, context_logits = batched_features_and_logits(model, context_images)
    query_features, query_logits = batched_features_and_logits(model, query_images)
    adapt_features = context_features[: adapt_images.shape[0]]
    calibration_features = context_features[adapt_images.shape[0] :]
    timings: dict[str, float] = {}

    start = time.perf_counter()
    with torch.no_grad():
        no_adapt = probabilities(query_logits)
    timings["no_adaptation"] = time.perf_counter() - start

    start = time.perf_counter()
    ridge_full = ridge_probabilities(context_features, context_labels, query_features, config.ridge_alpha)
    ridge_calibration = ridge_probabilities(adapt_features, adapt_labels, calibration_features, config.ridge_alpha)
    timings["frozen_feature_ridge"] = time.perf_counter() - start

    start = time.perf_counter()
    rbf = rbf_probabilities(context_features, context_labels, query_features, config.rbf_alpha, config.rbf_gamma_multiplier)
    timings["rbf_kernel_ridge"] = time.perf_counter() - start

    start = time.perf_counter()
    shallow = fit_shallow_head(
        context_features,
        context_logits,
        context_labels,
        query_features,
        query_logits,
        config.shallow_lr,
        config.shallow_weight_decay,
        config.shallow_width,
        config.shallow_steps,
        seed,
    )
    timings["shallow_residual_head"] = time.perf_counter() - start

    start = time.perf_counter()
    context_bank = bank_features(model, context_images, bank)
    query_bank = bank_features(model, query_images, bank)
    geometric_bank = ridge_probabilities(context_bank, context_labels, query_bank, config.geometric_bank_alpha)
    timings["geometric_bank_ridge"] = time.perf_counter() - start

    start = time.perf_counter()
    tangent_base, tangent_jac = tangent_jacobian(model, context_images)
    tangent_raw = tangent_raw_solution(tangent_base, tangent_jac, context_labels, config.tangent_alpha)
    tangent = internal_probabilities(model, query_images, tangent_raw)
    timings["closed_form_tangent"] = time.perf_counter() - start

    start = time.perf_counter()
    stage_raw, stage_state = fit_internal(
        model,
        adapt_images,
        adapt_labels,
        config.internal_lr,
        config.internal_regularization,
        config.internal_stage_steps,
    )
    internal_calibration = internal_probabilities(model, calibration_images, stage_raw)
    internal_risk = brier_value(internal_calibration, calibration_labels)
    ridge_risk = brier_value(ridge_calibration, calibration_labels)
    epsilon = 1e-6
    internal_weight = (ridge_risk + epsilon) / (internal_risk + ridge_risk + 2.0 * epsilon)
    warm_raw, _ = fit_internal(
        model,
        context_images,
        context_labels,
        config.internal_lr,
        config.internal_regularization,
        config.internal_stage_steps,
        stage_raw,
        stage_state,
    )
    internal_warm = internal_probabilities(model, query_images, warm_raw)
    risk_blend = internal_weight * internal_warm + (1.0 - internal_weight) * ridge_full
    timings["internal_two_stage"] = time.perf_counter() - start
    timings["context_risk_blend"] = timings["internal_two_stage"] + timings["frozen_feature_ridge"]

    start = time.perf_counter()
    full_raw, _ = fit_internal(
        model,
        context_images,
        context_labels,
        config.internal_lr,
        config.internal_regularization,
        2 * config.internal_stage_steps,
    )
    internal_full = internal_probabilities(model, query_images, full_raw)
    timings["internal_full_equal_compute"] = time.perf_counter() - start

    predictions = {
        "no_adaptation": no_adapt,
        "frozen_feature_ridge": ridge_full,
        "rbf_kernel_ridge": rbf,
        "shallow_residual_head": shallow,
        "geometric_bank_ridge": geometric_bank,
        "closed_form_tangent": tangent,
        "internal_two_stage": internal_warm,
        "context_risk_blend": risk_blend,
        "internal_full_equal_compute": internal_full,
    }
    rows = []
    physical = episode["corruption_parameters"]
    for method, prediction in predictions.items():
        rows.append({
            "seed": seed,
            "method": method,
            **metric_values(prediction, query_labels),
            "wall_time_seconds": timings[method],
            "internal_calibration_brier": internal_risk,
            "ridge_calibration_brier": ridge_risk,
            "internal_weight": internal_weight,
            "angle_degrees": float(physical[0] * 180.0 / math.pi),
            "translation_x": float(physical[1]),
            "translation_y": float(physical[2]),
            "log_scale": float(physical[3]),
            "shear_x": float(physical[4]),
            "shear_y": float(physical[5]),
        })
    diagnostics = {
        "seed": seed,
        "context_query_disjoint": not bool(set(episode["context_indices"]) & set(episode["query_indices"])),
        "query_targets_used_for_adaptation": False,
        "warm_raw": warm_raw.cpu().tolist(),
        "full_raw": full_raw.cpu().tolist(),
        "tangent_raw": tangent_raw.cpu().tolist(),
    }
    return rows, diagnostics


def summarize_confirmation(rows: list[dict], output: Path) -> dict:
    methods = sorted({row["method"] for row in rows})
    summaries = []
    arrays: dict[str, np.ndarray] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        brier = np.asarray([float(row["query_brier"]) for row in method_rows])
        arrays[method] = brier
        tail_cutoff = np.quantile(brier, 0.90)
        summaries.append({
            "method": method,
            "episodes": brier.size,
            "mean_query_brier": float(brier.mean()),
            "median_query_brier": float(np.median(brier)),
            "p90_query_brier": float(tail_cutoff),
            "cvar90_query_brier": float(brier[brier >= tail_cutoff].mean()),
            "mean_query_error": float(np.mean([float(row["query_error"]) for row in method_rows])),
            "mean_query_nll": float(np.mean([float(row["query_nll"]) for row in method_rows])),
            "mean_wall_time_seconds": float(np.mean([float(row["wall_time_seconds"]) for row in method_rows])),
        })
    write_csv(output / "confirmation_summary.csv", summaries)
    candidate = arrays["context_risk_blend"]
    rng = np.random.default_rng(20260923)
    comparisons = []
    for baseline in methods:
        if baseline == "context_risk_blend":
            continue
        difference = candidate - arrays[baseline]
        bootstrap = np.empty(20_000)
        for index in range(bootstrap.size):
            sample = rng.integers(0, difference.size, difference.size)
            bootstrap[index] = difference[sample].mean()
        try:
            p_value = float(wilcoxon(candidate, arrays[baseline], alternative="less").pvalue)
        except ValueError:
            p_value = 1.0
        comparisons.append({
            "candidate": "context_risk_blend",
            "baseline": baseline,
            "episodes": difference.size,
            "candidate_win_rate": float(np.mean(difference < 0.0)),
            "mean_candidate_minus_baseline": float(difference.mean()),
            "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
            "wilcoxon_one_sided_p_candidate_less": p_value,
        })
    write_csv(output / "confirmation_paired_comparisons.csv", comparisons)
    closed_form = [
        "frozen_feature_ridge",
        "rbf_kernel_ridge",
        "geometric_bank_ridge",
        "closed_form_tangent",
    ]
    mean_by_method = {row["method"]: row["mean_query_brier"] for row in summaries}
    strongest = min(closed_form, key=lambda method: mean_by_method[method])
    strongest_comparison = next(row for row in comparisons if row["baseline"] == strongest)
    p90_by_method = {row["method"]: row["p90_query_brier"] for row in summaries}
    checks = {
        "candidate_below_every_closed_form_mean": all(mean_by_method["context_risk_blend"] < mean_by_method[method] for method in closed_form),
        "strongest_closed_form": strongest,
        "strongest_closed_form_bootstrap_ci_below_zero": strongest_comparison["bootstrap_95_high"] < 0.0,
        "strongest_closed_form_one_sided_p_below_0_05": strongest_comparison["wilcoxon_one_sided_p_candidate_less"] < 0.05,
        "candidate_below_shallow_mean": mean_by_method["context_risk_blend"] < mean_by_method["shallow_residual_head"],
        "candidate_p90_not_worse_than_shallow": p90_by_method["context_risk_blend"] <= p90_by_method["shallow_residual_head"],
        "internal_below_no_adaptation_mean": mean_by_method["internal_two_stage"] < mean_by_method["no_adaptation"],
    }
    checks["all_preregistered_success_conditions"] = all(value for key, value in checks.items() if key not in {"strongest_closed_form"})
    (output / "confirmation_decision.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return {"summaries": summaries, "comparisons": comparisons, "decision": checks}


def run_confirmation(args: argparse.Namespace) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    selection_payload = json.loads(args.selected_config.read_text(encoding="utf-8"))
    config = SelectedConfig(**selection_payload["selected"])
    expected_hash = selection_payload["checkpoint_sha256"]
    observed_hash = sha256(args.checkpoint_dir / "backbone.pt")
    if expected_hash != observed_hash:
        raise RuntimeError("backbone checkpoint changed after validation selection")
    model, images, labels, splits = load_model_and_data(args.checkpoint_dir, device)
    bank = fixed_geometric_bank(config.geometric_bank_size, device)
    rows: list[dict] = []
    diagnostics: list[dict] = []
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for episode_index in range(args.episodes):
        seed = CONFIRMATION_SEED_OFFSET + episode_index
        episode = episode_tensors(images, labels, splits["confirmation"], seed, device)
        episode_rows, episode_diagnostics = evaluate_confirmation_episode(model, episode, seed, config, bank)
        rows.extend(episode_rows)
        diagnostics.append(episode_diagnostics)
        print(f"confirmation episode {episode_index + 1:3d}/{args.episodes}", flush=True)
    write_csv(output / "confirmation_episodes.csv", rows)
    result = summarize_confirmation(rows, output)
    resource_counts = {
        "context_examples": 30,
        "query_examples": 100,
        "online_trainable_parameters_internal": 6,
        "online_trainable_parameters_shallow": FEATURE_DIM * config.shallow_width + config.shallow_width + config.shallow_width * N_CLASSES + N_CLASSES,
        "internal_adam_state_scalars": 12,
        "ridge_solve_dimension": 30,
        "rbf_solve_dimension": 30,
        "geometric_bank_views": config.geometric_bank_size,
        "geometric_bank_feature_dimension": config.geometric_bank_size * FEATURE_DIM,
        "tangent_solve_dimension": 6,
        "tangent_finite_difference_backbone_passes_per_context_set": 12,
        "internal_two_stage_training_image_evaluations": config.internal_stage_steps * (20 + 30),
        "internal_full_training_image_evaluations": 2 * config.internal_stage_steps * 30,
        "geometric_bank_image_evaluations_context_plus_query": config.geometric_bank_size * 130,
        "peak_cuda_memory_bytes_whole_run": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    }
    metadata = {
        "selected_config": asdict(config),
        "confirmation_seed_range": [CONFIRMATION_SEED_OFFSET, CONFIRMATION_SEED_OFFSET + args.episodes - 1],
        "checkpoint_sha256": observed_hash,
        "query_targets_used_for_adaptation": False,
        "all_context_query_sets_disjoint": all(item["context_query_disjoint"] for item in diagnostics),
        "resource_counts": resource_counts,
        "decision": result["decision"],
    }
    (output / "confirmation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "episode_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2), flush=True)


def audit_protocol(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, images, labels, splits = load_model_and_data(args.checkpoint_dir, device)
    episode = episode_tensors(images, labels, splits["validation"], VALIDATION_SEED_OFFSET, device)
    assert not (set(episode["context_indices"]) & set(episode["query_indices"]))
    theta = episode["corruption_theta"]
    inverse = inverse_theta(torch.as_tensor(theta, device=device))
    composition = torch.cat([torch.as_tensor(theta, device=device), torch.tensor([[0.0, 0.0, 1.0]], device=device)]) @ torch.cat([inverse, torch.tensor([[0.0, 0.0, 1.0]], device=device)])
    identity_error = float((composition - torch.eye(3, device=device)).abs().max())
    raw = torch.zeros(6, device=device, requires_grad=True)
    loss = F.cross_entropy(model(warp(episode["adapt_images"], correction_theta(raw))), episode["adapt_labels"])
    gradient = torch.autograd.grad(loss, raw)[0]
    base, jacobian = tangent_jacobian(model, episode["context_images"])
    bank = fixed_geometric_bank(13, device)
    bank_shape = tuple(bank_features(model, episode["context_images"], bank).shape)
    report = {
        "context_query_disjoint": True,
        "context_count": int(episode["context_labels"].numel()),
        "query_count": int(episode["query_labels"].numel()),
        "all_classes_in_adapt": sorted(set(episode["adapt_labels"].cpu().tolist())),
        "all_classes_in_calibration": sorted(set(episode["calibration_labels"].cpu().tolist())),
        "all_classes_in_query": sorted(set(episode["query_labels"].cpu().tolist())),
        "affine_inverse_composition_max_error": identity_error,
        "internal_gradient_finite": bool(torch.isfinite(gradient).all()),
        "internal_gradient_norm": float(gradient.norm()),
        "tangent_base_shape": list(base.shape),
        "tangent_jacobian_shape": list(jacobian.shape),
        "geometric_bank_feature_shape": list(bank_shape),
        "query_targets_used_for_adaptation": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    train = subparsers.add_parser("train")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument("--device", default=default_device)
    train.set_defaults(function=train_backbone)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--checkpoint-dir", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--episodes", type=int, default=N_VALIDATION_EPISODES)
    validate.add_argument("--device", default=default_device)
    validate.set_defaults(function=validate_hyperparameters)

    confirm = subparsers.add_parser("confirm")
    confirm.add_argument("--checkpoint-dir", type=Path, required=True)
    confirm.add_argument("--selected-config", type=Path, required=True)
    confirm.add_argument("--output-dir", type=Path, required=True)
    confirm.add_argument("--episodes", type=int, default=N_CONFIRMATION_EPISODES)
    confirm.add_argument("--device", default=default_device)
    confirm.set_defaults(function=run_confirmation)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--checkpoint-dir", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.add_argument("--device", default=default_device)
    audit.set_defaults(function=audit_protocol)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
