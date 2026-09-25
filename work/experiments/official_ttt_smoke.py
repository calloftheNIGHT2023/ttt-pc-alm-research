"""Audit the unmodified official PyTorch TTT-MLP inner update.

The test intentionally exercises both the public forward path and the lower-level
TTT fast-weight routine.  It does not claim that the synthetic tensors constitute
a trained task model; the purpose is to pin down the mechanism we must compare
against before building a meta-trained bridge.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = ROOT / "work" / "third_party" / "ttt-lm-pytorch"


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def max_parameter_change(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    return max(
        float((before[name] - value.detach()).abs().max())
        for name, value in module.state_dict().items()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(OFFICIAL_REPO))
    import ttt  # type: ignore  # imported from the pinned official checkout

    torch.manual_seed(args.seed)
    config = ttt.TTTConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        mini_batch_size=4,
        ttt_layer_type="mlp",
        share_qk=False,
        use_cache=False,
    )
    model = ttt.TTTMLP(config, layer_idx=0).eval()
    slow_before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    batch, sequence, width = 2, 8, 8
    hidden = torch.randn(batch, sequence, width)
    positions = torch.arange(sequence).repeat(batch, 1)
    with torch.no_grad():
        public_output = model(hidden, position_ids=positions)

    heads, mini_batches, mini_batch, head_dim = 1, 2, 4, 8
    xq = torch.randn(batch, heads, mini_batches, mini_batch, head_dim)
    xk = torch.randn(batch, heads, mini_batches, mini_batch, head_dim)
    xv = torch.randn(batch, heads, mini_batches, mini_batch, head_dim)
    token_eta = torch.ones(batch, heads, mini_batches, mini_batch, 1)
    ttt_lr_eta = torch.full((batch, heads, mini_batches, 1, mini_batch), 0.01)
    eta = token_eta * ttt_lr_eta
    inputs = {
        "XQ": xq,
        "XK": xk,
        "XV": xv,
        "eta": eta,
        "token_eta": token_eta,
        "ttt_lr_eta": ttt_lr_eta,
    }
    with torch.no_grad():
        direct_output, fast_state = model.ttt(inputs, mini_batch, None, None)

    # Holding K/V and the learning rates fixed, Q is read-only: it changes the
    # retrieved output but cannot change the final fast weights.
    query_shifted = dict(inputs)
    query_shifted["XQ"] = xq + 0.25 * torch.randn_like(xq)
    with torch.no_grad():
        shifted_output, shifted_fast_state = model.ttt(query_shifted, mini_batch, None, None)

    fast_names = ("W1_states", "b1_states", "W2_states", "b2_states")
    query_fast_state_difference = max(
        float((fast_state[name] - shifted_fast_state[name]).abs().max()) for name in fast_names
    )
    fast_parameter_count = sum(int(fast_state[name][0].numel()) for name in fast_names)
    gradient_carry_count = sum(
        int(fast_state[name.replace("_states", "_grad")][0].numel()) for name in fast_names
    )
    metrics = {
        "purpose": "mechanism smoke test only; not downstream accuracy evidence",
        "official_repo_commit": git_commit(OFFICIAL_REPO),
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "seed": args.seed,
        "configuration": {
            "hidden_size": width,
            "heads": heads,
            "head_dim": head_dim,
            "mini_batch_size": mini_batch,
            "batch": batch,
            "sequence": sequence,
        },
        "inner_reconstruction_target": "XV - XK",
        "public_forward_shape": list(public_output.shape),
        "public_forward_all_finite": bool(torch.isfinite(public_output).all()),
        "direct_ttt_shape": list(direct_output.shape),
        "direct_ttt_all_finite": bool(torch.isfinite(direct_output).all()),
        "fast_parameter_count_per_episode": fast_parameter_count,
        "gradient_carry_count_per_episode": gradient_carry_count,
        "W1_fast_max_change": float(
            (fast_state["W1_states"] - model.W1.detach().unsqueeze(0)).abs().max()
        ),
        "W2_fast_max_change": float(
            (fast_state["W2_states"] - model.W2.detach().unsqueeze(0)).abs().max()
        ),
        "query_perturbation_output_max_change": float(
            (direct_output - shifted_output).abs().max()
        ),
        "query_perturbation_fast_state_max_change": query_fast_state_difference,
        "slow_parameter_max_change": max_parameter_change(slow_before, model),
    }

    assert metrics["public_forward_shape"] == [batch, sequence, width]
    assert metrics["direct_ttt_shape"] == [batch, sequence, width]
    assert metrics["public_forward_all_finite"]
    assert metrics["direct_ttt_all_finite"]
    assert metrics["W1_fast_max_change"] > 0.0
    assert metrics["W2_fast_max_change"] > 0.0
    assert metrics["query_perturbation_output_max_change"] > 0.0
    assert metrics["query_perturbation_fast_state_max_change"] == 0.0
    assert metrics["slow_parameter_max_change"] == 0.0

    output_path = args.output_dir / "official_ttt_smoke.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
