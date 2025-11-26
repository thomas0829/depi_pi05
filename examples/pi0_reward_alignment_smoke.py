"""Minimal smoke test for pi0 reward-aligned behavioral cloning.

This script builds a pi0 policy alongside the Qwen-VL reward model and runs a
single reward-aligned forward pass on dummy data to exercise the
`reward_aligned_forward` path without requiring a full training job.

Example:
    python examples/pi0_reward_alignment_smoke.py \
        --reward_model_id Qwen/Qwen2-VL-2B-Instruct \
        --device cuda
"""

from __future__ import annotations

import argparse

import torch

from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.reward_models.qwen_vl import QwenVLRewardModel


def build_dummy_batch(config: PI0Config, device: torch.device) -> dict:
    batch_size = 2
    image = torch.zeros(batch_size, 3, 480, 640, device=device)
    state = torch.zeros(batch_size, config.max_state_dim, device=device)
    action = torch.zeros(batch_size, config.chunk_size, config.max_action_dim, device=device)
    return {
        "task": ["Pick the red block" for _ in range(batch_size)],
        "observation.images.0": image,
        "observation.state": state,
        "action": action,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_model_id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device(args.device)
    config = PI0Config()
    policy = PI0Policy(config).to(device)
    reward_model = QwenVLRewardModel(model_id=args.reward_model_id, device=device)

    batch = build_dummy_batch(config, device)
    loss, logs = policy.reward_aligned_forward(
        batch,
        reward_model,
        num_candidates=2,
        temperature=args.temperature,
        reward_image_key="observation.images.0",
    )
    print(f"Loss: {loss.item():.4f}")
    print(f"Logs: {logs}")


if __name__ == "__main__":
    main()
