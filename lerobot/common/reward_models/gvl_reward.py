"""GVL-based reward model for reward-aligned behavioral cloning.

This module wraps the GenerativeValueLearner from the GVL package to provide
instruction-conditioned rewards for trajectory scoring. It uses the VLM's
log-likelihood of instructions given observations as the reward signal.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torchvision import transforms

# Add GVL to path for import
gvl_path = Path(__file__).parent.parent.parent.parent.parent / "gvl"
if str(gvl_path) not in sys.path:
    sys.path.insert(0, str(gvl_path))

from gen_value.gvl import GenerativeValueLearner


class GVLRewardModel:
    """Reward scorer built on top of GVL's GenerativeValueLearner.

    This model uses the VLM's instruction-following capability to score
    observation/action pairs. Higher rewards indicate better alignment between
    the visual observation and the task instruction.

    Unlike the QwenVLRewardModel which scores actions directly, this model
    scores based on visual instruction-following, which can capture more
    nuanced task understanding.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 256 * 28 * 28,
        reduction: str = "mean",
        use_video: bool = False,
        fps: float = 2.0,
    ) -> None:
        """Initialize GVL reward model.

        Args:
            model_id: Qwen model ID or "gemini-1.5-pro"
            device: Device for inference
            dtype: Data type for model weights
            max_pixels: Maximum image resolution (Qwen only)
            min_pixels: Minimum image resolution (Qwen only)
            reduction: How to reduce token log probs ("mean" or "sum")
            use_video: Whether to use video input (requires multiple frames)
            fps: Frames per second for video input
        """
        self.model_id = model_id
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        self.reduction = reduction
        self.use_video = use_video
        self.fps = fps
        self.to_pil = transforms.ToPILImage()

        # Initialize GVL
        self.gvl = GenerativeValueLearner(
            model_name=model_id,
            device=str(self.device),
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )

        # For Gemini models, we can't use instruction reward yet
        if self.gvl.is_gemini:
            raise NotImplementedError(
                "GVL instruction reward is not yet implemented for Gemini models. "
                "Please use a Qwen model instead."
            )

    def score(
        self,
        tasks: Iterable[str],
        actions: torch.Tensor,
        images: Iterable[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Score each action with the reward model.

        This method computes instruction-following rewards by measuring how well
        the visual observations align with the task instructions. The reward is
        based on the VLM's log-likelihood of the instruction given the observation.

        Args:
            tasks: A sequence of task strings with length equal to the batch size.
            actions: Tensor of shape `(batch, num_candidates, action_dim)` with
                unnormalized actions. Note: Actions are not used in scoring, only
                for consistency with the interface. We score based on visual
                instruction-following instead.
            images: Iterable of image tensors shaped `(3, H, W)` or `(num_frames, 3, H, W)`
                aligned with the batch dimension. If None, raises ValueError.

        Returns:
            Tensor of shape `(batch, num_candidates)` with higher-is-better rewards.
            All candidates receive the same reward since scoring is based on
            observations, not actions.
        """
        if images is None:
            raise ValueError(
                "GVL reward model requires images. Please provide reward_image_key "
                "in reward_alignment config."
            )

        batch, num_candidates, _ = actions.shape
        tasks_list = list(tasks)
        image_list = list(images)

        # Compute rewards for each observation-task pair
        rewards_list = []
        for task, image_tensor in zip(tasks_list, image_list):
            # Convert image tensor to PIL
            if image_tensor.dim() == 3:
                # Single frame: (3, H, W)
                if self.use_video:
                    # Repeat frame to create a 2-frame "video"
                    frames = [self.to_pil(image_tensor.cpu())] * 2
                else:
                    # Use single frame by creating a minimal 2-frame sequence
                    # (GVL requires at least 2 frames)
                    frames = [self.to_pil(image_tensor.cpu())] * 2
            elif image_tensor.dim() == 4:
                # Multiple frames: (num_frames, 3, H, W)
                frames = [self.to_pil(frame.cpu()) for frame in image_tensor]
                if len(frames) < 2:
                    # Duplicate if only 1 frame
                    frames = frames * 2
            else:
                raise ValueError(
                    f"Expected image tensor of shape (3, H, W) or (num_frames, 3, H, W), "
                    f"got shape {image_tensor.shape}"
                )

            # Compute instruction reward using GVL
            reward, _ = self.gvl.compute_instruction_reward(
                frames=frames,
                instruction=task,
                reduction=self.reduction,
                fps=self.fps,
            )

            rewards_list.append(reward)

        # Stack rewards: shape (batch,)
        rewards_tensor = torch.tensor(rewards_list, device=self.device, dtype=torch.float32)

        # Expand to (batch, num_candidates) - same reward for all candidates
        # since we score based on observations, not actions
        rewards = rewards_tensor.unsqueeze(1).expand(batch, num_candidates)

        return rewards
