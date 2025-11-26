"""Qwen-VL wrapper for reward-aligned behavioral cloning.

This module provides a thin interface around Qwen-VL to score observation/action
pairs with a scalar reward. It follows the reward-aligned behavioral cloning
setup referenced in https://arxiv.org/abs/2509.25358 by ranking candidate
trajectories and normalizing scores with a softmax.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoProcessor


class QwenVLRewardModel:
    """Reward scorer built on top of a Qwen-VL causal language model.

    The model turns (task, action) pairs and an optional image into a scalar
    reward by computing the negative language modeling loss of a descriptive
    prompt. Higher values indicate better alignment between the action and the
    task.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        prompt_template: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=None,
        ).to(self.device)
        self.prompt_template = (
            prompt_template
            or "Task: {task}\nProposed action vector: [{action}]\n"
            "Give a single numeric score between 0 and 1 for how well the action completes the task."
        )
        self.to_pil = transforms.ToPILImage()

    def score(
        self,
        tasks: Iterable[str],
        actions: torch.Tensor,
        images: Iterable[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Score each action with the reward model.

        Args:
            tasks: A sequence of task strings with length equal to the batch size.
            actions: Tensor of shape `(batch, num_candidates, action_dim)` with
                unnormalized actions.
            images: Optional iterable of image tensors shaped `(3, H, W)` aligned
                with the batch dimension.

        Returns:
            Tensor of shape `(batch, num_candidates)` with higher-is-better rewards.
        """

        batch, num_candidates, _ = actions.shape
        tasks_list = list(tasks)
        flat_actions = actions.reshape(-1, actions.shape[-1])
        prompt_texts: list[str] = []
        safe_images: list | None = None
        image_list = list(images) if images is not None else None

        for idx, action in enumerate(flat_actions):
            task_idx = idx // num_candidates
            action_desc = ", ".join(f"{value:.3f}" for value in action.tolist())
            prompt_texts.append(self.prompt_template.format(task=tasks_list[task_idx], action=action_desc))

        processor_kwargs: dict = {"text": prompt_texts, "padding": True, "return_tensors": "pt"}
        if image_list is not None:
            safe_images = [self.to_pil(image.cpu()) for image in image_list for _ in range(num_candidates)]
            processor_kwargs["images"] = safe_images

        inputs = self.processor(**processor_kwargs)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        shift_logits = logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]
        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].bool()
        else:
            shift_mask = torch.ones_like(shift_labels, dtype=torch.bool)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * shift_mask

        seq_log_prob = token_log_probs.sum(dim=-1)
        seq_lengths = shift_mask.sum(dim=-1).clamp_min(1)
        avg_log_prob = seq_log_prob / seq_lengths

        rewards = avg_log_prob.detach().reshape(batch, num_candidates)
        return rewards
