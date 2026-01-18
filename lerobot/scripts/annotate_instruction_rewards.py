#!/usr/bin/env python
"""Annotate LeRobot datasets with instruction rewards for advantage-weighted behavior cloning.

This script computes instruction rewards using Qwen3-VL for each trajectory prefix,
then derives TD-style advantages and stores them directly in the dataset using modify_features().

The advantage column is added directly to the parquet files, making it available
during training without any additional loading mechanism.

Usage:
    python lerobot/scripts/annotate_instruction_rewards.py \
        --input_repo_id thomas0829/eval_put_the_doll_into_the_box \
        --output_dir ./datasets/ \
        --output_repo_id put_the_doll_into_the_box_adv \
        --model_name Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm
from loguru import logger
from opengvl.utils.logging_config import setup_logging

def parse_args():
    # Configure logging format
    setup_logging(level="INFO", format_type="detailed")
    parser = argparse.ArgumentParser(
        description="Annotate LeRobot datasets with instruction rewards stored directly in parquet"
    )
    parser.add_argument(
        "--input_repo_id",
        type=str,
        required=True,
        help="Source dataset repo_id (e.g., 'lerobot/pusht')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/",
        help="Directory to save the annotated dataset",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        default=None,
        help="Output dataset repo_id (default: {input_repo_id}_with_advantages)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen VLM model to use for computing instruction rewards",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to annotate (default: all)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="FPS for video input to the VLM (default: 2.0)",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Reduction method for log probabilities (default: mean)",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Sample every N-th frame for VLM evaluation (default: 1 = all frames)",
    )
    parser.add_argument(
        "--reward_stride",
        type=int,
        default=50,
        help="Compute instruction rewards every N sampled frames (default: 1 = every frame).",
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=None,
        help="Root directory for input dataset (default: HF cache)",
    )
    return parser.parse_args()


def extract_frames_from_episode(dataset, episode_index: int, sample_interval: int = 1):
    """Extract frames from a LeRobot episode.

    Args:
        dataset: LeRobotDataset or LeRobotDatasetV3 instance.
        episode_index: Episode index to extract frames from.
        sample_interval: Sample every N-th frame for VLM evaluation.

    Returns:
        Tuple of (frames_list, instruction_text, frame_indices, total_frames)
    """
    # Get episode data indices
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
        # V3 dataset
        episodes = dataset.meta.episodes
        if episodes is not None and episode_index < len(episodes):
            ep_data = episodes[episode_index]
            start_idx = ep_data.get("dataset_from_index", 0)
            end_idx = ep_data.get("dataset_to_index", start_idx + ep_data.get("length", 0))
        else:
            raise ValueError(f"Episode {episode_index} not found in metadata")
    elif hasattr(dataset, "episode_data_index"):
        # V2.1 dataset
        start_idx = dataset.episode_data_index["from"].get(episode_index, 0)
        end_idx = dataset.episode_data_index["to"].get(episode_index, start_idx)
    else:
        raise ValueError("Cannot determine episode boundaries")

    total_frames = end_idx - start_idx
    frames = []
    # Sample frames at intervals for VLM evaluation
    sampled_indices = list(range(start_idx, end_idx, sample_interval))

    for idx in sampled_indices:
        item = dataset[idx]
        # Get the first available camera key
        camera_key = None
        for key in item:
            if key.startswith("observation.images"):
                camera_key = key
                break

        if camera_key is None:
            raise ValueError(f"No camera key found in dataset item: {list(item.keys())}")

        # Convert to numpy array (C, H, W) -> (H, W, C)
        frame = item[camera_key]
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        if frame.shape[0] in [1, 3]:  # Channels first
            frame = np.transpose(frame, (1, 2, 0))
        # Convert from [0, 1] to [0, 255] if needed
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        frames.append(frame)

    # Get instruction from first frame
    first_item = dataset[start_idx]
    instruction = first_item.get("task", "")
    if (
        (instruction is None or instruction == "")
        and hasattr(dataset, "meta")
        and hasattr(dataset.meta, "tasks")
    ):
        # Try to get from metadata
        task_index = first_item.get("task_index", 0)
        if hasattr(task_index, "item"):
            task_index = task_index.item()
        if hasattr(dataset.meta.tasks, "iloc"):
            instruction = dataset.meta.tasks.iloc[task_index].name
        elif isinstance(dataset.meta.tasks, dict):
            instruction = dataset.meta.tasks.get(task_index, "")

    return frames, instruction, sampled_indices, total_frames


def compute_advantages_for_episode(
    client,
    frames: list,
    instruction: str,
    total_frames: int,
    sampled_indices: list,
    start_idx: int,
    fps: float = 2.0,
    reduction: str = "mean",
    reward_stride: int = 50,
) -> np.ndarray:
    """Compute instruction rewards and TD-style advantages for an episode.

    Args:
        client: QwenClient instance.
        frames: List of frame images (numpy arrays in HWC format, uint8).
        instruction: Task instruction text.
        total_frames: Total number of frames in the episode.
        sampled_indices: Global indices of the sampled frames.
        start_idx: Starting global index of the episode.
        fps: FPS for video input.
        reduction: Reduction method for log probabilities.
        reward_stride: Compute rewards every N sampled frames (>=1), then broadcast.

    Returns:
        numpy array of advantages for ALL frames in the episode (not just sampled).
    """
    if len(frames) < 2:
        return np.zeros(total_frames, dtype=np.float32)

    prefix_rewards = []

    # Compute reward once per chunk boundary (sampled frame) with optional stride.
    # Start from t=2 since the reward function expects at least 2 frames.
    prefix_lengths = list(range(2, len(frames) + 1, max(reward_stride, 1)))
    if prefix_lengths[-1] != len(frames):
        prefix_lengths.append(len(frames))

    for t in tqdm(prefix_lengths, desc="Computing chunk rewards"):
        logger.info(f"Computing reward for prefix length {t}")
        prefix_frames = frames[:t]
        try:
            result = client.compute_instruction_reward(
                frames=prefix_frames,
                instruction=instruction,
                reduction=reduction,
                fps=fps,
                use_video_description=False,
                add_chat_template=False,
            )
            prefix_rewards.append(result.reward)
        except Exception as e:
            logger.warning(f"Error computing reward for prefix length {t}: {e}")
            # Use the previous reward as fallback
            prefix_rewards.append(prefix_rewards[-1] if prefix_rewards else 0.0)

    if not prefix_rewards:
        return np.zeros(total_frames, dtype=np.float32)

    # Map chunk rewards to each sampled frame; repeat the last reward for the final chunk.
    # Broadcast the last computed prefix reward forward until the next computed checkpoint.
    chunk_rewards = []
    pref_idx = 0
    for i in range(1, len(frames) + 1):
        while pref_idx + 1 < len(prefix_lengths) and prefix_lengths[pref_idx + 1] <= i:
            pref_idx += 1
        chunk_rewards.append(prefix_rewards[pref_idx])

    # Compute TD-style advantages for sampled frames
    # advantage_t = R(frames[0:t+1]) - R(frames[0:t])
    sampled_advantages = [0.0]  # First frame has no prior
    for t in range(1, len(chunk_rewards)):
        sampled_advantages.append(chunk_rewards[t] - chunk_rewards[t - 1])

    # Now interpolate advantages to all frames in the episode
    # Each sampled frame's advantage applies to itself and the frames until the next sample
    all_advantages = np.zeros(total_frames, dtype=np.float32)

    # Convert sampled_indices to local indices within episode
    local_sampled_indices = [idx - start_idx for idx in sampled_indices]

    for i, local_idx in enumerate(local_sampled_indices):
        adv = sampled_advantages[i]
        # Determine the range this advantage applies to
        end_local_idx = local_sampled_indices[i + 1] if i < len(local_sampled_indices) - 1 else total_frames

        # Assign the advantage to all frames in this range
        all_advantages[local_idx:end_local_idx] = adv

    # Normalize advantages per episode:
    # 1) clip to [-0.5, 1]
    # 2) rescale positive values so the maximum positive becomes 1 (if any positives exist)
    if all_advantages.size > 0:
        all_advantages = np.clip(all_advantages, -0.5, 1.0)
        max_pos = float(all_advantages[all_advantages > 0].max(initial=0.0))
        if max_pos > 0:
            pos_mask = all_advantages > 0
            all_advantages[pos_mask] = all_advantages[pos_mask] / max_pos

    return all_advantages


def annotate_dataset(
    input_repo_id: str,
    output_dir: Path,
    output_repo_id: str,
    model_name: str,
    max_episodes: int | None = None,
    fps: float = 2.0,
    reduction: str = "mean",
    sample_interval: int = 1,
    reward_stride: int = 50,
    input_root: str | None = None,
):
    """Annotate a LeRobot dataset with instruction rewards stored in parquet."""
    # Import here to avoid circular imports and allow for PYTHONPATH setup
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset_v3 import LeRobotDatasetV3
    from lerobot.common.datasets.v3.dataset_tools import modify_features
    from opengvl.clients.qwen import QwenClient

    # Load dataset
    logger.info(f"Loading dataset: {input_repo_id}")
    try:
        dataset = LeRobotDatasetV3(input_repo_id, root=input_root)
        is_v3 = True
    except Exception:
        logger.info("V3 loader failed, trying V2.1 loader...")
        dataset = LeRobotDataset(input_repo_id, root=input_root)
        is_v3 = False

    num_episodes = (
        dataset.meta.total_episodes if hasattr(dataset.meta, "total_episodes") else dataset.num_episodes
    )
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)

    total_frames = dataset.meta.total_frames if hasattr(dataset.meta, "total_frames") else len(dataset)
    logger.info(f"Dataset loaded: {num_episodes} episodes, {total_frames} frames to annotate")

    # Initialize VLM client
    logger.info(f"Loading VLM: {model_name}")
    client = QwenClient(model_name=model_name)

    # Compute advantages for all episodes
    all_advantages = []

    for ep_idx in range(num_episodes):
        logger.info(f"Annotating episode {ep_idx+1}/{num_episodes}")
        try:
            # Extract frames
            breakpoint()
            frames, instruction, sampled_indices, ep_total_frames = extract_frames_from_episode(
                dataset, ep_idx, sample_interval
            )

            if len(frames) < 2:
                logger.warning(f"Episode {ep_idx} has fewer than 2 frames, using zero advantages")
                ep_advantages = np.zeros(ep_total_frames, dtype=np.float32)
            else:
                logger.info(
                    f"Episode {ep_idx}: {ep_total_frames} frames, instruction: '{instruction[:50]}...'"
                )

                # Get episode start index
                if is_v3:
                    start_idx = dataset.meta.episodes[ep_idx].get("dataset_from_index", 0)
                else:
                    start_idx = dataset.episode_data_index["from"].get(ep_idx, 0)

                # Compute advantages
                ep_advantages = compute_advantages_for_episode(
                    client,
                    frames,
                    instruction,
                    ep_total_frames,
                    sampled_indices,
                    start_idx,
                    fps=fps,
                    reduction=reduction,
                    reward_stride=reward_stride,
                )

            all_advantages.append(ep_advantages)

        except Exception as e:
            logger.error(f"Error annotating episode {ep_idx}: {e}")
            # Get episode length and use zeros
            if is_v3:
                ep_length = dataset.meta.episodes[ep_idx].get("length", 0)
            else:
                start_idx = dataset.episode_data_index["from"].get(ep_idx, 0)
                end_idx = dataset.episode_data_index["to"].get(ep_idx, start_idx)
                ep_length = end_idx - start_idx
            all_advantages.append(np.zeros(ep_length, dtype=np.float32))
            continue

    # Concatenate all advantages into a single array
    advantages_array = np.concatenate(all_advantages)
    logger.info(f"Computed {len(advantages_array)} advantages")
    logger.info(
        f"Advantage stats: mean={advantages_array.mean():.4f}, std={advantages_array.std():.4f}, "
        f"min={advantages_array.min():.4f}, max={advantages_array.max():.4f}"
    )

    # Ensure the dataset is V3 for modify_features
    if not is_v3:
        logger.error("modify_features only works with V3 datasets. Please convert the dataset first.")
        raise ValueError("Dataset must be V3 format to use modify_features")

    # Use modify_features to add the advantage column
    logger.info(f"Creating new dataset with advantage column at {output_dir / output_repo_id}")

    new_dataset = modify_features(
        dataset=dataset,
        add_features={"advantage": (advantages_array, {"dtype": "float32", "shape": (1,), "names": None})},
        output_dir=output_dir,
        repo_id=output_repo_id,
    )

    logger.info(f"Dataset created: {new_dataset.repo_id}")
    logger.info(f"Features: {list(new_dataset.meta.features.keys())}")
    logger.info(f"'advantage' in features: {'advantage' in new_dataset.meta.features}")

    # Verify by loading a sample
    sample = new_dataset[0]
    if "advantage" in sample:
        logger.info(f"Sample advantage value: {sample['advantage']}")
    else:
        logger.warning("'advantage' not found in sample!")

    return new_dataset


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output repo_id
    if args.output_repo_id is None:
        # Strip any path prefix and add suffix
        base_name = args.input_repo_id.split("/")[-1]
        output_repo_id = f"{base_name}_with_advantages"
    else:
        output_repo_id = args.output_repo_id

    logger.info(f"Processing dataset: {args.input_repo_id}")
    logger.info(f"Output: {output_dir / output_repo_id}")

    try:
        annotate_dataset(
            input_repo_id=args.input_repo_id,
            output_dir=output_dir,
            output_repo_id=output_repo_id,
            model_name=args.model_name,
            max_episodes=args.max_episodes,
            fps=args.fps,
            reduction=args.reduction,
            sample_interval=args.sample_interval,
            reward_stride=args.reward_stride,
            input_root=args.input_root,
        )
    except Exception as e:
        logger.error(f"Failed to annotate {args.input_repo_id}: {e}")
        raise


if __name__ == "__main__":
    main()
