#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset tools for V2.1 LeRobot datasets."""

import json
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def modify_features(
    dataset: LeRobotDataset,
    add_features: dict[str, tuple[np.ndarray | torch.Tensor | Callable, dict]] | None = None,
    remove_features: str | list[str] | None = None,
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
) -> LeRobotDataset:
    """Modify a V2.1 LeRobotDataset by adding/removing features.

    Args:
        dataset: Source LeRobotDataset (V2.1).
        add_features: Dict mapping feature names to (values, feature_info) tuples.
            values: numpy array, torch tensor, or callable(row_dict, ep_idx, frame_idx)
            feature_info: {"dtype": str, "shape": list, "names": None}
        remove_features: Feature name(s) to remove.
        output_dir: Output directory. Defaults to HF cache.
        repo_id: New repo_id. Defaults to "{original}_modified".

    Returns:
        New LeRobotDataset with modified features.
    """
    if add_features is None and remove_features is None:
        raise ValueError("Must specify add_features or remove_features")

    remove_list = []
    if remove_features:
        remove_list = [remove_features] if isinstance(remove_features, str) else list(remove_features)

    # Validate
    if add_features:
        for name, (_, info) in add_features.items():
            if name in dataset.meta.features:
                raise ValueError(f"Feature '{name}' already exists")
            if not {"dtype", "shape"}.issubset(info.keys()):
                raise ValueError(f"feature_info for '{name}' must have 'dtype' and 'shape'")

    required = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    if any(f in required for f in remove_list):
        raise ValueError(f"Cannot remove required features: {required}")

    # Setup paths
    if repo_id is None:
        repo_id = f"{dataset.repo_id}_modified"
    if output_dir is None:
        output_dir = HF_LEROBOT_HOME / repo_id
    output_dir = Path(output_dir)

    # Build new features dict
    new_features = dataset.meta.features.copy()
    for name in remove_list:
        new_features.pop(name, None)
    if add_features:
        for name, (_, info) in add_features.items():
            new_features[name] = info

    # Copy and modify parquet files
    data_dir = dataset.root / "data"
    frame_idx = 0

    for chunk_dir in tqdm(sorted(data_dir.iterdir()), desc="Processing chunks"):
        if not chunk_dir.is_dir():
            continue
        for parquet_file in sorted(chunk_dir.glob("episode_*.parquet")):
            df = pd.read_parquet(parquet_file)

            # Remove columns
            if remove_list:
                df = df.drop(columns=remove_list, errors="ignore")

            # Add columns
            if add_features:
                end_idx = frame_idx + len(df)
                for name, (values, _) in add_features.items():
                    if callable(values):
                        col_values = []
                        for _, row in df.iterrows():
                            ep_idx = row["episode_index"]
                            frame_in_ep = row["frame_index"]
                            val = values(row.to_dict(), ep_idx, frame_in_ep)
                            if isinstance(val, np.ndarray) and val.size == 1:
                                val = val.item()
                            col_values.append(val)
                        df[name] = col_values
                    else:
                        if isinstance(values, torch.Tensor):
                            values = values.numpy()
                        slice_vals = values[frame_idx:end_idx]
                        if len(slice_vals.shape) > 1 and slice_vals.shape[1] == 1:
                            df[name] = slice_vals.flatten()
                        else:
                            df[name] = list(slice_vals)
                frame_idx = end_idx

            # Write to output
            rel_path = parquet_file.relative_to(dataset.root)
            dst_path = output_dir / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst_path)

    # Copy and update metadata
    _copy_metadata(dataset, output_dir, new_features, remove_list)

    # Copy videos (excluding removed video keys)
    video_keys_to_remove = [k for k in remove_list if k in dataset.meta.video_keys]
    _copy_videos(dataset, output_dir, exclude_keys=video_keys_to_remove)

    return LeRobotDataset(repo_id=repo_id, root=output_dir)


def _copy_metadata(dataset: LeRobotDataset, output_dir: Path, new_features: dict, remove_list: list):
    """Copy and update metadata files."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Update info.json with new features
    info = dataset.meta.info.copy()
    info["features"] = new_features
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Copy tasks.jsonl
    src_tasks = dataset.root / "meta" / "tasks.jsonl"
    if src_tasks.exists():
        shutil.copy(src_tasks, meta_dir / "tasks.jsonl")

    # Copy episodes.jsonl
    src_episodes = dataset.root / "meta" / "episodes.jsonl"
    if src_episodes.exists():
        shutil.copy(src_episodes, meta_dir / "episodes.jsonl")

    # Filter episodes_stats.jsonl to remove stats for removed features
    src_stats = dataset.root / "meta" / "episodes_stats.jsonl"
    if src_stats.exists():
        with open(src_stats) as f:
            lines = f.readlines()
        filtered = []
        for line in lines:
            data = json.loads(line)
            for key in remove_list:
                data.pop(key, None)
            filtered.append(json.dumps(data))
        with open(meta_dir / "episodes_stats.jsonl", "w") as f:
            f.write("\n".join(filtered))

    # Filter stats.json
    src_global_stats = dataset.root / "meta" / "stats.json"
    if src_global_stats.exists():
        with open(src_global_stats) as f:
            stats = json.load(f)
        for key in remove_list:
            stats.pop(key, None)
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)


def _copy_videos(dataset: LeRobotDataset, output_dir: Path, exclude_keys: list | None = None):
    """Copy video files, excluding specified keys."""
    exclude_keys = exclude_keys or []
    src_videos = dataset.root / "videos"
    if not src_videos.exists():
        return

    for chunk_dir in src_videos.iterdir():
        if not chunk_dir.is_dir():
            continue
        for key_dir in chunk_dir.iterdir():
            if key_dir.name in exclude_keys:
                continue
            dst_dir = output_dir / "videos" / chunk_dir.name / key_dir.name
            if key_dir.is_dir():
                shutil.copytree(key_dir, dst_dir, dirs_exist_ok=True)
