"""
Example script demonstrating the GVL reward model integration with DePi.

This script shows how to:
1. Load the GVL reward model
2. Score observations with task instructions
3. Compare GVL vs QwenVL reward models

Usage:
    python examples/test_gvl_reward.py
"""

import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.common.reward_models.gvl_reward import GVLRewardModel
from lerobot.common.reward_models.qwen_vl import QwenVLRewardModel


def create_dummy_data(batch_size=2, num_candidates=4, action_dim=7):
    """Create dummy data for testing reward models."""
    # Create dummy tasks
    tasks = [
        "Pick up the red block and place it in the bin",
        "Push the blue button",
    ][:batch_size]

    # Create dummy actions (batch, num_candidates, action_dim)
    actions = torch.randn(batch_size, num_candidates, action_dim)

    # Create dummy images (3, H, W) - simple colored squares
    images = []
    for i in range(batch_size):
        # Create a random colored image
        img_array = np.random.rand(224, 224, 3) * 255
        img_array = img_array.astype(np.uint8)
        img_pil = Image.fromarray(img_array)

        # Convert to tensor (3, H, W)
        img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
        images.append(img_tensor)

    return tasks, actions, images


def test_gvl_reward_model():
    """Test the GVL reward model."""
    print("=" * 80)
    print("Testing GVL Reward Model")
    print("=" * 80)

    # Create model
    print("\n1. Loading GVL reward model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")

    try:
        gvl_model = GVLRewardModel(
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            device=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            reduction="mean",
            use_video=False,
        )
        print("   ✓ GVL model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load GVL model: {e}")
        return False

    # Create dummy data
    print("\n2. Creating dummy data...")
    batch_size = 2
    num_candidates = 4
    action_dim = 7

    tasks, actions, images = create_dummy_data(batch_size, num_candidates, action_dim)
    print(f"   - Batch size: {batch_size}")
    print(f"   - Num candidates: {num_candidates}")
    print(f"   - Action dim: {action_dim}")
    print(f"   - Tasks: {tasks}")

    # Score with GVL
    print("\n3. Scoring with GVL reward model...")
    try:
        rewards = gvl_model.score(tasks, actions, images)
        print(f"   ✓ Rewards shape: {rewards.shape}")
        print(f"   ✓ Expected shape: ({batch_size}, {num_candidates})")
        print(f"   Rewards:\n{rewards}")

        # Check that rewards are finite
        assert torch.isfinite(rewards).all(), "Rewards contain NaN or Inf"
        print("   ✓ All rewards are finite")

        # Check shape
        assert rewards.shape == (batch_size, num_candidates), \
            f"Expected shape ({batch_size}, {num_candidates}), got {rewards.shape}"
        print("   ✓ Shape is correct")

        return True

    except Exception as e:
        print(f"   ✗ Failed to score: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_reward_models():
    """Compare GVL and QwenVL reward models."""
    print("\n" + "=" * 80)
    print("Comparing GVL vs QwenVL Reward Models")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Create models
    print("\n1. Loading both reward models...")
    try:
        gvl_model = GVLRewardModel(
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            device=device,
            dtype=dtype,
            reduction="mean",
        )
        print("   ✓ GVL model loaded")
    except Exception as e:
        print(f"   ✗ Failed to load GVL model: {e}")
        return

    try:
        qwen_model = QwenVLRewardModel(
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            device=device,
            dtype=dtype,
        )
        print("   ✓ QwenVL model loaded")
    except Exception as e:
        print(f"   ✗ Failed to load QwenVL model: {e}")
        return

    # Create data
    print("\n2. Creating test data...")
    tasks, actions, images = create_dummy_data(batch_size=1, num_candidates=3, action_dim=7)

    # Score with both models
    print("\n3. Scoring with both models...")

    print("\n   GVL Rewards (instruction-based):")
    gvl_rewards = gvl_model.score(tasks, actions, images)
    print(f"   {gvl_rewards}")
    print(f"   Mean: {gvl_rewards.mean():.4f}, Std: {gvl_rewards.std():.4f}")

    print("\n   QwenVL Rewards (action-based):")
    qwen_rewards = qwen_model.score(tasks, actions, images)
    print(f"   {qwen_rewards}")
    print(f"   Mean: {qwen_rewards.mean():.4f}, Std: {qwen_rewards.std():.4f}")

    print("\n4. Analysis:")
    print(f"   - GVL rewards are based on visual instruction-following")
    print(f"   - QwenVL rewards are based on action predictions")
    print(f"   - Both should produce higher-is-better scores")
    print(f"   - GVL rewards are typically more negative (log-likelihoods)")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("GVL Reward Model Integration Test")
    print("=" * 80)
    print()
    print("This script tests the GVL reward model integration with DePi.")
    print("It demonstrates how the reward model scores observations with tasks.")
    print()

    # Test GVL model
    success = test_gvl_reward_model()

    if success:
        print("\n✓ GVL reward model test PASSED")

        # Compare models (optional, requires more memory)
        try:
            compare_reward_models()
        except Exception as e:
            print(f"\n⚠ Model comparison skipped: {e}")
    else:
        print("\n✗ GVL reward model test FAILED")
        return 1

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
