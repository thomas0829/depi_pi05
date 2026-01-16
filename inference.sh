#!/bin/bash

python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick up the cube." \
  --control.repo_id=thomas0829/eval_depi_so101_test \
  --control.tags='["Inference"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30000 \
  --control.reset_time_s=300 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_so101_test/checkpoints/last/pretrained_model