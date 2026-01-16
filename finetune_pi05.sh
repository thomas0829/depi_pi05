#!/bin/bash

n_gpus=1

export PRETRAINED_MODEL_PATH="thomas0829/pi05-pytorch-base"

accelerate launch \
    --num_processes=${n_gpus} \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    lerobot/scripts/accelerate_train.py \
    --policy.type=pi05 \
    --dataset.repo_id="thomas0829/put_the_dolls_on_the_cloth" \
    --dataset.image_transforms.enable=true \
    --policy.max_state_dim=14 \
    --policy.max_action_dim=14 \
    --num_datasets=1500 \
    --batch_size=1 \
    --steps=5000 \
    --save_freq=2000 \
    --strict=false \
    --num_workers=4 \
    --log_freq=500 \
    --gradient_accumulation_steps=8 \
    --policy.scheduler_decay_lr=1e-5 \
    --policy.scheduler_decay_steps=1000000 \
    --policy.optimizer_lr=1e-4 \
    --dataset.use_annotated_tasks=false \
    --job_name=pi05_training \
    --wandb.enable=false