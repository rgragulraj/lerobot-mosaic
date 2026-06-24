#!/bin/bash
# MOSAIC training — INSERT (star) with Diffusion Policy
# Run this on the H100 server from the lerobot-mosaic repo root.
# Dataset must be at /data/rrangasa/datasets/mosaic_insert_star_20260612_191647

lerobot-train \
    --policy.type=diffusion \
    --dataset.repo_id=rgragulraj/mosaic_insert_star \
    --dataset.root=/data/rrangasa/datasets/mosaic_insert_star_20260612_191647 \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=rgragulraj/diffusion_mosaic_insert_star \
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_weight_decay=1e-6 \
    --batch_size=64 \
    --steps=80000 \
    --save_freq=10000 \
    --log_freq=200 \
    --eval_freq=80000 \
    --num_workers=4 \
    --prefetch_factor=4 \
    --persistent_workers=true \
    --seed=1000 \
    --output_dir=/data/rrangasa/outputs/train/diffusion_mosaic_insert_star_20260623 \
    --job_name=diffusion_mosaic_insert_star_20260623 \
    --wandb.enable=false
