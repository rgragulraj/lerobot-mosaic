#!/bin/bash
# MOSAIC training — INSERT (star) with Diffusion Policy — high-quality run on merged dataset
# Run this on the H100 server from the lerobot-mosaic repo root.
# Dataset must be at /data/rrangasa/datasets/mosaic_insert_star_merged

lerobot-train \
    --policy.type=diffusion \
    --dataset.repo_id=rgragulraj/mosaic_insert_star \
    --dataset.root=/data/rrangasa/datasets/mosaic_insert_star_merged \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=rgragulraj/diffusion_mosaic_insert_star_merged \
    --policy.vision_backbone=resnet34 \
    --policy.pretrained_backbone_weights=ResNet34_Weights.IMAGENET1K_V1 \
    --policy.n_obs_steps=2 \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --policy.optimizer_lr=5e-5 \
    --policy.optimizer_weight_decay=1e-6 \
    --batch_size=32 \
    --steps=200000 \
    --save_freq=20000 \
    --log_freq=200 \
    --eval_freq=200000 \
    --num_workers=4 \
    --prefetch_factor=4 \
    --persistent_workers=true \
    --seed=1000 \
    --output_dir=/data/rrangasa/outputs/train/diffusion_mosaic_insert_star_merged_20260624 \
    --job_name=diffusion_mosaic_insert_star_merged_20260624 \
    --wandb.enable=false
