#!/bin/bash
# MOSAIC training — INSERT ONLY (black star) with ACT
#
# Uses LeRobot's DEFAULT ACT hyperparameters — no policy config overrides.
# Defaults (from src/lerobot/policies/act/configuration_act.py and configs/train.py):
#   chunk_size=100  n_action_steps=100  n_obs_steps=1
#   vision_backbone=resnet18  dim_model=512  use_vae=true  latent_dim=32
#   optimizer_lr=1e-5  batch_size=8  steps=100000  save_freq=20000
#
# Run this on the H100 server from the lerobot-mosaic repo root.
# Dataset must be at /data/rrangasa/datasets/blackstarinsertion

lerobot-train \
    --policy.type=act \
    --dataset.repo_id=rgragulraj/blackstarinsertion \
    --dataset.root=/data/rrangasa/datasets/blackstarinsertion \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=rgragulraj/act_blackstarinsertion \
    --output_dir=/data/rrangasa/outputs/train/act_blackstarinsertion \
    --job_name=act_blackstarinsertion \
    --wandb.enable=false
