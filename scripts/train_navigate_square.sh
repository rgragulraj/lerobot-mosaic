#!/bin/bash
# MOSAIC training — NAVIGATE (square)
# Run this on the H100 server from the lerobot-mosaic repo root.
# Dataset must be at /data/rrangasa/datasets/mosaic_navigate_square_20260525_104704
# (or pushed to HF Hub and the root arg removed).

lerobot-train \
    --policy.type=act \
    --dataset.repo_id=rgragulraj/mosaic_navigate_square_20260525_104704 \
    --dataset.root=/data/rrangasa/datasets/mosaic_navigate_square_20260525_104704 \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=rgragulraj/act_mosaic_navigate_square \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.dim_model=512 \
    --policy.n_heads=8 \
    --policy.dim_feedforward=3200 \
    --policy.n_encoder_layers=4 \
    --policy.n_decoder_layers=1 \
    --policy.use_vae=true \
    --policy.latent_dim=32 \
    --policy.n_vae_encoder_layers=4 \
    --policy.kl_weight=10.0 \
    --policy.dropout=0.1 \
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.optimizer_lr=1e-5 \
    --policy.optimizer_lr_backbone=1e-5 \
    --policy.optimizer_weight_decay=1e-4 \
    --batch_size=8 \
    --steps=20000 \
    --save_freq=5000 \
    --log_freq=200 \
    --eval_freq=20000 \
    --num_workers=4 \
    --prefetch_factor=4 \
    --persistent_workers=true \
    --seed=1000 \
    --output_dir=/data/rrangasa/outputs/train/act_mosaic_navigate_square \
    --job_name=act_mosaic_navigate_square \
    --wandb.enable=true \
    --wandb.project=mosaic
