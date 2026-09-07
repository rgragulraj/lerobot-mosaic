#!/bin/bash
# MOSAIC training — INSERT (star) with ACT — high-quality run on merged 50-episode dataset
# Run this on the H100 server from the lerobot-mosaic repo root.
# Dataset must be at /data/rrangasa/datasets/mosaic_insert_star_merged

CUDA_VISIBLE_DEVICES=1 lerobot-train \
    --policy.type=act \
    --dataset.repo_id=rgragulraj/mosaic_insert_star \
    --dataset.root=/data/rrangasa/datasets/mosaic_insert_star_merged \
    --policy.device=cuda \
    --policy.push_to_hub=true \
    --policy.repo_id=rgragulraj/act_mosaic_insert_star_merged \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.dim_model=512 \
    --policy.n_heads=8 \
    --policy.dim_feedforward=3200 \
    --policy.n_encoder_layers=4 \
    --policy.n_decoder_layers=1 \
    --policy.use_vae=true \
    --policy.latent_dim=64 \
    --policy.n_vae_encoder_layers=4 \
    --policy.kl_weight=10.0 \
    --policy.dropout=0.1 \
    --policy.vision_backbone=resnet34 \
    --policy.pretrained_backbone_weights=ResNet34_Weights.IMAGENET1K_V1 \
    --policy.optimizer_lr=1e-5 \
    --policy.optimizer_lr_backbone=1e-5 \
    --policy.optimizer_weight_decay=1e-4 \
    --batch_size=16 \
    --steps=100000 \
    --save_freq=10000 \
    --log_freq=200 \
    --eval_freq=100000 \
    --num_workers=4 \
    --prefetch_factor=4 \
    --persistent_workers=true \
    --seed=1000 \
    --output_dir=/data/rrangasa/outputs/train/act_mosaic_insert_star_merged_20260625 \
    --job_name=act_mosaic_insert_star_merged_20260625 \
    --wandb.enable=false
