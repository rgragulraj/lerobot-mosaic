#!/bin/bash
# MOSAIC rollout — INSERT (star), merged-dataset DIFFUSION policy (checkpoint 200000)
# Precondition: block already grasped and arm positioned near the star slot.
#
# Diffusion runs iterative denoising per action chunk. Trained with DDPM /
# num_train_timesteps=100 and num_inference_steps=null (=> 100 denoise steps).
# On the laptop 3050 that is very slow, so this overrides to 10 inference steps.
# Remove the --policy.num_inference_steps line to run the faithful 100-step config.

uv run lerobot-rollout \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"overhead": {"type": "opencv", "index_or_path": "/dev/video7", "fps": 30, "width": 640, "height": 480}, "gripper": {"type": "opencv", "index_or_path": "/dev/video5", "fps": 30, "width": 640, "height": 480}}' \
    --policy.path=/home/rgragulraj/lerobot-mosaic/outputs/train/diffusion_mosaic_insert_star_merged_20260624/checkpoints/200000/pretrained_model \
    --policy.num_inference_steps=10 \
    --fps=30 \
    --duration=120 \
    "$@"
