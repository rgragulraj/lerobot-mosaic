#!/bin/bash
# MOSAIC rollout — INSERT (star), merged-dataset ACT policy (checkpoint 100000)
# Precondition: block already grasped and arm positioned near the star slot
# (this policy only covers the final insert motion, not grasp/navigate).

uv run lerobot-rollout \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"overhead": {"type": "opencv", "index_or_path": "/dev/video7", "fps": 30, "width": 640, "height": 480}, "gripper": {"type": "opencv", "index_or_path": "/dev/video5", "fps": 30, "width": 640, "height": 480}}' \
    --policy.path=/home/rgragulraj/lerobot-mosaic/outputs/train/act_mosaic_insert_star_merged_20260625/checkpoints/100000/pretrained_model \
    --fps=30 \
    --duration=20 \
    "$@"
