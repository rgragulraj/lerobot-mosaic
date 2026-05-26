#!/bin/bash
# Run MOSAIC grasp phase with visual done detection.
# Stops automatically when the overhead frame matches done-keyframes.
# Collect more done-keyframes first: bash scripts/grasp_keyframe_rollout.sh

uv run python scripts/run_mosaic_square.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"overhead": {"type": "opencv", "index_or_path": "/dev/video7", "fps": 30, "width": 640, "height": 480}, "gripper": {"type": "opencv", "index_or_path": "/dev/video5", "fps": 30, "width": 640, "height": 480}}' \
    --policy.path=/home/rgragulraj/lerobot-mosaic/outputs/train/act_mosaic_grasp_square/checkpoints/020000/pretrained_model \
    --duration=0
