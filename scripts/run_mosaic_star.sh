#!/bin/bash
# Run MOSAIC star pipeline: grasp → navigate.
# Collect star grasp done-keyframes first into data/grasp_keyframes_star/done/

python scripts/run_mosaic_star.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"overhead": {"type": "opencv", "index_or_path": "/dev/video7", "fps": 30, "width": 640, "height": 480}, "gripper": {"type": "opencv", "index_or_path": "/dev/video5", "fps": 30, "width": 640, "height": 480}}' \
    --policy.path=/home/rgragulraj/lerobot-mosaic/outputs/train/act_mosaic_grasp_star/checkpoints/020000/pretrained_model \
    --duration=0
