#!/bin/bash
# Collect star grasp done-keyframes.
# Press 'k' when the star block is lifted to save that frame.
# Press 'q' to stop.
# Keyframes are saved to data/grasp_keyframes_star/done/

uv run python scripts/collect_keyframes.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"overhead": {"type": "opencv", "index_or_path": "/dev/video7", "fps": 30, "width": 640, "height": 480}, "gripper": {"type": "opencv", "index_or_path": "/dev/video5", "fps": 30, "width": 640, "height": 480}}' \
    --policy.path=/home/rgragulraj/lerobot-mosaic/outputs/train/act_mosaic_grasp_star/checkpoints/020000/pretrained_model \
    --duration=0 \
    --output-dir=data/grasp_keyframes_star/done
