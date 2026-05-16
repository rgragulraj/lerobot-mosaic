#!/bin/bash
# MOSAIC data collection — SQUARE
# Controls: right arrow = save, left arrow = redo, ESC = finish

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"overhead": {"type": "opencv", "index_or_path": "/dev/video7", "fps": 30, "width": 640, "height": 480}, "gripper": {"type": "opencv", "index_or_path": "/dev/video5", "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/mosaic_raw_square \
    --dataset.single_task="Sort square block into correct slot" \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.push_to_hub=false \
    --display_data=true
