#!/bin/bash
# MOSAIC data collection — NAVIGATE (star)
# Start: block grasped, arm lifted at grasp position
# End:   arm positioned over the star slot, ready to insert — press right arrow
# Controls: right arrow = save, left arrow = redo, ESC = finish

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"overhead": {"type": "opencv", "index_or_path": "/dev/video7", "fps": 30, "width": 640, "height": 480}, "gripper": {"type": "opencv", "index_or_path": "/dev/video5", "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/mosaic_navigate_star \
    --dataset.single_task="Navigate the star block to the insertion slot" \
    --dataset.num_episodes=25 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.push_to_hub=false \
    --display_data=true \
    --dataset.root=/home/rgragulraj/.cache/huggingface/lerobot/rgragulraj/mosaic_navigate_star_20260612_170707 \
    --resume=true
