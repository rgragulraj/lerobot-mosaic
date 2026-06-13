#!/bin/bash
# MOSAIC data collection — GRASP (star)
# Start: arm at home, gripper open, star block on table
# End:   block grasped, arm lifted slightly — press right arrow
# Controls: right arrow = save, left arrow = redo, ESC = finish

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"overhead": {"type": "opencv", "index_or_path": "/dev/video7", "fps": 30, "width": 640, "height": 480}, "gripper": {"type": "opencv", "index_or_path": "/dev/video5", "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/mosaic_grasp_star \
    --dataset.single_task="Grasp the star block" \
    --dataset.num_episodes=25 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=20 \
    --dataset.push_to_hub=false \
    --display_data=true
