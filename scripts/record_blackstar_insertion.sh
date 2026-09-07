#!/bin/bash
# MOSAIC data collection — INSERT ONLY (black star)
#
# Scope: the final insertion motion only. The arm starts already holding the
# black star block, positioned right above the star slot.
#
# Per-episode reset:
#   1. Pull the block out of the slot and re-seat it in the gripper.
#   2. Return the arm to the start pose:
#        uv run python scripts/go_to_start_position.py --name insert_above_slot
#   3. Press right arrow to begin the next episode.
#
# End state: block seated flush in the slot, gripper released — press right arrow.
# Controls: right arrow = save, left arrow = redo, ESC = finish

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"overhead": {"type": "opencv", "index_or_path": "/dev/video7", "fps": 30, "width": 640, "height": 480}, "gripper": {"type": "opencv", "index_or_path": "/dev/video5", "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/blackstarinsertion \
    --dataset.root="${HOME}/.cache/huggingface/lerobot/rgragulraj/blackstarinsertion" \
    --dataset.single_task="Insert the black star block into the slot" \
    --dataset.num_episodes=25 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=30 \
    --dataset.push_to_hub=false \
    --display_data=true
