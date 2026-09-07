#!/bin/bash
# MOSAIC rollout — INSERT ONLY (black star), ACT policy (checkpoint 100000)
# Tuned for MAXIMUM control-loop Hz on the laptop (RTX 3050 6 GB).
#
# Precondition: black star block already grasped, arm at the insert start pose:
#   uv run python scripts/go_to_start_position.py --name blackstarinsert
#
# Speed levers applied here:
#   - overhead cam forced to MJPG (compressed) — /dev/video7 supports it, the
#     recording default (YUYV) was ~18 MB/s and throttled the loop
#   - NO --display_data: rerun image logging is the biggest per-tick cost and
#     ACT inference is amortized (model runs once per 100 ticks, chunk is cached)
#   - --policy.use_amp=true: FP16 autocast, halves the periodic inference spike
#   - --use_torch_compile=true: compiles predict_action_chunk (one-time ~1 min)
#   - TF32 on, thread caps, expandable CUDA allocator
#
# Do NOT raise --fps above 30: the cached action chunk is timed for 1/30 s steps,
# so a faster loop replays the trajectory too fast. "Max Hz" here = reliably
# reaching the trained 30 Hz. To smooth the robot beyond that, add
# `--interpolation_multiplier=2` (only helps if the loop already holds ~60 Hz).
#
# Ctrl+C to stop. Arm returns to its initial pose on exit.

export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=/home/rgragulraj/lerobot-mosaic/outputs/train/act_blackstarinsertion/checkpoints/100000/pretrained_model
# To pull from the Hub instead: CKPT=rgragulraj/act_blackstarinsertion

uv run lerobot-rollout \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"overhead": {"type": "opencv", "index_or_path": "/dev/video7", "fps": 30, "width": 640, "height": 480, "fourcc": "MJPG", "warmup_s": 2}, "gripper": {"type": "opencv", "index_or_path": "/dev/video5", "fps": 30, "width": 640, "height": 480, "warmup_s": 2}}' \
    --policy.path="${CKPT}" \
    --policy.device=cuda \
    --policy.use_amp=true \
    --use_torch_compile=true \
    --torch_compile_mode=default \
    --task="Insert the black star block into the slot" \
    --fps=30 \
    --duration=30 \
    "$@"
