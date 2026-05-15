"""
MOSAIC data collection script.

Interactive wrapper around lerobot-record that:
  - Records one episode at a time for a chosen shape
  - Prompts for quality rating and notes after each episode
  - Logs metadata via DataCollectionLogger
  - Shows running progress toward the 50-per-shape target

Usage:
    uv run python scripts/collect_data.py --shape circle
    uv run python scripts/collect_data.py          # interactive shape selection

Controls during recording (lerobot-record hotkeys):
    Right arrow  →   save episode and begin reset window
    Left  arrow  ←   discard episode and re-record
    Escape           stop session early
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lerobot.mosaic.logger import DataCollectionLogger
from lerobot.mosaic.logger.data_collection_logger import SHAPES, Quality, Shape

# ── Hardware defaults (edit here or override via CLI flags) ─────────────────
FOLLOWER_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"
ROBOT_ID = "vellai_kunjan"
OVERHEAD_CAM_IDX = "/dev/video5"
GRIPPER_CAM_IDX = "/dev/video7"
FPS = 30
EPISODE_TIME_S = 60  # seconds the operator has to complete the full demo
RESET_TIME_S = 30  # seconds between episodes to reposition
# ───────────────────────────────────────────────────────────────────────────

HF_USER = "rgragulraj"


def pick_shape() -> Shape:
    print("\nWhich shape are you collecting for?")
    for i, s in enumerate(SHAPES, 1):
        print(f"  {i}. {s}")
    while True:
        raw = input("Enter number or name: ").strip().lower()
        if raw in SHAPES:
            return cast(Shape, raw)
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(SHAPES):
                return SHAPES[idx]
        except ValueError:
            pass
        print("  Invalid. Try again.")


def prompt_quality() -> Quality:
    while True:
        raw = input("  Keep this episode? [Y/n]: ").strip().lower()
        if raw in ("", "y"):
            return "good"
        if raw == "n":
            return "discard"
        print("  Enter Y or N.")


def read_last_episode_frames(shape: str) -> int | None:
    """Read the frame count of the most recently saved episode from dataset metadata."""
    episodes_path = Path(f"datasets/raw_{shape}/meta/episodes.jsonl")
    if not episodes_path.exists():
        return None
    lines = episodes_path.read_text().strip().splitlines()
    if not lines:
        return None
    try:
        last = json.loads(lines[-1])
        return last.get("length")
    except Exception:
        return None


def dataset_episode_count(shape: str) -> int:
    info_path = Path(f"datasets/raw_{shape}/meta/info.json")
    if not info_path.exists():
        return 0
    try:
        return json.loads(info_path.read_text()).get("total_episodes", 0)
    except Exception:
        return 0


def build_record_cmd(
    shape: str,
    follower_port: str,
    leader_port: str,
    robot_id: str,
    overhead_cam: int,
    gripper_cam: int,
    fps: int,
    episode_time_s: int,
    reset_time_s: int,
    resume: bool,
) -> list[str]:
    # YAML-style dict — matches the format expected by draccus/OmegaConf CLI parsing.
    # index_or_path is quoted when it's a device path string (e.g. /dev/video5).
    def cam_path(v) -> str:
        return f'"{v}"' if isinstance(v, str) else str(v)

    cameras = (
        f"{{overhead: {{type: opencv, index_or_path: {cam_path(overhead_cam)},"
        f" width: 640, height: 480, fps: {fps}}},"
        f" gripper: {{type: opencv, index_or_path: {cam_path(gripper_cam)},"
        f" width: 640, height: 480, fps: {fps}}}}}"
    )
    cmd = [
        "uv",
        "run",
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={robot_id}",
        f"--robot.cameras={cameras}",
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--dataset.repo_id={HF_USER}/mosaic_raw_{shape}",
        f"--dataset.root=datasets/raw_{shape}",
        "--dataset.num_episodes=1",
        f"--dataset.single_task=Sort {shape} block",
        f"--dataset.fps={fps}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={reset_time_s}",
        "--dataset.push_to_hub=false",
        "--dataset.video=true",
        "--dataset.vcodec=auto",  # auto-selects best available hardware encoder
        "--dataset.streaming_encoding=true",  # encode frames in real-time → save_episode() is near-instant
        "--dataset.encoder_threads=4",
        "--dataset.num_image_writer_threads_per_camera=4",
        "--display_data=true",
    ]
    if resume:
        cmd.append("--resume=true")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="MOSAIC data collection")
    parser.add_argument("--shape", choices=SHAPES)
    parser.add_argument("--follower-port", default=FOLLOWER_PORT)
    parser.add_argument("--leader-port", default=LEADER_PORT)
    parser.add_argument("--robot-id", default=ROBOT_ID)
    parser.add_argument("--overhead-cam", default=OVERHEAD_CAM_IDX)
    parser.add_argument("--gripper-cam", default=GRIPPER_CAM_IDX)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--episode-time", type=int, default=EPISODE_TIME_S)
    parser.add_argument("--reset-time", type=int, default=RESET_TIME_S)
    parser.add_argument(
        "--num-episodes", type=int, default=0, help="Episodes to record this session (prompted if omitted)"
    )
    parser.add_argument("--operator", default="", help="Your name (logged per session)")
    args = parser.parse_args()

    shape: Shape = args.shape or pick_shape()

    if args.num_episodes > 0:
        num_episodes = args.num_episodes
    else:
        while True:
            raw = input("How many episodes to record this session? ").strip()
            if raw.isdigit() and int(raw) > 0:
                num_episodes = int(raw)
                break
            print("  Enter a positive integer.")
    logger = DataCollectionLogger(operator=args.operator)

    print(f"\n{'=' * 52}")
    print(f"  MOSAIC Data Collection  —  {shape.upper()}")
    print(f"{'=' * 52}")
    print(f"  Follower : {args.follower_port}  ({args.robot_id})")
    print(f"  Leader   : {args.leader_port}")
    print(f"  Cameras  : overhead={args.overhead_cam}  gripper={args.gripper_cam}")
    print(
        f"  Episode  : {args.episode_time}s  |  Reset: {args.reset_time}s  |  Target: {num_episodes} episodes"
    )
    print()
    logger.print_progress()

    print("Hotkeys during recording:")
    print("  →  save episode      ←  discard & re-record      Esc  end session")
    print()

    session_good = 0

    for ep_num in range(1, num_episodes + 1):
        count_before = dataset_episode_count(shape)

        print(f"─── Episode {ep_num}/{num_episodes}  (dataset #{count_before + 1}) ──────────────────")
        block_pos = input("Block position (e.g. 'left-center', 'far-right'): ").strip()
        notes_pre = input("Pre-recording notes (Enter to skip): ").strip()

        resume = Path(f"datasets/raw_{shape}").exists()
        cmd = build_record_cmd(
            shape=shape,
            follower_port=args.follower_port,
            leader_port=args.leader_port,
            robot_id=args.robot_id,
            overhead_cam=args.overhead_cam,
            gripper_cam=args.gripper_cam,
            fps=args.fps,
            episode_time_s=args.episode_time,
            reset_time_s=args.reset_time,
            resume=resume,
        )

        print()
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break

        count_after = dataset_episode_count(shape)
        if count_after <= count_before:
            print("No episode saved — skipping log entry.\n")
            cont = input("Try again? [Y/n]: ").strip().lower()
            if cont == "n":
                break
            continue

        episode_id = count_after - 1
        total_frames = read_last_episode_frames(shape) or 0

        print(f"\nEpisode {episode_id} saved  ({total_frames} frames).  Rate this demo:")
        quality = prompt_quality()

        print("  Keyframe frame indices (press Enter to skip):")
        raw_kf1 = input("    KF1 — grasp done (arm at carry height): ").strip()
        raw_kf2 = input("    KF2 — navigate done (arm above slot):   ").strip()
        kf1 = int(raw_kf1) if raw_kf1.isdigit() else 0
        kf2 = int(raw_kf2) if raw_kf2.isdigit() else 0

        notes_post = input("  Post-recording notes (Enter to skip): ").strip()
        notes = " | ".join(filter(None, [notes_pre, notes_post]))

        logger.log_episode(
            shape=shape,
            episode_id=episode_id,
            keyframe_1_frame=kf1,
            keyframe_2_frame=kf2,
            total_frames=total_frames,
            duration_ms=total_frames * 1000 // args.fps if total_frames else args.episode_time * 1000,
            block_position=block_pos,
            quality=quality,
            notes=notes,
            dataset_path=f"datasets/raw_{shape}",
        )

        if quality == "good":
            session_good += 1

        logger.print_progress()

        if ep_num < num_episodes:
            cont = input("Continue to next episode? [Y/n]: ").strip().lower()
            if cont == "n":
                print("Stopping early.")
                break
            print()

    print(f"\nSession done. Good demos this session: {session_good}")
    logger.print_progress()


if __name__ == "__main__":
    main()
