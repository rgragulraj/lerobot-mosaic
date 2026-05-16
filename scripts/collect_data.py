"""
MOSAIC data collection script with real-time keyframe annotation.

Records one episode at a time for a chosen shape, prompts for quality rating and
notes after each episode, logs metadata via DataCollectionLogger, and shows running
progress toward the 50-per-shape target.

Controls during recording:
  k          → mark keyframe (first press = KF1 grasp done, second = KF2 navigate done)
  right →    → save episode and begin reset window
  left  ←    → discard episode and re-record
  Escape     → stop session

Usage:
    uv run python scripts/collect_data.py --shape square
    uv run python scripts/collect_data.py         # interactive shape selection
    uv run python scripts/collect_data.py --shape cross --num-episodes 10
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401 — triggers registry
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.mosaic.logger import DataCollectionLogger
from lerobot.mosaic.logger.data_collection_logger import SHAPES, Shape
from lerobot.processor import make_default_processors
from lerobot.robots import (
    make_robot_from_config,
    so_follower,  # noqa: F401 — registers so101_follower
)
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators import (
    make_teleoperator_from_config,
    so_leader,  # noqa: F401 — registers so101_leader
)
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ── Hardware defaults ────────────────────────────────────────────────────────
FOLLOWER_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"
ROBOT_ID = "vellai_kunjan"
OVERHEAD_CAM_IDX = "/dev/video7"
GRIPPER_CAM_IDX = "/dev/video5"
FPS = 30
EPISODE_TIME_S = 60
RESET_TIME_S = 30
# ────────────────────────────────────────────────────────────────────────────

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


def _make_events() -> dict:
    return {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "keyframe_pressed": False,
        "keyframes": [],
    }


def _start_keyboard_listener(events: dict):
    """Start a background thread that reads raw keypresses from stdin. Works on Wayland and X11."""
    import select
    import termios
    import threading
    import tty

    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except termios.error:
        logging.warning("Cannot read terminal settings — keyboard input unavailable.")
        return None

    def read_keys():
        tty.setraw(fd)
        try:
            while not events.get("_kb_stop"):
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch == "k":
                        kf_num = len(events["keyframes"]) + 1
                        events["keyframe_pressed"] = True
                        print(f"\n  [k] Keyframe {kf_num} queued...")
                    elif ch == "\x1b":  # escape sequence start
                        # check for arrow keys: ESC [ A/B/C/D
                        if select.select([sys.stdin], [], [], 0.2)[0]:
                            ch2 = sys.stdin.read(1)
                            if ch2 == "[" and select.select([sys.stdin], [], [], 0.2)[0]:
                                ch3 = sys.stdin.read(1)
                                if ch3 == "C":
                                    print("\nRight arrow → saving episode...")
                                    events["exit_early"] = True
                                elif ch3 == "D":
                                    print("\nLeft arrow → discarding episode, will re-record...")
                                    events["rerecord_episode"] = True
                                    events["exit_early"] = True
                            else:
                                # plain ESC
                                print("\nEscape → stopping session.")
                                events["stop_recording"] = True
                                events["exit_early"] = True
                        else:
                            # plain ESC (no follow-up bytes)
                            print("\nEscape → stopping session.")
                            events["stop_recording"] = True
                            events["exit_early"] = True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    events["_kb_stop"] = False
    t = threading.Thread(target=read_keys, daemon=True)
    t.start()
    return t


@safe_stop_image_writer
def _record_episode(
    robot,
    teleop,
    dataset: LeRobotDataset,
    events: dict,
    fps: int,
    episode_time_s: int,
    single_task: str,
    display_data: bool,
) -> list[int]:
    """Run one episode recording loop. Returns list of keyframe frame indices captured."""
    events["exit_early"] = False
    events["keyframe_pressed"] = False
    events["keyframes"] = []

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    control_interval = 1.0 / fps
    frame_index = 0
    start_t = time.perf_counter()

    while (time.perf_counter() - start_t) < episode_time_s:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        act = teleop.get_action()
        act_teleop = teleop_action_processor((act, obs))
        robot_action_to_send = robot_action_processor((act_teleop, obs))
        robot.send_action(robot_action_to_send)

        action_frame = build_dataset_frame(dataset.features, act_teleop, prefix=ACTION)
        dataset.add_frame({**observation_frame, **action_frame, "task": single_task})

        if display_data:
            log_rerun_data(observation=obs_processed, action=act_teleop)

        if events["keyframe_pressed"]:
            events["keyframe_pressed"] = False
            events["keyframes"].append(frame_index)
            kf_num = len(events["keyframes"])
            label = "KF1 (grasp done)" if kf_num == 1 else "KF2 (navigate done)"
            print(f"  Keyframe {kf_num} captured at frame {frame_index}  [{label}]")

        frame_index += 1
        dt = time.perf_counter() - loop_start
        precise_sleep(max(0.0, control_interval - dt))

    return list(events["keyframes"])


def _reset_window(robot, teleop, events: dict, fps: int, reset_time_s: int):
    """Run reset window — robot moves but nothing is recorded."""
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    control_interval = 1.0 / fps
    start_t = time.perf_counter()

    while (time.perf_counter() - start_t) < reset_time_s:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        robot_observation_processor(obs)
        act = teleop.get_action()
        act_teleop = teleop_action_processor((act, obs))
        robot_action_to_send = robot_action_processor((act_teleop, obs))
        robot.send_action(robot_action_to_send)

        dt = time.perf_counter() - loop_start
        precise_sleep(max(0.0, control_interval - dt))


def dataset_episode_count(shape: str) -> int:
    import json

    info_path = Path(f"datasets/raw_{shape}/meta/info.json")
    if not info_path.exists():
        return 0
    try:
        return json.loads(info_path.read_text()).get("total_episodes", 0)
    except Exception:
        return 0


def read_last_episode_frames(shape: str) -> int | None:
    import json

    path = Path(f"datasets/raw_{shape}/meta/episodes.jsonl")
    if not path.exists():
        return None
    lines = path.read_text().strip().splitlines()
    if not lines:
        return None
    try:
        return json.loads(lines[-1]).get("length")
    except Exception:
        return None


def build_robot_config(follower_port: str, robot_id: str, overhead_cam, gripper_cam, fps: int) -> RobotConfig:
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

    def cam_cfg(idx):
        return OpenCVCameraConfig(index_or_path=idx, width=640, height=480, fps=fps)

    cfg = SO101FollowerConfig(
        port=follower_port,
        cameras={"overhead": cam_cfg(overhead_cam), "gripper": cam_cfg(gripper_cam)},
    )
    cfg.id = robot_id
    return cfg


def build_teleop_config(leader_port: str) -> TeleoperatorConfig:
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

    cfg = SO101LeaderConfig(port=leader_port)
    cfg.id = "my_leader_arm"
    return cfg


def main() -> None:
    init_logging()

    parser = argparse.ArgumentParser(description="MOSAIC data collection with live keyframe annotation")
    parser.add_argument("--shape", choices=SHAPES)
    parser.add_argument("--follower-port", default=FOLLOWER_PORT)
    parser.add_argument("--leader-port", default=LEADER_PORT)
    parser.add_argument("--robot-id", default=ROBOT_ID)
    parser.add_argument("--overhead-cam", default=OVERHEAD_CAM_IDX)
    parser.add_argument("--gripper-cam", default=GRIPPER_CAM_IDX)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--episode-time", type=int, default=EPISODE_TIME_S)
    parser.add_argument("--reset-time", type=int, default=RESET_TIME_S)
    parser.add_argument("--num-episodes", type=int, default=0)
    parser.add_argument("--operator", default="")
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

    print(f"\n{'=' * 56}")
    print(f"  MOSAIC Data Collection  —  {shape.upper()}")
    print(f"{'=' * 56}")
    print(f"  Follower : {args.follower_port}  ({args.robot_id})")
    print(f"  Leader   : {args.leader_port}")
    print(f"  Cameras  : overhead={args.overhead_cam}  gripper={args.gripper_cam}")
    print(f"  Episode  : {args.episode_time}s  |  Reset: {args.reset_time}s  |  Target: {num_episodes} eps")
    print()
    print("Hotkeys during recording:")
    print("  k          → mark keyframe (KF1 first press, KF2 second press)")
    print("  right →    → save episode")
    print("  left  ←    → discard and re-record")
    print("  Escape     → stop session")
    print()
    logger.print_progress()

    robot_cfg = build_robot_config(
        args.follower_port, args.robot_id, args.overhead_cam, args.gripper_cam, args.fps
    )
    teleop_cfg = build_teleop_config(args.leader_port)

    robot = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)

    resume = Path(f"datasets/raw_{shape}").exists()
    repo_id = f"{HF_USER}/mosaic_raw_{shape}"
    single_task = f"Sort {shape} block"

    teleop_action_processor_feat, _, robot_obs_processor = make_default_processors()

    init_rerun(session_name="mosaic_data_collection")

    robot.connect()
    teleop.connect()

    events = _make_events()
    listener = _start_keyboard_listener(events)

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor_feat,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_obs_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    if resume:
        dataset = LeRobotDataset.resume(
            repo_id,
            root=f"datasets/raw_{shape}",
            streaming_encoding=True,
            encoder_threads=4,
            image_writer_threads=4 * len(robot.cameras),
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, args.fps, dataset_features)
    else:
        dataset = LeRobotDataset.create(
            repo_id,
            args.fps,
            root=f"datasets/raw_{shape}",
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_threads=4 * len(robot.cameras),
            streaming_encoding=True,
            encoder_threads=4,
        )

    session_good = 0

    try:
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < num_episodes and not events["stop_recording"]:
                count_before = dataset_episode_count(shape)

                print(
                    f"\n─── Episode {recorded_episodes + 1}/{num_episodes}  (dataset #{count_before + 1}) ───"
                )

                log_say(f"Recording episode {dataset.num_episodes}", play_sounds=True)
                keyframes = _record_episode(
                    robot=robot,
                    teleop=teleop,
                    dataset=dataset,
                    events=events,
                    fps=args.fps,
                    episode_time_s=args.episode_time,
                    single_task=single_task,
                    display_data=True,
                )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", play_sounds=True)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    print("Episode discarded. Starting over.\n")
                    continue

                if events["stop_recording"]:
                    dataset.clear_episode_buffer()
                    break

                kf1 = keyframes[0] if len(keyframes) > 0 else 0
                kf2 = keyframes[1] if len(keyframes) > 1 else 0

                print(f"\nKeyframes captured: KF1={kf1}  KF2={kf2}")
                if len(keyframes) < 2:
                    print(
                        f"  Warning: expected 2 keyframes, got {len(keyframes)}. "
                        "Values missing will default to 0."
                    )

                dataset.save_episode()
                recorded_episodes += 1

                count_after = dataset_episode_count(shape)
                episode_id = count_after - 1
                total_frames = read_last_episode_frames(shape) or 0

                print(f"Episode {episode_id} saved  ({total_frames} frames)")

                logger.log_episode(
                    shape=shape,
                    episode_id=episode_id,
                    keyframe_1_frame=kf1,
                    keyframe_2_frame=kf2,
                    total_frames=total_frames,
                    duration_ms=total_frames * 1000 // args.fps if total_frames else args.episode_time * 1000,
                    quality="good",
                    dataset_path=f"datasets/raw_{shape}",
                )

                session_good += 1
                logger.print_progress()

                if not events["stop_recording"] and recorded_episodes < num_episodes:
                    log_say("Reset the environment", play_sounds=True)
                    _reset_window(
                        robot=robot,
                        teleop=teleop,
                        events=events,
                        fps=args.fps,
                        reset_time_s=args.reset_time,
                    )

    finally:
        log_say("Stop recording", play_sounds=True, blocking=True)
        dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()

        if listener:
            events["_kb_stop"] = True

    print(f"\nSession done. Good demos this session: {session_good}")
    logger.print_progress()


if __name__ == "__main__":
    main()
