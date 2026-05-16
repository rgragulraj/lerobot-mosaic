"""
Simple teleoperation check script.

Connects the leader and follower arms and mirrors leader motion to follower.
Press Ctrl+C to stop.

Usage:
    uv run python scripts/teleop_check.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.processor import make_default_processors
from lerobot.robots import (
    make_robot_from_config,
    so_follower,  # noqa: F401
)
from lerobot.teleoperators import (
    make_teleoperator_from_config,
    so_leader,  # noqa: F401
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FOLLOWER_PORT = "/dev/ttyACM0"
LEADER_PORT = "/dev/ttyACM1"
OVERHEAD_CAM_IDX = "/dev/video7"
GRIPPER_CAM_IDX = "/dev/video5"
FPS = 30


def main():
    init_logging()

    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig

    robot_cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        cameras={
            "overhead": OpenCVCameraConfig(index_or_path=OVERHEAD_CAM_IDX, width=640, height=480, fps=FPS),
            "gripper": OpenCVCameraConfig(index_or_path=GRIPPER_CAM_IDX, width=640, height=480, fps=FPS),
        },
    )
    robot_cfg.id = "vellai_kunjan"
    teleop_cfg = SO101LeaderConfig(port=LEADER_PORT, id="my_leader_arm")

    init_rerun(session_name="teleop_check")

    robot = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)

    robot.connect()
    teleop.connect()

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    control_interval = 1.0 / FPS
    print("Teleoperation running. Press Ctrl+C to stop.\n")

    try:
        while True:
            loop_start = time.perf_counter()

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)

            act = teleop.get_action()
            act_teleop = teleop_action_processor((act, obs))
            robot_action = robot_action_processor((act_teleop, obs))
            robot.send_action(robot_action)

            log_rerun_data(observation=obs_processed, action=act_teleop)

            precise_sleep(max(0.0, control_interval - (time.perf_counter() - loop_start)))

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()


if __name__ == "__main__":
    main()
