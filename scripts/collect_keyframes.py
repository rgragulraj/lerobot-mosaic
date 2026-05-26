"""Run the grasp policy and capture keyframes for orchestrator training.

No display needed — watch the physical robot.
Press 'k' when the block is lifted to save overhead + gripper frames.
Press 'q' or Ctrl+C to stop.

Usage:
    bash scripts/grasp_keyframe_rollout.sh

Output:
    data/grasp_keyframes/done/NNNN_overhead.jpg
    data/grasp_keyframes/done/NNNN_gripper.jpg
"""

import select
import sys
import termios
import time
import tty
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import so_follower  # noqa: F401
from lerobot.rollout import RolloutConfig, build_rollout_context
from lerobot.rollout.strategies.core import send_next_action
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

OUTPUT_DIR = Path("data/grasp_keyframes/done")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _keypress() -> str | None:
    """Non-blocking single-keypress read. Terminal must be in cbreak mode."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


@parser.wrap()
def collect(cfg: RolloutConfig):
    init_logging()

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    ctx = build_rollout_context(cfg, shutdown_event)

    # Resume numbering from any previously saved keyframes
    existing = sorted(OUTPUT_DIR.glob("*_overhead.jpg"))
    kf_count = int(existing[-1].stem.split("_")[0]) + 1 if existing else 0
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print(f"Starting at keyframe #{kf_count:04d}")
    print("Controls: k = save keyframe   q = quit\n")

    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    engine = ctx.policy.inference
    engine.reset()
    engine.start()
    engine.resume()

    control_interval = 1.0 / cfg.fps

    # cbreak: per-character input without requiring Enter; keeps Ctrl+C as SIGINT
    old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while not shutdown_event.is_set():
            loop_start = time.perf_counter()

            obs_raw = ctx.hardware.robot_wrapper.get_observation()
            obs_processed = ctx.processors.robot_observation_processor(obs_raw)

            send_next_action(obs_processed, obs_raw, ctx, interpolator)

            key = _keypress()

            if key == "k":
                overhead = obs_raw.get("overhead")
                gripper = obs_raw.get("gripper")
                if overhead is not None:
                    name = f"{kf_count:04d}"
                    # Camera returns RGB; cv2.imwrite expects BGR
                    ok_oh = cv2.imwrite(
                        str(OUTPUT_DIR / f"{name}_overhead.jpg"), cv2.cvtColor(overhead, cv2.COLOR_RGB2BGR)
                    )
                    ok_gr = (
                        cv2.imwrite(
                            str(OUTPUT_DIR / f"{name}_gripper.jpg"), cv2.cvtColor(gripper, cv2.COLOR_RGB2BGR)
                        )
                        if gripper is not None
                        else False
                    )
                    if ok_oh:
                        print(f"  Saved keyframe {kf_count:04d}  (gripper={'ok' if ok_gr else 'missing'})")
                        kf_count += 1
                    else:
                        print(f"  ERROR: cv2.imwrite failed for keyframe {kf_count:04d}")
                else:
                    print("  WARNING: no overhead frame in observation")

            elif key == "q":
                break

            precise_sleep(max(0.0, control_interval - (time.perf_counter() - loop_start)))

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        engine.stop()
        robot = ctx.hardware.robot_wrapper.inner
        if robot.is_connected:
            robot.disconnect()

    print(f"\nDone — {kf_count} keyframes saved to {OUTPUT_DIR.resolve()}")


def main():
    collect()


if __name__ == "__main__":
    main()
