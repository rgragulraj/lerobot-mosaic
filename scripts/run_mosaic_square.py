"""MOSAIC orchestrator — grasp → navigate square.

Phase 1 (grasp):
  Runs grasp_square ACT policy. Every GRASP_CHECK_EVERY steps (after a warmup
  window) compares the overhead frame to pre-collected done-keyframes using
  ResNet18 cosine similarity. Stops when similarity >= GRASP_SIM_THRESHOLD or
  after GRASP_MAX_STEPS.

Phase 2 (navigate):
  Runs navigate_square ACT policy for a fixed number of steps (NAVIGATE_STEPS).
  Robot holds position after completion.

Usage:
    bash scripts/run_mosaic_square.sh

Tuning:
    GRASP_SIM_THRESHOLD  — lower (0.90) if never triggers; raise (0.99) if early
    GRASP_SKIP_STEPS     — increase if robot hasn't started moving before first check
    NAVIGATE_STEPS       — increase/decrease based on how long navigation takes
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.mosaic.perception.grasp_done_detector import GraspDoneDetector
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.robots import so_follower  # noqa: F401
from lerobot.rollout import RolloutConfig, build_rollout_context
from lerobot.rollout.inference import create_inference_engine
from lerobot.rollout.strategies.core import send_next_action
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

GRASP_CHECKPOINT = Path("outputs/train/act_mosaic_grasp_square/checkpoints/020000/pretrained_model")
NAVIGATE_CHECKPOINT = Path("outputs/train/act_mosaic_navigate_square/checkpoints/020000/pretrained_model")
DONE_KEYFRAMES_DIR = Path("data/grasp_keyframes/done")

GRASP_SKIP_STEPS = 300
GRASP_CHECK_EVERY = 20
GRASP_MAX_STEPS = 600
GRASP_SIM_THRESHOLD = 0.98

NAVIGATE_STEPS = 300
FPS = 30


def _build_engine(checkpoint: Path, ctx, cfg, shutdown_event):
    """Load a policy from checkpoint and wire it to the existing hardware context."""
    print(f"Loading policy from {checkpoint} ...")
    policy_class = get_policy_class("act")
    policy = policy_class.from_pretrained(str(checkpoint))
    policy = policy.to(cfg.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    engine = create_inference_engine(
        cfg.inference,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=ctx.hardware.robot_wrapper,
        hw_features=ctx.data.hw_features,
        dataset_features=ctx.data.dataset_features,
        ordered_action_keys=ctx.data.ordered_action_keys,
        task=cfg.task or "",
        fps=cfg.fps,
        device=cfg.device,
        use_torch_compile=False,
        compile_warmup_inferences=0,
        shutdown_event=shutdown_event,
    )
    return engine


@parser.wrap()
def run(cfg: RolloutConfig):
    init_logging()

    detector = GraspDoneDetector(DONE_KEYFRAMES_DIR, threshold=GRASP_SIM_THRESHOLD)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    # Build context with grasp policy (also connects the robot)
    ctx = build_rollout_context(cfg, shutdown_event)
    control_interval = 1.0 / cfg.fps

    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    grasp_engine = ctx.policy.inference
    grasp_engine.reset()
    grasp_engine.start()
    grasp_engine.resume()

    grasp_done = False

    try:
        # ── Phase 1: Grasp ────────────────────────────────────────────────────
        print(f"\n=== PHASE 1: GRASP (max {GRASP_MAX_STEPS} steps) ===")
        print(f"Similarity checks start at step {GRASP_SKIP_STEPS}, threshold={GRASP_SIM_THRESHOLD}\n")

        for step in range(GRASP_MAX_STEPS):
            if shutdown_event.is_set():
                break

            loop_start = time.perf_counter()

            obs_raw = ctx.hardware.robot_wrapper.get_observation()
            obs_processed = ctx.processors.robot_observation_processor(obs_raw)
            send_next_action(obs_processed, obs_raw, ctx, interpolator)

            if step >= GRASP_SKIP_STEPS and step % GRASP_CHECK_EVERY == 0:
                overhead = obs_raw.get("overhead")
                if overhead is not None:
                    done, sim = detector.is_done(overhead)
                    print(f"  [step {step:3d}] similarity={sim:.3f}{'  ← DONE' if done else ''}")
                    if done:
                        grasp_done = True
                        break

            precise_sleep(max(0.0, control_interval - (time.perf_counter() - loop_start)))

        grasp_engine.stop()

        if grasp_done:
            print("\nGrasp done! Moving to navigate phase...")
        else:
            print(f"\nMax steps ({GRASP_MAX_STEPS}) reached — proceeding to navigate anyway.")

        # ── Phase 2: Navigate ─────────────────────────────────────────────────
        print(f"\n=== PHASE 2: NAVIGATE ({NAVIGATE_STEPS} steps) ===\n")

        nav_engine = _build_engine(NAVIGATE_CHECKPOINT, ctx, cfg, shutdown_event)
        nav_interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)

        # Swap ctx.policy.inference so send_next_action uses the navigate engine
        ctx.policy.inference = nav_engine
        nav_engine.reset()
        nav_engine.start()
        nav_engine.resume()

        for _step in range(NAVIGATE_STEPS):
            if shutdown_event.is_set():
                break

            loop_start = time.perf_counter()

            obs_raw = ctx.hardware.robot_wrapper.get_observation()
            obs_processed = ctx.processors.robot_observation_processor(obs_raw)
            send_next_action(obs_processed, obs_raw, ctx, nav_interpolator)

            precise_sleep(max(0.0, control_interval - (time.perf_counter() - loop_start)))

        nav_engine.stop()
        print("\nNavigate done. Holding position.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        robot = ctx.hardware.robot_wrapper.inner
        if robot.is_connected:
            input("\nPress Enter to disconnect...")
            robot.disconnect()


def main():
    run()


if __name__ == "__main__":
    main()
