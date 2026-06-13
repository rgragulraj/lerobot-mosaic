"""MOSAIC orchestrator — grasp → navigate → insert (star).

All three phases run for a fixed number of steps with no vision-based termination.

Phase 1 (grasp):   GRASP_STEPS fixed steps.
Phase 2 (navigate): NAVIGATE_STEPS fixed steps.
Phase 3 (insert):   INSERT_STEPS fixed steps. Robot holds position after completion.

Usage:
    bash scripts/run_mosaic_star.sh

Tuning:
    GRASP_STEPS    — increase/decrease based on how long grasping takes
    NAVIGATE_STEPS — increase/decrease based on how long navigation takes
    INSERT_STEPS   — increase/decrease based on how long insertion takes
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.robots import so_follower  # noqa: F401
from lerobot.rollout import RolloutConfig, build_rollout_context
from lerobot.rollout.inference import create_inference_engine
from lerobot.rollout.strategies.core import send_next_action
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

GRASP_CHECKPOINT = Path("outputs/train/act_mosaic_grasp_star/checkpoints/020000/pretrained_model")
NAVIGATE_CHECKPOINT = Path("outputs/train/act_mosaic_navigate_star/checkpoints/020000/pretrained_model")
INSERT_CHECKPOINT = Path("outputs/train/act_mosaic_insert_star/checkpoints/020000/pretrained_model")

GRASP_STEPS = 400
NAVIGATE_STEPS = 300
INSERT_STEPS = 200
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

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    ctx = build_rollout_context(cfg, shutdown_event)
    control_interval = 1.0 / cfg.fps

    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    grasp_engine = ctx.policy.inference
    grasp_engine.reset()
    grasp_engine.start()
    grasp_engine.resume()

    try:
        # ── Phase 1: Grasp ────────────────────────────────────────────────────
        print(f"\n=== PHASE 1: GRASP ({GRASP_STEPS} steps) ===\n")

        for _step in range(GRASP_STEPS):
            if shutdown_event.is_set():
                break

            loop_start = time.perf_counter()

            obs_raw = ctx.hardware.robot_wrapper.get_observation()
            obs_processed = ctx.processors.robot_observation_processor(obs_raw)
            send_next_action(obs_processed, obs_raw, ctx, interpolator)

            precise_sleep(max(0.0, control_interval - (time.perf_counter() - loop_start)))

        grasp_engine.stop()
        print("\nGrasp done. Moving to navigate phase...")

        # ── Phase 2: Navigate ─────────────────────────────────────────────────
        print(f"\n=== PHASE 2: NAVIGATE ({NAVIGATE_STEPS} steps) ===\n")

        nav_engine = _build_engine(NAVIGATE_CHECKPOINT, ctx, cfg, shutdown_event)
        nav_interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)

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
        print("\nNavigate done. Moving to insert phase...")

        # ── Phase 3: Insert ───────────────────────────────────────────────────
        print("\n=== PHASE 3: INSERT (running until Ctrl+C) ===\n")

        insert_engine = _build_engine(INSERT_CHECKPOINT, ctx, cfg, shutdown_event)
        insert_interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)

        ctx.policy.inference = insert_engine
        insert_engine.reset()
        insert_engine.start()
        insert_engine.resume()

        while not shutdown_event.is_set():
            loop_start = time.perf_counter()

            obs_raw = ctx.hardware.robot_wrapper.get_observation()
            obs_processed = ctx.processors.robot_observation_processor(obs_raw)
            send_next_action(obs_processed, obs_raw, ctx, insert_interpolator)

            precise_sleep(max(0.0, control_interval - (time.perf_counter() - loop_start)))

        insert_engine.stop()
        print("\nInsert stopped.")

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
