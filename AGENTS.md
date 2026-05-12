# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                           # All tests
uv run pytest tests/policies/test_act.py -svv -k "test_name"   # Single test or file
DEVICE=cuda make test-end-to-end                                # All E2E tests (act, diffusion, tdmpc, smolvla)
pre-commit run --all-files                                      # Lint + format (ruff, typos, bandit, etc.)
```

Training and evaluation (all via `uv run` or after `uv sync`):

```bash
lerobot-train --policy.type=act --dataset.repo_id=<HF_REPO> --policy.device=cuda --output_dir=outputs/train/my_run
lerobot-train --config_path=<checkpoint>/train_config.json --resume=true   # Resume from checkpoint
lerobot-eval  --policy.path=<checkpoint>/pretrained_model --env.type=pusht --eval.n_episodes=50
lerobot-rollout --policy.path=<checkpoint> --robot.type=so101_follower ...  # One-shot rollout
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points mapped in `pyproject.toml [project.scripts]`: `lerobot-train`, `lerobot-eval`, `lerobot-record`, `lerobot-replay`, `lerobot-teleoperate`, `lerobot-calibrate`, `lerobot-rollout`, `lerobot-edit-dataset`, `lerobot-info`, `lerobot-find-port`, `lerobot-find-cameras`, `lerobot-find-joint-limits`, `lerobot-setup-motors`, `lerobot-setup-can`, `lerobot-dataset-viz`, `lerobot-imgtransform-viz`, `lerobot-train-tokenizer`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level for IL). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir: `act`, `diffusion`, `smolvla`, `pi0`, `pi05`, `pi0_fast`, `wall_x`, `xvla`, `tdmpc`, `vqbet`, `eo1`, `groot`, `rtc`, `gaussian_actor`, `multi_task_dit`. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. The two required abstract methods are `forward()` and `select_action()`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`rl/`** — Online RL stack (SAC implemented). `train_rl.py` is the entry point; config is `TrainRLServerPipelineConfig` (extends `TrainPipelineConfig`). Actor/learner processes communicate over `transport/`. `rewards/` holds reward model training and classifiers.
- **`model/`** — Shared model primitives (kinematics, classifiers) used across policies and the RL stack.
- **`common/`** — Cross-cutting helpers that may import from `policies`, `processor`, `configs`: `control_utils`, `train_utils`, `wandb_utils`. Not re-exported from the top-level package.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators (`require_cuda`, `require_hf_token`, `require_env`, etc.) from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt`.
- **`docker/`** — `Dockerfile.user`, `Dockerfile.internal`, plus per-benchmark Dockerfiles (`Dockerfile.benchmark.*`) for LIBERO, MetaWorld, RoboCasa, RoboCerebra, RoboMME, RoboTwin, VLABench.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`.

## Notes

- **Prioritize `uv run`** to execute Python commands (not raw `python` or `pip`).
- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[feetech]`, `lerobot[aloha]`, `lerobot[smolvla]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: `LeRobotDataset` handles frame extraction from video files. Tests need ffmpeg installed.
- **PR conventions**: rebase on `upstream/main`, use a descriptive branch (don't work on `main`), run `pre-commit` and tests locally before submitting. Community review policy: review one other PR before yours receives attention.
- **Benchmark envs** have native dependencies that are painful to install locally — use the pre-baked `docker/Dockerfile.benchmark.*` images for `lerobot-eval` runs.
