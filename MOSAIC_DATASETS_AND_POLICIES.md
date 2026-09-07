# MOSAIC — Datasets & Policies Reference

Inventory of every dataset and trained policy/checkpoint used by the MOSAIC
shape-sorting stack in this repo.

- **Snapshot:** 2026-09-02 (branch `main`, commit `c61a6884`)
- **Namespace:** all HF repo ids under `rgragulraj/`
- **Local dataset cache:** `~/.cache/huggingface/lerobot/rgragulraj/<dir>`
- **Server copy:** ASU SCAI H100, `/data/rrangasa/datasets/` and `/data/rrangasa/outputs/train/`
- **Local training outputs:** `outputs/train/<run>/`

---

## 1. Common recording / data format

Every MOSAIC dataset was recorded with `lerobot-record` (scripts in `scripts/record_*.sh`)
on the SO-101 arm and shares this schema:

| Field                          | Value                                                                                                                       |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `robot_type`                   | `so_follower` (SO-101 follower, Feetech STS3215)                                                                            |
| `fps`                          | 30                                                                                                                          |
| `action` / `observation.state` | `float32[6]` — `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos` |
| `observation.images.overhead`  | video `480×640×3`, AV1, yuv420p, 30 fps (overhead cam `/dev/video7`)                                                        |
| `observation.images.gripper`   | video `480×640×3`, AV1, yuv420p, 30 fps (wrist cam `/dev/video5`)                                                           |

Recording hardware: follower `--robot.port=/dev/ttyACM0 --robot.id=vellai_kunjan`,
leader `--teleop.port=/dev/ttyACM1 --teleop.id=my_leader_arm`. All record scripts
set `--dataset.push_to_hub=false`, so the datasets live locally / on the server, not
on the Hub. CLAHE (LAB L-channel) normalization is applied to camera frames at both
train and inference time (see `CLAUDE.md` → MOSAIC).

---

## 2. Datasets

### 2.1 Task datasets (used to train the six policies)

| Local dir                                | repo_id (as used)                                   | Phase / shape     | Episodes | Frames | `single_task`                                     | Record script                            |
| ---------------------------------------- | --------------------------------------------------- | ----------------- | -------- | ------ | ------------------------------------------------- | ---------------------------------------- |
| `mosaic_grasp_square_20260515_161734`    | `rgragulraj/mosaic_grasp_square`                    | grasp / square    | 30       | 15,626 | "Grasp the square block"                          | `record_grasp_square.sh`                 |
| `mosaic_grasp_star_20260612_152207`      | `rgragulraj/mosaic_grasp_star`                      | grasp / star      | 25       | 13,636 | "Grasp the star block"                            | `record_grasp_star.sh`                   |
| `mosaic_navigate_square_20260525_104704` | `rgragulraj/mosaic_navigate_square_20260525_104704` | navigate / square | 30       | 9,049  | "Navigate the square block to the insertion slot" | `record_navigate_square.sh`              |
| `mosaic_navigate_star_20260612_170707`   | `rgragulraj/mosaic_navigate_star`                   | navigate / star   | 27       | 6,858  | "Navigate the star block to the insertion slot"   | `record_navigate_star.sh`                |
| `mosaic_insert_square_20260612_130337`   | `rgragulraj/mosaic_insert_square`                   | insert / square   | 25       | 8,227  | "Insert the square block into the slot"           | `record_insert.sh`                       |
| `mosaic_insert_star_20260612_191647`     | `rgragulraj/mosaic_insert_star`                     | insert / star     | 50       | 26,678 | "Insert the star block into the slot"             | `record_insert_star.sh`                  |
| `mosaic_insert_star_merged`              | `rgragulraj/mosaic_insert_star` (merged root)       | insert / star     | 50       | 26,678 | (merged)                                          | manual merge of the insert-star captures |

Notes:

- `record_navigate_square.sh` writes `--dataset.repo_id=rgragulraj/mosaic_navigate_square`,
  but training (`train_navigate_square.sh`) points at the timestamped dir/repo id above.
- `record_navigate_star.sh` requested 25 episodes and was `--resume=true`d to 27.
- `record_insert_star.sh` is `--resume=true` against
  `mosaic_insert_star_20260612_191647`. A 1-episode / 211-frame side capture
  `mosaic_insert_star_20260622_164220` also exists in the cache.
- `mosaic_insert_star_merged` is the consolidated 50-episode set that the
  high-quality ACT and Diffusion insert-star runs train on
  (`--dataset.repo_id=rgragulraj/mosaic_insert_star --dataset.root=.../mosaic_insert_star_merged`).

### 2.2 Raw / exploratory captures (not used by the shipped policies)

Present in the local cache, kept for reference only:

- `mosaic_raw_square_20260515_154456 … _160507` — early end-to-end "sort square into
  correct slot" captures (`record_square.sh`, `rgragulraj/mosaic_raw_square`), mostly
  1 episode each (e.g. `_160507` = 1 ep / 1,116 frames).
- Non-MOSAIC leftovers from earlier projects: `LENS1_square`, `lenslab_square_pickplace`,
  `lenslab_block_insert_test`, `eval_lenslab_square_pickplace`, `eval_throwaway`,
  `policy1_diverse_session_{a,b,c}`, `policy1_diverse_all`.

---

## 3. Policies / models

All trained with `lerobot-train` on the H100 server; checkpoints synced to
`outputs/train/<run>/checkpoints/<step>/pretrained_model/`. Training scripts set
`--policy.push_to_hub=true` with the repo ids below (Hub push not verified for every run).

### 3.1 ACT policies — standard 20k-step runs

Shared config (`train_grasp_star.sh`, `train_navigate_*.sh`, `train_insert_*.sh`):

```
policy.type=act   chunk_size=100   n_action_steps=100   n_obs_steps=1
dim_model=512   n_heads=8   dim_feedforward=3200
n_encoder_layers=4   n_decoder_layers=1
use_vae=true   latent_dim=32   n_vae_encoder_layers=4   kl_weight=10.0   dropout=0.1
vision_backbone=resnet18 (ResNet18_Weights.IMAGENET1K_V1)
optimizer_lr=1e-5   optimizer_lr_backbone=1e-5   weight_decay=1e-4
batch_size=8   steps=20000   save_freq=5000   seed=1000
```

Checkpoints: `005000, 010000, 015000, 020000, last`. Weights ≈ 198 MB (`model.safetensors`).

| Output dir                   | HF repo_id                              | Phase / shape     | Training dataset                         | Train script                                |
| ---------------------------- | --------------------------------------- | ----------------- | ---------------------------------------- | ------------------------------------------- |
| `act_mosaic_grasp_square`    | `rgragulraj/act_mosaic_grasp_square`    | grasp / square    | `mosaic_grasp_square_20260515_161734`    | _(no committed script)_                     |
| `act_mosaic_grasp_star`      | `rgragulraj/act_mosaic_grasp_star`      | grasp / star      | `mosaic_grasp_star_20260612_152207`      | `train_grasp_star.sh`                       |
| `act_mosaic_navigate_square` | `rgragulraj/act_mosaic_navigate_square` | navigate / square | `mosaic_navigate_square_20260525_104704` | `train_navigate_square.sh` (wandb `mosaic`) |
| `act_mosaic_navigate_star`   | `rgragulraj/act_mosaic_navigate_star`   | navigate / star   | `mosaic_navigate_star_20260612_170707`   | `train_navigate_star.sh`                    |
| `act_mosaic_insert_square`   | `rgragulraj/act_mosaic_insert_square`   | insert / square   | `mosaic_insert_square_20260612_130337`   | `train_insert_square.sh`                    |
| `act_mosaic_insert_star`     | `rgragulraj/act_mosaic_insert_star`     | insert / star     | `mosaic_insert_star_20260612_191647`     | `train_insert_star.sh`                      |

### 3.2 ACT — insert/star, high-quality merged run

- **Output:** `act_mosaic_insert_star_merged_20260625` (checkpoints `010000`…`100000`, `last`)
  — also an older `act_mosaic_insert_star_merged/pretrained_model` (236 MB).
- **HF repo_id:** `rgragulraj/act_mosaic_insert_star_merged`
- **Dataset:** `mosaic_insert_star_merged` (50 eps / 26,678 frames)
- **Train script:** `scripts/train_act_insert_star_merged.sh` (`CUDA_VISIBLE_DEVICES=1`)
- **Config delta vs. §3.1:** `chunk_size=50`, `n_action_steps=50`, `latent_dim=64`,
  `vision_backbone=resnet34` (ResNet34 IMAGENET1K_V1), `batch_size=16`,
  `steps=100000`, `save_freq=10000`. Weights ≈ 236 MB.
- **Rollout:** `scripts/insert_star_merged_rollout.sh` → checkpoint `100000`, `--fps=30 --duration=20`.

### 3.3 Diffusion policies — insert/star only

| Output dir                                     | HF repo_id                                       | Dataset                              | Config                                                                                                                                                                                                                         | Steps   | Checkpoints               | Weights  |
| ---------------------------------------------- | ------------------------------------------------ | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | ------------------------- | -------- |
| `diffusion_mosaic_insert_star`                 | `rgragulraj/diffusion_mosaic_insert_star`        | `mosaic_insert_star_20260612_191647` | `n_obs_steps=2`, `horizon=64`, `n_action_steps=32`, `resnet18` (IMAGENET1K_V1), DDPM `num_train_timesteps=100`, `num_inference_steps=null`, `lr=1e-4`, `batch=64`                                                              | 20,000  | `005000`…`020000`, `last` | ≈ 1.1 GB |
| `diffusion_mosaic_insert_star_merged_20260624` | `rgragulraj/diffusion_mosaic_insert_star_merged` | `mosaic_insert_star_merged`          | `n_obs_steps=2`, `horizon=16`, `n_action_steps=8`, `resnet34` (IMAGENET1K_V1), DDPM `num_train_timesteps=100`, `beta_schedule=squaredcos_cap_v2`, `prediction_type=epsilon`, `num_inference_steps=null`, `lr=5e-5`, `batch=32` | 200,000 | `200000` only             | ≈ 1.2 GB |

- Train scripts: `train_diffusion_insert_star.sh`, `train_diffusion_insert_star_merged.sh`
  (older un-timestamped `diffusion_mosaic_insert_star_merged/` holds just a `pretrained_model`).
- Rollout: `scripts/insert_star_merged_diffusion_rollout.sh` → checkpoint `200000`,
  `--policy.num_inference_steps=10` (override for the laptop 3050; drop the flag for the
  faithful 100-step denoise), `--fps=30 --duration=120`.

### 3.4 Orchestrator checkpoint wiring

`scripts/run_mosaic_square.py` — fixed step counts per phase:

| Phase    | Checkpoint                                                                     |
| -------- | ------------------------------------------------------------------------------ |
| grasp    | `outputs/train/act_mosaic_grasp_square/checkpoints/020000/pretrained_model`    |
| navigate | `outputs/train/act_mosaic_navigate_square/checkpoints/020000/pretrained_model` |
| insert   | `outputs/train/act_mosaic_insert_square/checkpoints/020000/pretrained_model`   |

`scripts/run_mosaic_star.py` — `GRASP_STEPS=400`, `NAVIGATE_STEPS=300`, `INSERT_STEPS=200`, `FPS=30`:

| Phase    | Checkpoint                                                                                   |
| -------- | -------------------------------------------------------------------------------------------- |
| grasp    | `outputs/train/act_mosaic_grasp_star/checkpoints/020000/pretrained_model`                    |
| navigate | `outputs/train/act_mosaic_navigate_star/checkpoints/020000/pretrained_model`                 |
| insert   | `outputs/train/diffusion_mosaic_insert_star/checkpoints/020000/pretrained_model` (Diffusion) |

Single-phase rollout helpers: `grasp_star_rollout.sh`, `grasp_square_rollout`,
`grasp_keyframe_rollout.sh`, `grasp_keyframe_rollout_star.sh` (all use the grasp
ACT `020000` checkpoints).

---

## 4. Coverage vs. the MOSAIC design

`CLAUDE.md` describes six ACT checkpoints: `grasp`, `navigate_{rectangle,square,cross,star}`,
`insert`. Current state:

- **Done:** `square` and `star` lines for all three phases (grasp / navigate / insert),
  plus Diffusion + merged-dataset variants for insert/star.
- **Not yet recorded or trained:** `navigate_rectangle`, `navigate_cross`, and their
  grasp/insert counterparts.
