# ASU SCAI Server — Training Reference

## Connection

```bash
ssh rrangasa@en4230881l.scai.dhcp.asu.edu
```

## Server Layout

```
/data/rrangasa/
├── conda/               # Miniconda installation
├── envs/
│   └── lerobot/         # Conda environment
├── datasets/            # Transferred training datasets (HF cache format)
├── lerobot/             # Cloned lerobot-mosaic repo
│   └── scripts/         # Training scripts live here
├── outputs/
│   └── train/           # Training checkpoints and logs
│       └── act_mosaic_<policy>/
│           └── checkpoints/
│               ├── 005000/pretrained_model/
│               ├── 010000/pretrained_model/
│               ├── 015000/pretrained_model/
│               └── 020000/pretrained_model/
└── tmp/
```

**GPU**: H100

## Activate Environment

```bash
conda activate /data/rrangasa/envs/lerobot
```

Always activate before running any `lerobot-train` command.

## Workflow

### 1. Record dataset on laptop

```bash
bash scripts/record_<policy>_<shape>.sh
```

Dataset is saved to `~/.cache/huggingface/lerobot/rgragulraj/<dataset_name>_<timestamp>/`.

### 2. Push dataset to HF Hub (from laptop)

```python
uv run python -c "
from lerobot.datasets import LeRobotDataset
ds = LeRobotDataset('rgragulraj/<repo_id>', root='/home/rgragulraj/.cache/huggingface/lerobot/rgragulraj/<dataset_folder>')
ds.push_to_hub()
"
```

### 3. Transfer dataset to server (from laptop)

```bash
rsync -avz --progress \
  ~/.cache/huggingface/lerobot/rgragulraj/<dataset_folder> \
  rrangasa@en4230881l.scai.dhcp.asu.edu:/data/rrangasa/datasets/
```

### 4. Transfer training script to server (from laptop)

```bash
rsync -avz scripts/train_<policy>_<shape>.sh \
  rrangasa@en4230881l.scai.dhcp.asu.edu:/data/rrangasa/lerobot/scripts/
```

### 5. Run training on server

```bash
conda activate /data/rrangasa/envs/lerobot
cd /data/rrangasa/lerobot
bash scripts/train_<policy>_<shape>.sh
```

Training runs in the foreground. Do not close the SSH session until it finishes.

### 6. Transfer trained model back to laptop (from laptop)

```bash
rsync -avz --progress \
  rrangasa@en4230881l.scai.dhcp.asu.edu:/data/rrangasa/outputs/train/act_mosaic_<policy> \
  outputs/train/
```

### 7. Push trained policy to HF Hub (from laptop)

```python
uv run python -c "
from lerobot.policies import get_policy_class
policy = get_policy_class('act').from_pretrained('outputs/train/act_mosaic_<policy>/checkpoints/020000/pretrained_model')
policy.push_to_hub('rgragulraj/act_mosaic_<policy>')
print('Push complete.')
"
```

## Training Script Template

All MOSAIC policies use the same hyperparameters. Copy `train_navigate_square.sh` and update:

- `--dataset.repo_id`
- `--dataset.root`
- `--policy.repo_id`
- `--output_dir`
- `--job_name`

Key hyperparameters (do not change without reason):

| Parameter                | Value      |
| ------------------------ | ---------- |
| `policy.type`            | `act`      |
| `policy.vision_backbone` | `resnet18` |
| `policy.dim_model`       | `512`      |
| `policy.chunk_size`      | `100`      |
| `batch_size`             | `8`        |
| `steps`                  | `20000`    |
| `save_freq`              | `5000`     |
| `wandb.enable`           | `false`    |

## Dataset Naming Conventions

| Role              | Dataset repo_id                      | Local folder pattern               |
| ----------------- | ------------------------------------ | ---------------------------------- |
| Grasp (square)    | `mosaic_grasp_square`                | `mosaic_grasp_square_<timestamp>`  |
| Navigate (square) | `mosaic_navigate_square_<timestamp>` | same                               |
| Insert (square)   | `mosaic_insert_square`               | `mosaic_insert_square_<timestamp>` |
| Grasp (star)      | `mosaic_grasp_star`                  | `mosaic_grasp_star_<timestamp>`    |

## Policy Naming Conventions

| Policy          | HF Hub repo_id                          |
| --------------- | --------------------------------------- |
| Grasp square    | `rgragulraj/act_mosaic_grasp_square`    |
| Navigate square | `rgragulraj/act_mosaic_navigate_square` |
| Insert square   | `rgragulraj/act_mosaic_insert_square`   |
| Grasp star      | `rgragulraj/act_mosaic_grasp_star`      |

## Notes

- The disconnect error `RuntimeError: Failed to write 'Torque_Enable' on id_=6` at the end of recording is harmless — it's a servo overload on disconnect, not a data loss issue.
- Always verify episode count after recording: `cat <dataset_path>/meta/info.json | grep total_episodes`
- Checkpoints save every 5000 steps. Use `020000` (final) unless evaluating mid-training.
- WandB is disabled on the server — monitor training via stdout logs only.
