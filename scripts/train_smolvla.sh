#!/usr/bin/env bash
set -euo pipefail

: "${HF_DATASET_REPO_ID:?Set HF_DATASET_REPO_ID, e.g. export HF_DATASET_REPO_ID=my-org/my-robot-dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/smolvla}"
JOB_NAME="${JOB_NAME:-smolvla-pixi-ros2-mujoco}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

lerobot-train \
  --policy.path="${POLICY_PATH}" \
  --dataset.repo_id="${HF_DATASET_REPO_ID}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --policy.device="${POLICY_DEVICE}" \
  --wandb.enable="${WANDB_ENABLE}"
