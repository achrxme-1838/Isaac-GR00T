#!/bin/bash
#SBATCH --job-name=gr00t-robocasa
#SBATCH --output=robocasa-%j.out
#SBATCH --error=robocasa-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --time=72:00:00
#SBATCH --exclude=node1,node2
#SBATCH --export=ALL

set -eo pipefail

source "$HOME/miniconda3/bin/activate" "$HOME/shlim/conda/envs/gr00t_lsh"
cd "$HOME/shlim/Isaac-GR00T"

unset PYTHONPATH PYTHONHOME VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HOME="$HOME/shlim/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN_PATH="$HF_HOME/token"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export HF_HUB_DISABLE_XET=1

GROOT_DATA_DIR="$PWD/data/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
DATASET_PATH="$(find "$GROOT_DATA_DIR" -maxdepth 1 -type d \
    -name 'single_panda_gripper.*' | sort | paste -sd: -)"

export NUM_GPUS="$SLURM_GPUS_ON_NODE"
export GLOBAL_BATCH_SIZE=$((NUM_GPUS * 4))
export MAX_STEPS=10000
export SAVE_STEPS=500
export DATALOADER_NUM_WORKERS=2
export USE_WANDB=1
export WANDB_MODE=online
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

python scripts/repair_lerobot_metadata.py "$DATASET_PATH" \
    --embodiment-tag ROBOCASA_PANDA_OMRON

unset RESUME_FROM_CHECKPOINT SAVE_ONLY_MODEL
bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --output-dir "$PWD/outputs/robocasa_n17_$SLURM_JOB_ID" \
    -- \
    --gradient-accumulation-steps 8 \
    --save-total-limit 2
