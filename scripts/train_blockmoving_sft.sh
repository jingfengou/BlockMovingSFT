#!/bin/bash
# scripts/train_blockmoving_sft.sh
# Training script for BlockMoving COT SFT with BAGEL

# ============================================================
# Configuration - Update these paths for your environment
# ============================================================

# Path to pretrained BAGEL model
MODEL_PATH="your_model_path/ByteDance-Seed/BAGEL-7B-MoT"

# Optional: Resume from a previous checkpoint
# RESUME_FROM="your_save_path/blockmoving_sft--0/checkpoints/0080000"
RESUME_FROM=""

# Weights & Biases configuration
export WANDB_API_KEY="your_wandb_key"
WANDB_ENABLE="True"
WANDB_NAME="blockmoving_sft"
WANDB_RUNID="0"
WANDB_RESUME="allow"
WANDB_OFFLINE="False"

# Output directories
RESULTS_DIR="./results/${WANDB_NAME}--${WANDB_RUNID}"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
mkdir -p $RESULTS_DIR
mkdir -p $CKPT_DIR

# ============================================================
# Training Launch
# ============================================================

# First, ensure the MathCanvas BAGEL-Canvas is in the Python path
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))/../MathCanvas/BAGEL-Canvas"

# Launch training with torchrun for distributed training
torchrun \
  --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29501 \
  --nproc_per_node=8 \
  -m train.pretrain_unified_navit \
  --dataset_config_file ./configs/blockmoving_sft.yaml \
  --results_dir $RESULTS_DIR \
  --checkpoint_dir $CKPT_DIR \
  --model_path $MODEL_PATH \
  --num_shard 8 \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --timestep_shift 2.0 \
  --use_flex True \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 20 \
  --save_every 4000 \
  --del_previous_state True \
  --lr 1e-5 \
  --lr_scheduler cosine \
  --min_lr 1e-7 \
  --ce_weight 0.25 \
  --mse_weight 1 \
  --warmup_steps 200 \
  --total_steps 8000 \
  --ema 0.995 \
  --num_workers 4 \
  --expected_num_tokens 46080 \
  --max_num_tokens 51200 \
  --max_num_tokens_per_sample 25600 \
  --prefer_buffer_before 25600 \
  --text_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.1 \
  --vae_cond_dropout_prob 0.1 \
  --debug_batches 3 \
  --enable_wandb $WANDB_ENABLE \
  --wandb_name $WANDB_NAME \
  --wandb_runid $WANDB_RUNID \
  --wandb_resume $WANDB_RESUME \
  --wandb_offline $WANDB_OFFLINE
