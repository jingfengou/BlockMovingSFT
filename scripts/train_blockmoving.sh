#!/bin/bash
# BlockMoving SFT Training Script
# Directly fine-tunes BAGEL-7B-MoT on BlockMoving interleaved reasoning data

MODEL_PATH="/home/ubuntu/oujingfeng/project/spatialreasoning/models/BAGEL-7B-MoT"

# WandB Configuration (optional)
export WANDB_API_KEY="your_wandb_key"
WANDB_ENABLE="False"
WANDB_NAME="blockmoving_sft"
WANDB_RUNID="0"
WANDB_RESUME="allow"
WANDB_OFFLINE="True"

# Output directories
RESULTS_DIR="/home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/outputs/${WANDB_NAME}--${WANDB_RUNID}"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
mkdir -p $RESULTS_DIR
mkdir -p $CKPT_DIR

# Training configuration
# With 613 samples and batch size ~4, one epoch ≈ 153 steps
# For 10 epochs: 1530 steps
TOTAL_STEPS=1530
SAVE_EVERY=500
LOG_EVERY=10

cd /home/ubuntu/oujingfeng/project/spatialreasoning/MathCanvas/BAGEL-Canvas

# Single GPU training (1x RTX 4090) - still need torchrun for distributed env setup
# Memory optimization: cpu_offload enabled, EMA disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
  --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29501 \
  --nproc_per_node=1 \
  -m train.pretrain_unified_navit \
  --dataset_config_file ./data/configs/blockmoving_sft.yaml \
  --results_dir $RESULTS_DIR \
  --checkpoint_dir $CKPT_DIR \
  --model_path $MODEL_PATH \
  --num_shard 1 \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --timestep_shift 2.0 \
  --use_flex True \
  --finetune_from_hf True \
  --auto_resume True \
  --log_every $LOG_EVERY \
  --save_every $SAVE_EVERY \
  --del_previous_state True \
  --lr 1e-5 \
  --lr_scheduler cosine \
  --min_lr 1e-7 \
  --ce_weight 0.25 \
  --mse_weight 1 \
  --warmup_steps 100 \
  --total_steps $TOTAL_STEPS \
  --ema 0 \
  --cpu_offload True \
  --freeze_llm True \
  --freeze_vit True \
  --num_workers 2 \
  --expected_num_tokens 4096 \
  --max_num_tokens 4096 \
  --max_num_tokens_per_sample 4096 \
  --prefer_buffer_before 4096 \
  --text_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.1 \
  --vae_cond_dropout_prob 0.1 \
  --debug_batches 3 \
  --enable_wandb $WANDB_ENABLE \
  --wandb_name $WANDB_NAME \
  --wandb_runid $WANDB_RUNID \
  --wandb_resume $WANDB_RESUME \
  --wandb_offline $WANDB_OFFLINE
