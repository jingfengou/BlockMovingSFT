#!/bin/bash
# DeepSpeed ZeRO-3 Training Script for BAGEL-7B-MoT
# Single GPU with full CPU offload

cd /home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/scripts

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Skip DeepSpeed CUDA JIT compilation (use PyTorch bundled CUDA)
export DS_BUILD_OPS=0
export DS_BUILD_CPU_ADAM=0

# Run with DeepSpeed
deepspeed --num_gpus=1 train_deepspeed.py \
    --deepspeed_config /home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/configs/ds_zero3_offload.json \
    --model_path /home/ubuntu/oujingfeng/project/spatialreasoning/models/BAGEL-7B-MoT \
    --data_path /home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/data/output/train_split.parquet \
    --output_dir /home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/outputs/deepspeed_sft \
    --total_steps 1530 \
    --save_every 500 \
    --log_every 10
