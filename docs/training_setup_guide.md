# BlockMoving SFT 训练部署指南

本文档详细记录了在新服务器上部署 BlockMoving SFT 训练所需的所有文件和依赖。

## 目录结构

```
服务器上需要的目录结构：

/path/to/project/
├── MathCanvas/
│   └── BAGEL-Canvas/           # MathCanvas 代码库 (必需)
│       ├── modeling/           # 模型定义
│       ├── data/              # 数据处理
│       │   ├── configs/       # 训练配置
│       │   │   └── blockmoving_sft.yaml  # BlockMoving 配置
│       │   └── dataset_info.py  # 数据集注册
│       └── train/             # 训练代码
│           ├── pretrain_unified_navit.py
│           └── fsdp_utils.py
│
├── BlockMovingSFT/            # BlockMoving SFT 项目
│   ├── data/
│   │   └── output/
│   │       ├── train_split.parquet    # 训练数据
│   │       └── test_split.parquet     # 测试数据
│   ├── scripts/
│   │   └── train_blockmoving.sh       # 训练脚本
│   └── docs/
│       └── training_setup_guide.md    # 本文档
│
└── models/
    └── BAGEL-7B-MoT/          # 预训练模型
        ├── ema.safetensors    # 权重文件 (~29GB)
        ├── ae.safetensors     # VAE 权重
        ├── llm_config.json
        ├── vit_config.json
        └── tokenizer.json
```

---

## 必需的文件

### 1. MathCanvas/BAGEL-Canvas 代码库

**来源**: 克隆或复制整个 BAGEL-Canvas 目录

需要修改的文件：
- `data/configs/blockmoving_sft.yaml` - 训练配置
- `data/dataset_info.py` - 数据集注册

#### blockmoving_sft.yaml
```yaml
# BlockMoving SFT training config
data_weights:
  interleave_reasoning: 1.0

data_names:
  interleave_reasoning:
    - blockmoving_train
```

#### dataset_info.py 添加内容
在 `DATASET_INFO["interleave_reasoning"]` 字典中添加：
```python
"blockmoving_train": {
    "path": "/path/to/BlockMovingSFT/data/output",
    "parquet": "train_split.parquet",
    "parquet_info": "train_split_info.json",
},
```

---

### 2. BlockMovingSFT 数据文件

| 文件 | 说明 |
|------|------|
| `train_split.parquet` | 训练数据 (613 样本) |
| `test_split.parquet` | 测试数据 (69 样本) |
| `train_split_info.json` | 训练数据元信息 |
| `test_split_info.json` | 测试数据元信息 |

**Parquet 文件格式**:
- `answer`: 答案文本
- `question_interleave`: 问题的图文交织列表
- `solution_interleave`: 解答的图文交织列表
- `question_images`: 问题图片 (PIL bytes)
- `solution_images`: 解答图片 (PIL bytes)

---

### 3. 预训练模型 (BAGEL-7B-MoT)

**下载方式**:
```bash
# 使用 huggingface-cli
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir /path/to/models/BAGEL-7B-MoT

# 或使用 Python
from huggingface_hub import snapshot_download
snapshot_download("ByteDance-Seed/BAGEL-7B-MoT", local_dir="/path/to/models/BAGEL-7B-MoT")
```

**必需文件**:
- `ema.safetensors` (~29GB, float32)
- `ae.safetensors` (~335MB)
- `llm_config.json`
- `vit_config.json`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`

---

## 环境配置

### Conda 环境
```bash
conda create -n bagel-canvas python=3.10
conda activate bagel-canvas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate safetensors pandas pyarrow pillow
pip install flash-attn --no-build-isolation
```

### CUDA 要求
- CUDA Toolkit 12.x
- 需要设置环境变量:
```bash
export CUDA_HOME=/usr/local/cuda-12.x
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## 硬件要求

### 多卡训练 (推荐)
- **最低**: 2x RTX 4090 (24GB) 或 2x A100 (40GB)
- **推荐**: 4x RTX 4090 或 4x A100 (80GB)
- **内存**: 128GB+ RAM

### 单卡训练
- **不支持**: BAGEL-7B-MoT (~29GB float32) 无法在单张 24GB GPU 上运行
- 需要使用 LoRA 或更小的模型

---

## 训练脚本

### train_blockmoving.sh (多卡配置)

```bash
#!/bin/bash
MODEL_PATH="/path/to/models/BAGEL-7B-MoT"

export WANDB_API_KEY="your_key"
WANDB_ENABLE="False"
WANDB_NAME="blockmoving_sft"
WANDB_RUNID="0"

RESULTS_DIR="/path/to/outputs/${WANDB_NAME}--${WANDB_RUNID}"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
mkdir -p $RESULTS_DIR $CKPT_DIR

TOTAL_STEPS=1530
SAVE_EVERY=500
LOG_EVERY=10

cd /path/to/MathCanvas/BAGEL-Canvas

# 4卡训练
torchrun \
  --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29501 \
  --nproc_per_node=4 \
  -m train.pretrain_unified_navit \
  --dataset_config_file ./data/configs/blockmoving_sft.yaml \
  --results_dir $RESULTS_DIR \
  --checkpoint_dir $CKPT_DIR \
  --model_path $MODEL_PATH \
  --num_shard 4 \
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
  --ema 0.9999 \
  --cpu_offload False \
  --num_workers 4 \
  --expected_num_tokens 8192 \
  --max_num_tokens 8192 \
  --max_num_tokens_per_sample 8192 \
  --prefer_buffer_before 8192 \
  --text_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.1 \
  --vae_cond_dropout_prob 0.1 \
  --debug_batches 3 \
  --enable_wandb $WANDB_ENABLE \
  --wandb_name $WANDB_NAME \
  --wandb_runid $WANDB_RUNID
```

---

## 快速部署清单

1. [ ] 克隆 MathCanvas/BAGEL-Canvas 代码库
2. [ ] 复制 BlockMovingSFT 数据文件 (parquet + info.json)
3. [ ] 下载 BAGEL-7B-MoT 预训练模型
4. [ ] 创建 `blockmoving_sft.yaml` 配置文件
5. [ ] 修改 `dataset_info.py` 注册数据集
6. [ ] 修改 `train_blockmoving.sh` 中的路径
7. [ ] 配置 Conda 环境和 CUDA
8. [ ] 运行训练: `bash train_blockmoving.sh`

---

## 注意事项

1. **路径**: 所有脚本中的绝对路径需要根据新服务器调整
2. **GPU 数量**: `--nproc_per_node` 和 `--num_shard` 需要匹配实际 GPU 数量
3. **显存**: 如果 OOM，减小 `--max_num_tokens` 和 `--expected_num_tokens`
4. **WandB**: 如需启用，设置 `WANDB_ENABLE="True"` 和正确的 API Key
