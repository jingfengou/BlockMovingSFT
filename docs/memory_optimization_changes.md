# 单卡 RTX 4090 显存优化更改记录

本文档记录了为了在单张 RTX 4090 (24GB) 上运行 BAGEL-7B-MoT 训练所做的所有代码修改。

## 1. 训练脚本参数优化

**文件**: `/home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/scripts/train_blockmoving.sh`

关键参数修改：
```bash
--ema 0                          # 禁用 EMA 模型 (节省 ~14GB)
--cpu_offload True               # 启用 CPU 卸载
--freeze_llm True                # 冻结 LLM 主干 (节省 ~10GB 优化器状态)
--freeze_vit True                # 冻结 ViT 编码器
--expected_num_tokens 4096       # 减小 token 数量
--max_num_tokens 4096
--max_num_tokens_per_sample 4096
--prefer_buffer_before 4096
--num_workers 2                  # 减少 worker 数
--nproc_per_node=1               # 单 GPU
--num_shard 1

# 环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 2. 训练代码修改 - 跳过 EMA 模型创建

**文件**: `/home/ubuntu/oujingfeng/project/spatialreasoning/MathCanvas/BAGEL-Canvas/train/pretrain_unified_navit.py`

### 修改 1: 第 504-516 行 (跳过 EMA 创建)

**原始代码:**
```python
ema_model = deepcopy(model)
model, ema_model = FSDPCheckpoint.try_load_ckpt(
    resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
)
ema_model = fsdp_ema_setup(ema_model, fsdp_config)
fsdp_model = fsdp_wrapper(model, fsdp_config)
```

**修改后:**
```python
# Only create EMA model if ema > 0 to save GPU memory
if training_args.ema > 0:
    ema_model = deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
else:
    ema_model = None
    model, _ = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, None, resume_from_ema=finetune_from_ema
    )
fsdp_model = fsdp_wrapper(model, fsdp_config)
```

### 修改 2: 第 623-625 行 (跳过 EMA eval)

**原始代码:**
```python
fsdp_model.train()
ema_model.eval()
```

**修改后:**
```python
fsdp_model.train()
if ema_model is not None:
    ema_model.eval()
```

### 修改 3: 第 680-681 行 (跳过 EMA 更新)

**原始代码:**
```python
fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
```

**修改后:**
```python
if ema_model is not None:
    fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
```

---

## 显存节省估算

| 优化项 | 节省显存 |
|--------|----------|
| 禁用 EMA 模型 | ~14 GB |
| 冻结 LLM 主干 | ~10 GB (优化器状态) |
| 冻结 ViT | ~1 GB |
| CPU Offload | ~5-8 GB |
| 减小 token 数量 | ~2-4 GB |
| **总计** | **~30+ GB** |

> **注意**: 冻结 LLM 和 ViT 后，只有 Connector 层 (~100M 参数) 被训练。

---

## 3. FSDP 单卡 CPU Offload 初始化修改

**文件**: `/home/ubuntu/oujingfeng/project/spatialreasoning/MathCanvas/BAGEL-Canvas/train/fsdp_utils.py`

### 问题
FSDP 包装阶段会尝试将所有参数 flatten 到 GPU，导致单卡 OOM。

### 修改: fsdp_wrapper 函数 (第 49-95 行)

在 `fsdp_wrapper` 函数中添加单卡特殊处理：

```python
# For single GPU with CPU offload, use special initialization
if fsdp_config.cpu_offload and dist.get_world_size() == 1:
    # Move model to bfloat16 on CPU first to reduce memory
    original_model = original_model.to(dtype=torch.bfloat16)
    
    def param_init_fn(module):
        # Keep params on CPU during init
        pass
    
    return FSDP(
        original_model,
        # ... auto_wrap_policy ...
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.NO_SHARD,  # Force NO_SHARD
        cpu_offload=CPUOffload(offload_params=True),
        sync_module_states=False,
        param_init_fn=param_init_fn,
        device_mesh=None,
    )
```

---

## 恢复步骤

在新服务器上，需要修改以下文件：

1. `train/pretrain_unified_navit.py` - EMA 相关 3 处修改
2. `train/fsdp_utils.py` - FSDP 单卡初始化修改

或者直接复制已修改的文件。
