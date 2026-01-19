#!/usr/bin/env python3
"""
DeepSpeed ZeRO-3 Training Script for BAGEL-7B-MoT
Designed for single GPU with CPU offload.

Usage:
    deepspeed --num_gpus=1 train_deepspeed.py --deepspeed_config configs/ds_zero3_offload.json
"""

import os
import json
import argparse
from time import time

import torch
import deepspeed
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch.utils.data import DataLoader
import pandas as pd

# Add BAGEL-Canvas to path
import sys
sys.path.insert(0, '/home/ubuntu/oujingfeng/project/spatialreasoning/MathCanvas/BAGEL-Canvas')

from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from data.data_utils import add_special_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                        default='/home/ubuntu/oujingfeng/project/spatialreasoning/models/BAGEL-7B-MoT')
    parser.add_argument('--data_path', type=str,
                        default='/home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/data/output/train_split.parquet')
    parser.add_argument('--output_dir', type=str,
                        default='/home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/outputs/deepspeed_sft')
    parser.add_argument('--total_steps', type=int, default=1530)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--local_rank', type=int, default=-1)
    
    # DeepSpeed will add its own arguments (including --deepspeed_config)
    parser = deepspeed.add_config_arguments(parser)
    
    return parser.parse_args()


def get_model_config(model_path):
    """Get model configs without creating the model"""
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + (-2)  # vit_select_layer
    vit_config.rope = False
    
    # Load VAE config
    vae_model, vae_config = load_ae(
        local_path=os.path.join(model_path, "ae.safetensors")
    )
    
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=2,
        max_latent_size=64,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=2.0,
    )
    
    return llm_config, vit_config, vae_config, bagel_config, vae_model


def create_model_with_zero_init(model_path, ds_config):
    """Create model with DeepSpeed ZeRO-3 initialization"""
    print("Loading model configuration...")
    llm_config, vit_config, vae_config, bagel_config, vae_model = get_model_config(model_path)
    
    print("Creating model with ZeRO-3 initialization...")
    # Use zero.Init context to create model in sharded way
    with zero.Init(config_dict_or_path=ds_config,
                   enabled=True,
                   mem_efficient_linear=True):
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_config)
        
        # Convert ViT patch embedding from Conv2d to Linear (required for BAGEL)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    
    # Load tokenizer and add special tokens
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
    
    # Freeze VAE
    for param in vae_model.parameters():
        param.requires_grad = False
    
    return model, vae_model, tokenizer, new_token_ids


def load_checkpoint_to_zero3_model(model, model_path):
    """Load checkpoint weights into ZeRO-3 partitioned model"""
    print("Loading pretrained weights (ZeRO-3 compatible)...")
    from safetensors.torch import load_file
    state_dict = load_file(os.path.join(model_path, "ema.safetensors"), device="cpu")
    
    # Remove position embeddings (fixed sinusoidal)
    if 'latent_pos_embed.pos_embed' in state_dict:
        state_dict.pop('latent_pos_embed.pos_embed')
    if 'vit_pos_embed.pos_embed' in state_dict:
        state_dict.pop('vit_pos_embed.pos_embed')
    
    # For ZeRO-3, we need to gather params before loading
    for name, param in model.named_parameters():
        if name in state_dict:
            # Gather the partitioned param
            if hasattr(param, 'ds_id'):
                with zero.GatheredParameters([param], modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        param.data.copy_(state_dict[name].to(param.dtype))
            else:
                param.data.copy_(state_dict[name].to(param.dtype))
    
    del state_dict
    print("Checkpoint loaded successfully")


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for BlockMoving SFT"""
    
    def __init__(self, parquet_path, tokenizer):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        print(f"Loaded {len(self.df)} samples from {parquet_path}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Build text from question and solution interleave
        question_interleave = row['question_interleave']
        solution_interleave = row['solution_interleave']
        answer = row['answer']
        
        # Simple text concatenation for now
        text_parts = []
        for item in question_interleave:
            if item['type'] == 'text':
                text_parts.append(item['content'])
        
        for item in solution_interleave:
            if item['type'] == 'text':
                text_parts.append(item['content'])
        
        full_text = " ".join(text_parts)
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text, max_length=4096, truncation=True)
        
        return {
            'input_ids': torch.tensor(tokens[:-1]),
            'labels': torch.tensor(tokens[1:]),
        }


def collate_fn(batch):
    """Collate function with padding"""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad sequences
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_labels = []
    attention_mask = []
    
    for ids, lbls in zip(input_ids, labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([lbls, torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)]))
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'labels': torch.stack(padded_labels),
        'attention_mask': torch.stack(attention_mask),
    }


def main():
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize distributed
    deepspeed.init_distributed()
    
    # Get DS config path
    ds_config = args.deepspeed_config
    
    # Create model with ZeRO-3 sharded initialization
    model, vae_model, tokenizer, new_token_ids = create_model_with_zero_init(
        args.model_path, ds_config
    )
    
    # Create dataset
    dataset = SimpleDataset(args.data_path, tokenizer)
    
    # Initialize DeepSpeed engine
    print("Initializing DeepSpeed engine...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )
    
    # Load checkpoint after DeepSpeed initialization
    load_checkpoint_to_zero3_model(model_engine.module, args.model_path)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # DeepSpeed handles gradient accumulation
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"Training for {args.total_steps} steps...")
    
    step = 0
    start_time = time()
    
    while step < args.total_steps:
        for batch in dataloader:
            if step >= args.total_steps:
                break
            
            # Move to device
            input_ids = batch['input_ids'].to(model_engine.device)
            labels = batch['labels'].to(model_engine.device)
            attention_mask = batch['attention_mask'].to(model_engine.device)
            
            # Forward pass (simplified - just language modeling loss)
            outputs = model_engine.module.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            # Backward pass
            model_engine.backward(loss)
            model_engine.step()
            
            step += 1
            
            # Logging
            if step % args.log_every == 0:
                elapsed = time() - start_time
                steps_per_sec = args.log_every / elapsed
                print(f"Step {step}/{args.total_steps} | Loss: {loss.item():.4f} | Steps/sec: {steps_per_sec:.2f}")
                start_time = time()
            
            # Save checkpoint
            if step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                model_engine.save_checkpoint(save_path)
                print(f"Saved checkpoint to {save_path}")
    
    # Final save
    final_path = os.path.join(args.output_dir, "final")
    model_engine.save_checkpoint(final_path)
    print(f"Training complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
