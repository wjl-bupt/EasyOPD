"""Merge FSDP sharded checkpoints into HuggingFace format.
Works with verl's FSDP checkpoint format (model_world_size_N_rank_K.pt).
"""
import os
import sys
import glob
import torch
import json
from pathlib import Path
from collections import OrderedDict

def merge_fsdp_checkpoint(ckpt_dir, base_model_path, output_dir):
    """Merge FSDP sharded model files into a single HF model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load base model architecture
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Get world size
    config_path = os.path.join(ckpt_dir, "fsdp_config.json")
    with open(config_path) as f:
        fsdp_config = json.load(f)
    world_size = fsdp_config["world_size"]
    print(f"World size: {world_size}")
    
    # Load all shards
    print(f"Loading {world_size} model shards...")
    all_shards = []
    for rank in range(world_size):
        shard_path = os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        print(f"  Loading rank {rank}: {shard_path}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        all_shards.append(shard)
    
    # Merge shards - FSDP shards parameters along dim 0
    print("Merging shards...")
    merged_state_dict = OrderedDict()
    
    # Get parameter names from first shard
    param_names = list(all_shards[0].keys())
    print(f"  Total parameters: {len(param_names)}")
    
    for name in param_names:
        tensors = [shard[name] for shard in all_shards]
        
        # Check if this is a DTensor (has _local_tensor attribute) or regular tensor
        first = tensors[0]
        if hasattr(first, '_local_tensor'):
            # DTensor - extract local tensors and concatenate
            local_tensors = [t._local_tensor for t in tensors]
            merged_state_dict[name] = torch.cat(local_tensors, dim=0)
        elif first.shape == tensors[1].shape if len(tensors) > 1 else True:
            # Check if all shards have same shape (replicated) or different (sharded)
            if all(t.shape == first.shape for t in tensors):
                # Could be replicated or sharded along dim 0
                # Check if concatenating gives expected shape
                ref_param = dict(model.named_parameters()).get(name)
                if ref_param is not None:
                    if first.numel() * world_size == ref_param.numel():
                        # Sharded - concatenate
                        merged_state_dict[name] = torch.cat(tensors, dim=0).reshape(ref_param.shape)
                    else:
                        # Replicated - just use first
                        merged_state_dict[name] = first
                else:
                    merged_state_dict[name] = first
            else:
                # Different shapes - concatenate along dim 0
                merged_state_dict[name] = torch.cat(tensors, dim=0)
        else:
            merged_state_dict[name] = first
    
    # Load merged state dict into model
    print("Loading merged weights into model...")
    missing, unexpected = model.load_state_dict(merged_state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="Path to FSDP actor checkpoint dir")
    parser.add_argument("--base_model", required=True, help="Path to base HF model")
    parser.add_argument("--output_dir", required=True, help="Output directory for merged model")
    args = parser.parse_args()
    
    merge_fsdp_checkpoint(args.ckpt_dir, args.base_model, args.output_dir)
