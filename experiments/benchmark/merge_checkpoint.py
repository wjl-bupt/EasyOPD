"""Merge FSDP sharded checkpoint into a single HuggingFace model."""

import os
import sys
import torch
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, "/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD")


def merge_fsdp_checkpoint(ckpt_dir: str, output_dir: str, base_model_path: str):
    """Merge FSDP sharded model weights into a single HuggingFace model.
    
    Args:
        ckpt_dir: Path to the FSDP checkpoint directory (e.g., global_step_200/actor)
        output_dir: Path to save the merged HuggingFace model
        base_model_path: Path to the base model (for config/tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading base model config from: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Find all shard files
    shard_files = sorted(Path(ckpt_dir).glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        print(f"No shard files found in {ckpt_dir}")
        return False
    
    print(f"Found {len(shard_files)} shard files")
    
    # Load and merge shards
    # FSDP2 saves DTensors sharded along dim 0 across ranks.
    # We need to load all ranks and concatenate _local_tensor along dim 0.
    print(f"Found {len(shard_files)} shard files")
    print("Loading all shards and concatenating DTensor local tensors...")
    
    # Load all shards
    all_shards = []
    for shard_file in shard_files:
        print(f"  Loading {shard_file.name}...")
        shard = torch.load(shard_file, map_location="cpu", weights_only=False)
        all_shards.append(shard)
    
    # Merge: concatenate _local_tensor from each rank along dim 0
    merged_state_dict = OrderedDict()
    keys = list(all_shards[0].keys())
    
    for key in keys:
        local_tensors = []
        for shard in all_shards:
            val = shard[key]
            if hasattr(val, '_local_tensor'):
                local_tensors.append(val._local_tensor)
            elif isinstance(val, torch.Tensor):
                local_tensors.append(val)
            else:
                local_tensors.append(val)
                
        if len(local_tensors) == 1:
            merged_state_dict[key] = local_tensors[0]
        else:
            # Check if tensors need concatenation (different local sizes)
            # or if they're replicated (same data on all ranks)
            first_shape = local_tensors[0].shape
            if all(t.shape == first_shape for t in local_tensors):
                # Check if all tensors are identical (replicated parameter)
                if torch.equal(local_tensors[0], local_tensors[1]):
                    # Replicated - just use rank 0
                    merged_state_dict[key] = local_tensors[0]
                else:
                    # Sharded along dim 0 - concatenate
                    merged_state_dict[key] = torch.cat(local_tensors, dim=0)
            else:
                # Different shapes - concatenate along the differing dimension
                merged_state_dict[key] = torch.cat(local_tensors, dim=0)
    
    print(f"Merged state dict has {len(merged_state_dict)} keys")
    
    # Load into model
    missing, unexpected = model.load_state_dict(merged_state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("Done!")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, help="FSDP checkpoint dir (e.g., .../global_step_200/actor)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output HuggingFace model dir")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path for config/tokenizer")
    args = parser.parse_args()
    
    merge_fsdp_checkpoint(args.ckpt_dir, args.output_dir, args.base_model)
