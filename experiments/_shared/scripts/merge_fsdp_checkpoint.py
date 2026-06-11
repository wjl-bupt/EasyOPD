"""
Merge FSDP sharded checkpoints into a standard HuggingFace model.

Usage:
    python merge_fsdp_checkpoint.py \
        --checkpoint_dir /path/to/global_step_XXX \
        --output_dir /path/to/merged_hf_model
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from torch.distributed._tensor import Placement, Shard
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM


def get_world_size(checkpoint_dir: str) -> int:
    """Extract world size from fsdp_config.json."""
    config_path = Path(checkpoint_dir) / "fsdp_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} does not exist.")
    with open(config_path) as f:
        config = json.load(f)
    world_size = config.get("world_size", None)
    if world_size is None:
        raise ValueError("World size not found in the config file.")
    return world_size


def merge_by_placement(tensors: list, placement: Placement) -> torch.Tensor:
    """Merge tensors based on their DTensor placement."""
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise NotImplementedError(f"Unsupported placement: {placement}")


def merge_fsdp_checkpoint(checkpoint_dir: str, output_dir: str):
    """Merge FSDP sharded checkpoint into a single HuggingFace model."""
    checkpoint_dir = str(Path(checkpoint_dir).resolve())
    output_dir = str(Path(output_dir).resolve())

    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Output dir: {output_dir}")

    # Get world size
    world_size = get_world_size(checkpoint_dir)
    print(f"World size: {world_size}")

    # Load rank 0 to get sharding info
    rank0_path = Path(checkpoint_dir) / f"model_world_size_{world_size}_rank_0.pt"
    rank0_state_dict = torch.load(rank0_path, map_location="cpu", weights_only=False)

    # Determine mesh info from rank 0
    pivot_key = sorted(list(rank0_state_dict.keys()))[0]
    weight = rank0_state_dict[pivot_key]

    if isinstance(weight, DTensor):
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        mesh = np.array([world_size], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Device mesh: {mesh}, mesh_dim_names: {mesh_dim_names}")

    total_shards = mesh.shape[-1]
    print(f"Total shards: {total_shards}")

    del rank0_state_dict

    # Load all shards in parallel
    model_state_dict_lst = [None] * total_shards

    def load_shard(rank: int):
        model_path = Path(checkpoint_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        futures = [executor.submit(load_shard, rank) for rank in range(total_shards)]
        for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
            future.result()

    # Merge state dicts
    print("Merging shards...")
    state_dict = {}
    param_placements = {}

    for key in sorted(model_state_dict_lst[0].keys()):
        state_dict[key] = []
        for model_state_shard in model_state_dict_lst:
            tensor = model_state_shard.pop(key)
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # Discard replicated placement at dp dimension
                if mesh_dim_names[0] in ("dp", "ddp"):
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    # Merge tensors based on placement
    for key in tqdm(sorted(state_dict.keys()), desc="Merging tensors"):
        if not isinstance(state_dict[key], list):
            continue
        if key in param_placements:
            placements = param_placements[key]
            assert len(placements) == 1, f"Only 1-D FSDP supported, got {placements}"
            state_dict[key] = merge_by_placement(state_dict[key], placements[0])
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    # Save as HuggingFace model
    os.makedirs(output_dir, exist_ok=True)

    # Copy config and tokenizer from huggingface subdirectory
    hf_subdir = Path(checkpoint_dir) / "huggingface"
    if hf_subdir.exists():
        for f in hf_subdir.iterdir():
            shutil.copy2(str(f), output_dir)
        print(f"Copied config/tokenizer from {hf_subdir}")

    # Load config and create model structure, then save weights
    config_path = Path(output_dir) / "config.json"
    if config_path.exists():
        config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
        # Save the state dict using safetensors via HF
        from safetensors.torch import save_file

        # Save as safetensors
        safetensors_path = Path(output_dir) / "model.safetensors"
        save_file(state_dict, str(safetensors_path))
        print(f"Saved model weights to {safetensors_path}")

        # Update config to indicate safetensors format
        # Create a model index if needed (for single file, not strictly necessary)
    else:
        # Fallback: save as pytorch bin
        torch.save(state_dict, Path(output_dir) / "pytorch_model.bin")
        print(f"Saved model weights to {output_dir}/pytorch_model.bin")

    print(f"\nMerge complete! Model saved to: {output_dir}")
    print(f"Total parameters: {sum(p.numel() for p in state_dict.values()):,}")


def main():
    parser = argparse.ArgumentParser(description="Merge FSDP sharded checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to the FSDP checkpoint directory (e.g., global_step_XXX)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save the merged HuggingFace model")
    args = parser.parse_args()

    merge_fsdp_checkpoint(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
