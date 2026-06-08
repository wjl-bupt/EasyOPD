"""Prepare training data formatted with Phi-4-mini's chat template.

For cross-tokenizer distillation (Simple/SimCT), the student is Phi-4-mini
and the prompts need to be formatted with Phi-4-mini's chat template.
"""
import sys
sys.path.insert(0, "/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD")

from datasets import load_from_disk
from transformers import AutoTokenizer
import pandas as pd
import os

def main():
    # Paths
    src_dataset = "/apdcephfs_cq8/share_1324356/shinejiesun/workspace/dataset/mixed_math_code_10k"
    phi4_model = "/apdcephfs_cq8/share_1324356/shinejiesun/workspace/KDFlow/output/ckpts/phi4-mini-sft-warmup-10k-qwen-lr2e-6/checkpoint-40"
    output_dir = "/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/benchmark/data_phi4mini"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading Phi-4-mini tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(phi4_model, trust_remote_code=True)
    print(f"  Chat template: {tokenizer.chat_template is not None}")
    
    # Load dataset
    print("Loading dataset...")
    ds = load_from_disk(src_dataset)
    print(f"  Total samples: {len(ds)}")
    
    # Prepare data
    records = []
    for i, row in enumerate(ds):
        messages = row["messages"]
        # Keep only user messages (remove assistant response)
        user_msgs = [m for m in messages if m["role"] != "assistant"]
        
        # Format with Phi-4-mini chat template
        try:
            prompt = tokenizer.apply_chat_template(
                user_msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"  Warning: Failed to format row {i}: {e}")
            # Fallback: just use the content directly
            prompt = user_msgs[-1]["content"] if user_msgs else ""
        
        # Determine data source
        label = row.get("label", "")
        if "####" in str(label):
            data_source = "math"
        else:
            data_source = "code"
        
        records.append({
            "prompt": prompt,
            "data_source": data_source,
            "reward_model": "math",
            "ground_truth": str(label) if label else "",
        })
    
    # Split train/val
    val_size = 50
    train_records = records[val_size:]
    val_records = records[:val_size]
    
    # Save as parquet
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_df)} samples)")
    print(f"  Val: {val_path} ({len(val_df)} samples)")
    print(f"\nSample prompt (first 200 chars):")
    print(f"  {train_df['prompt'].iloc[0][:200]}")

if __name__ == "__main__":
    main()
