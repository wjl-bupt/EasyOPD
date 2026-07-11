"""Fix Phi-4-mini data: reward_model must be dict with ground_truth key."""
import sys
sys.path.insert(0, "/path/to/EasyOPD")

from datasets import load_from_disk
from transformers import AutoTokenizer
import pandas as pd
import os

# Load original dataset
ds = load_from_disk('/path/to/workspace/workspace/dataset/mixed_math_code_10k')
print(f"Dataset: {len(ds)} rows")

# Prepare data with correct format
records = []
for i, row in enumerate(ds):
    messages = row['messages']
    user_msgs = [m for m in messages if m['role'] != 'assistant']
    
    label = row.get('label', '')
    data_source = 'math' if '####' in str(label) else 'code'
    
    records.append({
        'prompt': user_msgs,
        'data_source': data_source,
        'reward_model': {'ground_truth': str(label) if label else ''},
    })

# Split
val_size = 50
train_records = records[val_size:]
val_records = records[:val_size]

# Save
output_dir = '/path/to/EasyOPD/experiments/benchmark/data_phi4mini'
os.makedirs(output_dir, exist_ok=True)

train_df = pd.DataFrame(train_records)
val_df = pd.DataFrame(val_records)
train_df.to_parquet(os.path.join(output_dir, 'train.parquet'))
val_df.to_parquet(os.path.join(output_dir, 'val.parquet'))

print(f"Train: {len(train_df)} samples")
print(f"Val: {len(val_df)} samples")
print(f"reward_model type: {type(train_df['reward_model'].iloc[0])}")
print(f"reward_model[0]: {train_df['reward_model'].iloc[0]}")
print(f"prompt type: {type(train_df['prompt'].iloc[0])}")
print("DONE")
