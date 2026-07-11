#!/usr/bin/env python3
"""Generate teacher responses for SFT training."""
import os, sys, json, argparse
import pandas as pd
sys.path.insert(0, "/path/to/EasyOPD")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", default="/path/to/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", default="/path/to/EasyOPD/experiments/benchmark/data_phi4mini/train.parquet")
    parser.add_argument("--output_dir", default="/path/to/EasyOPD/experiments/benchmark/teacher_sft_data")
    parser.add_argument("--tp", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} samples")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    prompts = []
    for _, row in df.iterrows():
        msgs = row["prompt"] if not isinstance(row["prompt"], str) else json.loads(row["prompt"])
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.teacher_model, tensor_parallel_size=args.tp, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.85)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=2048, top_p=0.95))
    sft_data = []
    for out, (_, row) in zip(outputs, df.iterrows()):
        resp = out.outputs[0].text.strip()
        msgs = row["prompt"] if not isinstance(row["prompt"], str) else json.loads(row["prompt"])
        full = list(msgs) + [{"role": "assistant", "content": resp}]
        gt = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""
        sft_data.append({"messages": full, "ground_truth": gt, "data_source": row.get("data_source", "math")})
    with open(os.path.join(args.output_dir, "teacher_sft_train.jsonl"), "w") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    rl_recs = [{"prompt": it["messages"][:-1], "data_source": it["data_source"], "reward_model": {"ground_truth": it["ground_truth"]}} for it in sft_data]
    pd.DataFrame(rl_recs).to_parquet(os.path.join(args.output_dir, "train_for_rl.parquet"))
    print(f"Done: {len(sft_data)} samples saved")

if __name__ == "__main__":
    main()
