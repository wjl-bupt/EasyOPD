"""Generate teacher responses using Qwen2.5-7B-Instruct for SFT training."""
import os, sys, json, argparse
import pandas as pd
sys.path.insert(0, "/path/to/EasyOPD")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="/path/to/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, default="/path/to/EasyOPD/experiments/benchmark/data_phi4mini/train.parquet")
    parser.add_argument("--output_dir", type=str, default="/path/to/EasyOPD/experiments/benchmark/teacher_sft_data")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} samples")
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    from transformers import AutoTokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)

    prompts_for_vllm = []
    for idx, row in df.iterrows():
        messages = row['prompt']
        if isinstance(messages, str):
            messages = json.loads(messages)
        text = teacher_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts_for_vllm.append(text)
    print(f"Prepared {len(prompts_for_vllm)} prompts")

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.teacher_model, tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.90,
              enforce_eager=True)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, top_p=0.95)

    print("Starting generation...")
    outputs = llm.generate(prompts_for_vllm, sampling_params)
    print(f"Generated {len(outputs)} responses")

    sft_data = []
    for idx, (output, (_, row)) in enumerate(zip(outputs, df.iterrows())):
        teacher_response = output.outputs[0].text.strip()
        messages = row['prompt']
        if isinstance(messages, str):
            messages = json.loads(messages)
        full_messages = list(messages) + [{"role": "assistant", "content": teacher_response}]
        gt = row['reward_model'].get('ground_truth', '') if isinstance(row['reward_model'], dict) else ''
        sft_data.append({"messages": full_messages, "ground_truth": gt, "data_source": row.get('data_source', 'math'), "teacher_response": teacher_response})

    output_path = os.path.join(args.output_dir, "teacher_sft_train.jsonl")
    with open(output_path, 'w') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(sft_data)} samples to {output_path}")

    rl_records = []
    for item in sft_data:
        rl_records.append({'prompt': item['messages'][:-1], 'data_source': item['data_source'], 'reward_model': {'ground_truth': item['ground_truth']}})
    rl_df = pd.DataFrame(rl_records)
    rl_df.to_parquet(os.path.join(args.output_dir, "train_for_rl.parquet"))
    print(f"Saved RL-compatible data")

if __name__ == "__main__":
    main()
