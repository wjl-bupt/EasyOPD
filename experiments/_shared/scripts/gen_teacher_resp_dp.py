"""Generate teacher responses using data-parallel vLLM instances across multiple GPUs."""
import os
import json
import argparse
import multiprocessing as mp
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer


def generate_shard(args):
    """Each worker generates responses for its shard on a specific GPU."""
    rank, shard_prompts, shard_indices, teacher_model, teacher_tp, gpu_offset, data_dir, max_model_len, temperature, max_tokens, top_p = args
    # Each DP worker uses teacher_tp consecutive GPUs
    gpu_start = gpu_offset + rank * teacher_tp
    gpu_ids = ','.join(str(gpu_start + i) for i in range(teacher_tp))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    from vllm import LLM, SamplingParams
    llm = LLM(model=teacher_model, tensor_parallel_size=teacher_tp,
              trust_remote_code=True, max_model_len=max_model_len, gpu_memory_utilization=0.9)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p)

    outputs = llm.generate(shard_prompts, sampling_params)
    print(f'[Worker {rank}] Generated {len(outputs)} responses on GPU {gpu_ids}')

    # Save shard results to temp file
    shard_results = []
    for idx, output in zip(shard_indices, outputs):
        shard_results.append((idx, output.outputs[0].text.strip()))

    shard_path = os.path.join(data_dir, f'_shard_{rank}.json')
    with open(shard_path, 'w') as f:
        json.dump(shard_results, f, ensure_ascii=False)
    return shard_path


def main():
    parser = argparse.ArgumentParser(description="Generate teacher responses with DP parallelism")
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--raw_dataset", type=str, required=True)
    parser.add_argument("--output_parquet", type=str, required=True)
    parser.add_argument("--teacher_tp", type=int, default=1)
    parser.add_argument("--teacher_dp", type=int, default=8)
    parser.add_argument("--gpu_offset", type=int, default=0)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    data_dir = os.path.dirname(args.output_parquet)
    os.makedirs(data_dir, exist_ok=True)

    # Load raw dataset
    ds = load_from_disk(args.raw_dataset)
    print(f'Loaded {len(ds)} samples from raw dataset')

    # Extract user prompts
    prompts_messages = []
    for item in ds:
        user_msgs = [m for m in item['messages'] if m['role'] == 'user']
        prompts_messages.append(user_msgs)

    # Prepare prompts for vLLM
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    prompts_for_vllm = []
    for msgs in prompts_messages:
        text = teacher_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts_for_vllm.append(text)
    print(f'Prepared {len(prompts_for_vllm)} prompts for teacher generation')

    # Split data into DP shards
    n = len(prompts_for_vllm)
    shard_size = (n + args.teacher_dp - 1) // args.teacher_dp
    worker_args = []
    for rank in range(args.teacher_dp):
        start = rank * shard_size
        end = min(start + shard_size, n)
        shard_prompts = prompts_for_vllm[start:end]
        shard_indices = list(range(start, end))
        worker_args.append((
            rank, shard_prompts, shard_indices,
            args.teacher_model, args.teacher_tp, args.gpu_offset,
            data_dir, args.max_model_len, args.temperature, args.max_tokens, args.top_p
        ))

    print(f'Launching {args.teacher_dp} DP workers (TP={args.teacher_tp} each), {shard_size} samples per shard')

    # Launch workers in parallel using spawn (non-daemon Process to allow vLLM sub-processes)
    mp.set_start_method('spawn', force=True)
    processes = []
    for w_args in worker_args:
        p = mp.Process(target=generate_shard, args=(w_args,), daemon=False)
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process {p.pid} failed with exit code {p.exitcode}")

    # Merge all shards
    print('Merging shard results...')
    all_responses = [None] * n
    for rank in range(args.teacher_dp):
        shard_path = os.path.join(data_dir, f'_shard_{rank}.json')
        with open(shard_path, 'r') as f:
            shard_results = json.load(f)
        for idx, response in shard_results:
            all_responses[idx] = response
        os.remove(shard_path)

    # Build SFT parquet directly (messages format for verl)
    records = []
    for msgs, response in zip(prompts_messages, all_responses):
        full_messages = list(msgs) + [{'role': 'assistant', 'content': response}]
        records.append({'messages': full_messages})

    df = pd.DataFrame(records)
    df.to_parquet(args.output_parquet)
    print(f'Saved {len(df)} SFT samples to {args.output_parquet}')


if __name__ == "__main__":
    main()
