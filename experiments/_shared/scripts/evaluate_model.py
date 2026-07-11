"""Evaluate a trained model on MATH-500 and GSM8K benchmarks.

Supports data-parallel evaluation: launches multiple vLLM instances (one per GPU)
each processing a shard of the data, then merges results.

Usage:
    python evaluate_model.py --model_path <path> --output_dir <dir> [--benchmarks math500,gsm8k] [--dp_size 8]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Disable vLLM V1 engine to avoid CUDA fork issues
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import pandas as pd

# Add project root to path
sys.path.insert(0, "/path/to/EasyOPD")

from verl.utils.reward_score.math import compute_score as math_compute_score
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score


# Default eval-data dir is shared across all methods/experiments.
# Eval results are method-specific and MUST be passed in via --output_dir
# (typically <experiment>/methods/<method>/results/).
DATA_DIR = "/path/to/EasyOPD/experiments/_shared/eval_data"


def evaluate_with_vllm(model_path: str, eval_data_path: str, benchmark_name: str,
                       max_tokens: int = 2048, temperature: float = 0.0,
                       tensor_parallel_size: int = 1, dp_size: int = 8):
    """Generate responses using vLLM with data parallelism.
    Launches dp_size subprocesses, each on a different GPU, processing a shard of data.
    """
    import subprocess
    import tempfile

    # Create the worker script that each GPU will run
    eval_script = '''
import os, sys, json, time
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
sys.path.insert(0, "/path/to/EasyOPD")

import pandas as pd
from vllm import LLM, SamplingParams
from verl.utils.reward_score.math import compute_score as math_compute_score
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score
import re as _re

def mcq_compute_score(response, ground_truth):
    """SciKnowEval MCQ scoring — faithful to lasgroup/SDPO ``feedback/mcq.py``.

    Strict exact match: the stripped content of the LAST ``<answer>...</answer>``
    block must equal the ground-truth option string (e.g. "A"). Mirrors the
    training reward (``reward_fn.py:mcq_score``) so eval numbers are comparable to
    the SDPO reference.
    """
    if response is None:
        return 0.0
    answer = str(response).split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    answer = answer.strip()
    return float(answer == str(ground_truth))

def tooluse_compute_score(response, ground_truth):
    """Tool-use scoring — faithful to feedback/tooluse.py: correct iff the multiset
    of predicted actions (Action:) and the merged Action-Input JSON both match GT."""
    if response is None:
        return 0.0
    from collections import Counter as _Counter
    try:
        gt_list = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except Exception:
        return 0.0
    if not isinstance(gt_list, list):
        return 0.0
    try:
        gt_actions = [item["Action"] for item in gt_list]
    except Exception:
        return 0.0
    gt_inputs = {}
    for item in gt_list:
        ai = item.get("Action_Input")
        try:
            parsed = json.loads(ai) if isinstance(ai, str) else (ai or {})
        except Exception:
            parsed = {}
        if parsed:
            gt_inputs.update(parsed)
    pred_actions = _re.findall(r"Action:\\s*(\\w+)", response)
    pred_inputs = {}
    for block in _re.findall(r"Action Input:\\s*({.*?})", response, _re.DOTALL):
        try:
            pred_inputs.update(json.loads(block))
        except Exception:
            pass
    return float(_Counter(pred_actions) == _Counter(gt_actions) and pred_inputs == gt_inputs)

def boxed_compute_score(response, ground_truth):
    """Boxed-answer math scoring matching the boxed prompt (used for gsm8k whose
    answers here are in boxed{}, not '#### n'): extract the last boxed{...} and
    compare to ground_truth (exact, then numeric)."""
    if response is None:
        return 0.0
    s = str(response)
    key = chr(92) + "boxed{"  # literal backslash-boxed{ without source escaping
    idx = s.rfind(key)
    if idx < 0:
        return 0.0
    i = idx + len(key)
    depth = 1
    buf = []
    while i < len(s) and depth > 0:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(c)
        i += 1
    pred = "".join(buf).strip()
    gt = str(ground_truth).strip()
    if pred == gt:
        return 1.0
    try:
        return 1.0 if abs(float(pred.replace(",", "")) - float(gt.replace(",", ""))) < 1e-6 else 0.0
    except Exception:
        return 0.0

# Read config from command line args
config_file = sys.argv[1]
output_file = sys.argv[2]
progress_file = sys.argv[3]

with open(config_file) as f:
    config = json.load(f)

model_path = config["model_path"]
eval_data_path = config["eval_data_path"]
benchmark_name = config["benchmark_name"]
max_tokens = config["max_tokens"]
temperature = config["temperature"]
tensor_parallel_size = config["tensor_parallel_size"]
shard_indices = config["shard_indices"]
gpu_id = config["gpu_id"]

def report_progress(stage, done=0, total=0):
    """Write progress to file for the main process to read."""
    with open(progress_file, "w") as pf:
        json.dump({"gpu_id": gpu_id, "stage": stage, "done": done, "total": total, "time": time.time()}, pf)

# Set CUDA visible device
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

df = pd.read_parquet(eval_data_path)
prompts = df["prompt"].tolist()
reward_model_data = df["reward_model"].tolist()

# math/gsm8k prompts are pre-templated strings; sciknoweval/chemistry prompts are
# chat-message lists -> apply the model's chat template to turn them into text.
if len(prompts) > 0 and not isinstance(prompts[0], str):
    from transformers import AutoTokenizer
    _eval_tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = [_eval_tok.apply_chat_template([dict(m) for m in list(p)], tokenize=False, add_generation_prompt=True) for p in prompts]

# Select only this shard's data
shard_prompts = [prompts[i] for i in shard_indices]
shard_reward_data = [reward_model_data[i] for i in shard_indices]
print(f"[GPU {gpu_id}] Processing {len(shard_prompts)} samples (shard of {len(prompts)} total)", flush=True)

report_progress("loading_model", 0, len(shard_prompts))

llm = LLM(
    model=model_path,
    tensor_parallel_size=tensor_parallel_size,
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)

report_progress("generating", 0, len(shard_prompts))

sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=1.0)
print(f"[GPU {gpu_id}] Generating responses...", flush=True)
start_time = time.time()
outputs = llm.generate(shard_prompts, sampling_params)
gen_time = time.time() - start_time
print(f"[GPU {gpu_id}] Generation completed in {gen_time:.1f}s ({len(shard_prompts)/gen_time:.1f} samples/s)", flush=True)

report_progress("scoring", len(shard_prompts), len(shard_prompts))

correct = 0
total = len(outputs)
results = []
for i, output in enumerate(outputs):
    response = output.outputs[0].text
    ground_truth = shard_reward_data[i]["ground_truth"]
    if benchmark_name in ("math500", "math_hard"):
        score = math_compute_score(response, ground_truth)
    elif benchmark_name == "gsm8k":
        # This gsm8k dataset uses \boxed{} answers (not "#### n"), so score with the
        # boxed extractor to stay consistent with the boxed training reward.
        score = boxed_compute_score(response, ground_truth)
    elif benchmark_name in ("chemistry", "sciknoweval", "biology", "material", "physics"):
        score = mcq_compute_score(response, ground_truth)
    elif benchmark_name == "tooluse":
        score = tooluse_compute_score(response, ground_truth)
    else:
        score = 0.0
    correct += score
    results.append({"index": shard_indices[i], "prompt": shard_prompts[i][:200], "response": response[:500], "ground_truth": ground_truth, "score": score})

accuracy = correct / total * 100 if total > 0 else 0.0
print(f"[GPU {gpu_id}] Shard accuracy: {accuracy:.2f}% ({int(correct)}/{total})", flush=True)

report_progress("done", total, total)

output_data = {"gpu_id": gpu_id, "accuracy": accuracy, "correct": int(correct), "total": total, "gen_time_s": gen_time, "results": results}
with open(output_file, "w") as f:
    json.dump(output_data, f, ensure_ascii=False)
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
        f.write(eval_script)
        script_path = f.name

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"Benchmark: {benchmark_name}")
    print(f"Data Parallel: {dp_size} GPUs, Tensor Parallel: {tensor_parallel_size}")
    print(f"{'='*60}")

    # Load data to determine total samples and create shards
    df = pd.read_parquet(eval_data_path)
    total_samples = len(df)
    
    # Split indices into dp_size shards
    all_indices = list(range(total_samples))
    shards = []
    shard_size = (total_samples + dp_size - 1) // dp_size
    for i in range(dp_size):
        start = i * shard_size
        end = min(start + shard_size, total_samples)
        if start < total_samples:
            shards.append(all_indices[start:end])

    actual_dp_size = len(shards)
    print(f"Total samples: {total_samples}, split into {actual_dp_size} shards")

    # Launch subprocesses for each GPU
    processes = []
    config_files = []
    output_files = []

    for gpu_idx in range(actual_dp_size):
        # Write config for this shard
        config_data = {
            "model_path": model_path,
            "eval_data_path": eval_data_path,
            "benchmark_name": benchmark_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tensor_parallel_size": tensor_parallel_size,
            "shard_indices": shards[gpu_idx],
            "gpu_id": gpu_idx,
        }
        config_path = f"/tmp/eval_config_{benchmark_name}_{os.getpid()}_{gpu_idx}.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)
        config_files.append(config_path)

        output_path = f"/tmp/eval_result_{benchmark_name}_{os.getpid()}_{gpu_idx}.json"
        output_files.append(output_path)

        progress_path = f"/tmp/eval_progress_{benchmark_name}_{os.getpid()}_{gpu_idx}.json"

        env = os.environ.copy()
        env["VLLM_USE_V1"] = "0"
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        proc = subprocess.Popen(
            [sys.executable, script_path, config_path, output_path, progress_path],
            env=env,
        )
        processes.append(proc)
        print(f"  Launched worker on GPU {gpu_idx} ({len(shards[gpu_idx])} samples)")

    # Wait for all processes to complete with progress bar
    from tqdm import tqdm
    print(f"\nWaiting for {actual_dp_size} workers to complete...")
    start_time = time.time()

    progress_files = [f"/tmp/eval_progress_{benchmark_name}_{os.getpid()}_{i}.json" for i in range(actual_dp_size)]
    pbar = tqdm(total=total_samples, desc=f"Eval {benchmark_name}", unit="sample")
    last_total_done = 0
    model_loaded = [False] * actual_dp_size

    while True:
        # Check if all processes are done
        all_done = all(proc.poll() is not None for proc in processes)

        # Read progress from each worker
        total_done = 0
        stages = []
        for i in range(actual_dp_size):
            try:
                with open(progress_files[i]) as pf:
                    prog = json.load(pf)
                    total_done += prog["done"]
                    stages.append(prog["stage"])
                    if prog["stage"] != "loading_model":
                        model_loaded[i] = True
            except (FileNotFoundError, json.JSONDecodeError):
                stages.append("starting")

        # Update progress bar
        delta = total_done - last_total_done
        if delta > 0:
            pbar.update(delta)
            last_total_done = total_done

        # Update description with stage info
        loading_count = stages.count("loading_model") + stages.count("starting")
        generating_count = stages.count("generating")
        done_count = stages.count("done") + stages.count("scoring")
        pbar.set_postfix_str(f"loading:{loading_count} gen:{generating_count} done:{done_count}")

        if all_done:
            # Final update
            pbar.update(total_samples - last_total_done)
            break

        time.sleep(1)

    pbar.close()
    total_time = time.time() - start_time
    print(f"All workers completed in {total_time:.1f}s")

    # Cleanup progress files
    for pf in progress_files:
        if os.path.exists(pf):
            os.unlink(pf)

    # Cleanup script
    os.unlink(script_path)

    # Merge results from all shards
    all_correct = 0
    all_total = 0
    all_results = []
    all_gen_time = 0
    failed_gpus = []

    for gpu_idx in range(actual_dp_size):
        # Cleanup config file
        if os.path.exists(config_files[gpu_idx]):
            os.unlink(config_files[gpu_idx])

        if processes[gpu_idx].returncode != 0:
            print(f"  WARNING: GPU {gpu_idx} worker failed (exit code {processes[gpu_idx].returncode})")
            failed_gpus.append(gpu_idx)
            continue

        if not os.path.exists(output_files[gpu_idx]):
            print(f"  WARNING: GPU {gpu_idx} output file not found")
            failed_gpus.append(gpu_idx)
            continue

        with open(output_files[gpu_idx]) as f:
            shard_result = json.load(f)
        os.unlink(output_files[gpu_idx])

        all_correct += shard_result["correct"]
        all_total += shard_result["total"]
        all_results.extend(shard_result["results"])
        all_gen_time = max(all_gen_time, shard_result["gen_time_s"])

    if failed_gpus:
        print(f"  WARNING: {len(failed_gpus)} GPU(s) failed: {failed_gpus}")

    # Sort results by original index
    all_results.sort(key=lambda x: x["index"])

    accuracy = all_correct / all_total * 100 if all_total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Results for {benchmark_name}:")
    print(f"  Accuracy: {accuracy:.2f}% ({all_correct}/{all_total})")
    print(f"  Wall time: {total_time:.1f}s (max gen time per GPU: {all_gen_time:.1f}s)")
    if failed_gpus:
        print(f"  WARNING: Results incomplete due to {len(failed_gpus)} failed worker(s)")
    print(f"{'='*60}")

    return {
        "model_path": model_path,
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": all_correct,
        "total": all_total,
        "gen_time_s": all_gen_time,
        "wall_time_s": total_time,
        "dp_size": actual_dp_size,
        "failed_gpus": failed_gpus,
        "results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on benchmarks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default=None, help="Name for this model in results")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to dump <model>_<bench>_details.json and <model>_summary.json. "
                             "Typically <experiment>/methods/<method>/results/")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help=f"Directory containing *_eval.parquet files (default: {DATA_DIR})")
    parser.add_argument("--benchmarks", type=str, default="math500,gsm8k", help="Comma-separated benchmarks")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="TP size for each vLLM instance")
    parser.add_argument("--dp_size", type=int, default=8, help="Data parallel size (number of GPUs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_name or Path(args.model_path).name
    benchmarks = args.benchmarks.split(",")

    all_results = {}

    for bench in benchmarks:
        bench = bench.strip()
        if bench == "math500":
            eval_path = os.path.join(args.data_dir, "math500_eval.parquet")
        elif bench == "gsm8k":
            eval_path = os.path.join(args.data_dir, "gsm8k_eval.parquet")
        elif bench == "math_hard":
            eval_path = os.path.join(args.data_dir, "math_hard_eval.parquet")
        elif bench in ("chemistry", "sciknoweval"):
            eval_path = os.path.join(args.data_dir, "chemistry_eval.parquet")
        elif bench in ("biology", "material", "physics"):
            # Other SciKnowEval domains (same MCQ format/reward as chemistry).
            eval_path = os.path.join(args.data_dir, f"{bench}_eval.parquet")
        elif bench == "tooluse":
            # Agentic Action/Action-Input task (scored by tooluse_compute_score).
            eval_path = os.path.join(args.data_dir, "tooluse_eval.parquet")
        else:
            print(f"Unknown benchmark: {bench}, skipping")
            continue

        if not os.path.exists(eval_path):
            print(f"Eval data not found: {eval_path}, skipping")
            continue

        result = evaluate_with_vllm(
            model_path=args.model_path,
            eval_data_path=eval_path,
            benchmark_name=bench,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            dp_size=args.dp_size,
        )
        all_results[bench] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
        }

        # Save detailed results
        detail_path = os.path.join(args.output_dir, f"{model_name}_{bench}_details.json")
        with open(detail_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "model_name": model_name,
        "model_path": args.model_path,
        "results": all_results,
    }
    summary_path = os.path.join(args.output_dir, f"{model_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*60}")
    for bench, res in all_results.items():
        print(f"  {bench}: {res['accuracy']:.2f}% ({res['correct']}/{res['total']})")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
