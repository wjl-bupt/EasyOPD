"""Evaluate a trained model on MBPP and LiveCodeBench-v6 code benchmarks.

Supports data-parallel evaluation: launches multiple vLLM instances (one per GPU)
each processing a shard of the data, then merges results.

Usage:
    python evaluate_code.py --model_path <path> --output_dir <dir> [--benchmarks mbpp,lcb] [--dp_size 8]
"""

import argparse
import json
import os
import sys
import time
import re
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Disable vLLM V1 engine to avoid CUDA fork issues
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

multiprocessing.set_start_method("spawn", force=True)

# Add project root to path
sys.path.insert(0, "/path/to/EasyOPD")

# Default data dir
DATA_DIR = "/path/to/EasyOPD/experiments/_shared/eval_data"


# ============================================================================
# MBPP Evaluation Logic
# ============================================================================

MBPP_SYSTEM_PROMPT = (
    "You are a Python programming assistant. Write a Python function to solve "
    "the given task. Your code must pass the provided test cases. "
    "Only output the Python code, without any explanation."
)


def _extract_code_block(text: str) -> str:
    """Extract code from markdown code block if present, otherwise return as-is.

    Handles both complete ```...``` blocks and incomplete ones (opening
    marker only, no closing marker). The latter happens occasionally when
    models generate truncated outputs.
    """
    if not text:
        return ""
    # Try complete code block first
    pattern = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).rstrip()
    # If only opening marker present without closing, strip the opening line
    stripped = re.sub(r'^```(?:python)?\s*\n', '', text, count=1)
    return stripped.rstrip()


def _mbpp_exec_worker(code_str, result_queue):
    """Module-level worker (kept module-level so it remains picklable for spawn,
    even though we currently use fork for speed)."""
    try:
        exec_globals = {}
        exec(code_str, exec_globals)
        result_queue.put(True)
    except Exception:
        result_queue.put(False)


def _lcb_exec_worker(code_str, inputs_list, outputs_list, per_test_timeout, result_queue):
    """Module-level worker for LCB test execution in a forked process.
    
    Uses fork (like MBPP) instead of subprocess to avoid ~0.5-1s per-sample
    Python interpreter startup overhead. This brings LCB evaluation from
    ~15 min down to ~seconds for 1055 samples.
    """
    import sys
    import io
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError()

    total = len(inputs_list)

    try:
        code_str = code_str.rstrip()
        compiled_code = compile(code_str, "<solution>", "exec")
    except Exception:
        result_queue.put((0, total))
        return

    passed = 0

    for i in range(total):
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(per_test_timeout)
            old_stdin = sys.stdin
            old_stdout = sys.stdout
            sys.stdin = io.StringIO(inputs_list[i])
            capture = io.StringIO()
            sys.stdout = capture
            exec_globals = {"__name__": "__main__"}
            exec(compiled_code, exec_globals)
            sys.stdin = old_stdin
            sys.stdout = old_stdout
            signal.alarm(0)
            actual = capture.getvalue().strip()
            if actual == outputs_list[i]:
                passed += 1
        except (TimeoutError, SystemExit):
            sys.stdin = old_stdin
            sys.stdout = old_stdout
            signal.alarm(0)
        except Exception:
            sys.stdin = old_stdin
            sys.stdout = old_stdout
            signal.alarm(0)

    result_queue.put((passed, total))


# Cache the fork context once. We deliberately use fork here (not spawn) for
# MBPP test execution: fork starts a new process in ~ms, while spawn takes
# ~500ms because it has to re-import all heavy libs (vLLM, torch, ...).
# By the time we reach this evaluation phase the vLLM workers have already
# finished (see the "All workers completed" log line), so forking the main
# process is safe — the test sandbox does not touch CUDA.
_MBPP_FORK_CTX = multiprocessing.get_context("fork")


def _run_mbpp_tests(code: str, test_list: List[str], test_setup_code: str = "",
                     timeout: float = 10.0) -> bool:
    """Execute MBPP test cases against generated code."""
    full_code = ""
    if test_setup_code:
        full_code += test_setup_code + "\n"
    full_code += code + "\n"
    for test in test_list:
        full_code += test + "\n"

    result_queue = _MBPP_FORK_CTX.Queue()
    proc = _MBPP_FORK_CTX.Process(
        target=_mbpp_exec_worker, args=(full_code, result_queue)
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return False

    if result_queue.empty():
        return False

    return result_queue.get()


# ============================================================================
# LiveCodeBench-v6 Evaluation Logic
# ============================================================================

LCB_SYSTEM_PROMPT = (
    "You are a Python programming assistant. Solve the given competitive "
    "programming problem. Read from standard input and write to standard output. "
    "Only output the Python code, without any explanation."
)


def _run_lcb_tests(code: str, test_cases: List[Dict[str, str]],
                    timeout: float = 10.0) -> Tuple[int, int]:
    """Execute LiveCodeBench test cases (stdin/stdout) against generated code.

    Uses fork (same as MBPP) instead of subprocess for speed.
    Returns (num_passed, num_total).
    """
    total = len(test_cases)
    if total == 0:
        return 0, 0

    test_inputs = [tc.get("input", "") for tc in test_cases]
    test_outputs = [tc.get("output", "").strip() for tc in test_cases]
    per_test_timeout = int(timeout)
    overall_timeout = min(timeout * total + 10, 300.0)

    result_queue = _MBPP_FORK_CTX.Queue()
    proc = _MBPP_FORK_CTX.Process(
        target=_lcb_exec_worker,
        args=(code, test_inputs, test_outputs, per_test_timeout, result_queue),
    )
    proc.start()
    proc.join(timeout=overall_timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return 0, total

    if result_queue.empty():
        return 0, total

    return result_queue.get()


# ============================================================================
# vLLM DP Evaluation Engine
# ============================================================================

def evaluate_code_with_vllm(model_path: str, benchmark_name: str, data_path: str,
                            max_tokens: int = 2048, temperature: float = 0.0,
                            tensor_parallel_size: int = 1, dp_size: int = 8):
    """Generate responses using vLLM with data parallelism, then evaluate code correctness.
    
    For MBPP: data_path points to a HF dataset directory (saved with save_to_disk).
    For LCB: data_path points to a directory containing problems.jsonl.
    """
    import subprocess
    import tempfile

    # Create the worker script
    eval_script = '''
import os, sys, json, time
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Monkey-patch: add all_special_tokens_extended for vllm compatibility
# TokenizersBackend (custom transformers) lacks this property that vllm requires.
import transformers as _transformers
if hasattr(_transformers, 'TokenizersBackend'):
    if not hasattr(_transformers.TokenizersBackend, 'all_special_tokens_extended'):
        @property
        def _all_special_tokens_extended(self):
            return self.all_special_tokens
        _transformers.TokenizersBackend.all_special_tokens_extended = _all_special_tokens_extended

# Read config
config_file = sys.argv[1]
output_file = sys.argv[2]
progress_file = sys.argv[3]

with open(config_file) as f:
    config = json.load(f)

model_path = config["model_path"]
benchmark_name = config["benchmark_name"]
max_tokens = config["max_tokens"]
temperature = config["temperature"]
tensor_parallel_size = config["tensor_parallel_size"]
shard_data = config["shard_data"]
gpu_id = config["gpu_id"]

def report_progress(stage, done=0, total=0):
    with open(progress_file, "w") as pf:
        json.dump({"gpu_id": gpu_id, "stage": stage, "done": done, "total": total, "time": time.time()}, pf)

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

print(f"[GPU {gpu_id}] Processing {len(shard_data)} samples", flush=True)
report_progress("loading_model", 0, len(shard_data))

# Load tokenizer to apply chat template
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Build prompts using chat template
MBPP_SYSTEM_PROMPT = (
    "You are a Python programming assistant. Write a Python function to solve "
    "the given task. Your code must pass the provided test cases. "
    "Only output the Python code, without any explanation."
)
LCB_SYSTEM_PROMPT = (
    "You are a Python programming assistant. Solve the given competitive "
    "programming problem. Read from standard input and write to standard output. "
    "Only output the Python code, without any explanation."
)

prompts = []
for sample in shard_data:
    if benchmark_name == "mbpp":
        text = sample["text"]
        test_list = sample["test_list"]
        test_cases_str = "\\n".join(test_list)
        user_content = f"{text}\\n\\nYour code should pass these tests:\\n{test_cases_str}"
        messages = [
            {"role": "system", "content": MBPP_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    elif benchmark_name == "lcb":
        desc = sample["description"]
        messages = [
            {"role": "system", "content": LCB_SYSTEM_PROMPT},
            {"role": "user", "content": desc},
        ]
    else:
        messages = [{"role": "user", "content": sample.get("text", "")}]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts.append(prompt)

# Load vLLM
llm = LLM(
    model=model_path,
    tensor_parallel_size=tensor_parallel_size,
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)

report_progress("generating", 0, len(shard_data))

sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=1.0)
print(f"[GPU {gpu_id}] Generating responses...", flush=True)
start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
gen_time = time.time() - start_time
print(f"[GPU {gpu_id}] Generation completed in {gen_time:.1f}s ({len(shard_data)/gen_time:.1f} samples/s)", flush=True)

report_progress("done", len(shard_data), len(shard_data))

# Collect generated texts
results = []
for i, output in enumerate(outputs):
    response = output.outputs[0].text
    results.append({
        "index": shard_data[i]["_index"],
        "generated": response,
    })

output_data = {"gpu_id": gpu_id, "results": results, "gen_time_s": gen_time}
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

    # Load data
    if benchmark_name == "mbpp":
        from datasets import load_from_disk
        dataset = load_from_disk(data_path)
        if isinstance(dataset, dict):
            data = dataset.get("test", dataset[list(dataset.keys())[0]])
        else:
            data = dataset
        samples = []
        for i in range(len(data)):
            sample = data[i]
            samples.append({
                "_index": i,
                "task_id": sample["task_id"],
                "text": sample["text"],
                "code": sample.get("code", ""),
                "test_list": sample["test_list"],
                "test_setup_code": sample.get("test_setup_code", ""),
            })
    elif benchmark_name == "lcb":
        problems_file = os.path.join(data_path, "problems.jsonl")
        samples = []
        with open(problems_file) as f:
            for i, line in enumerate(f):
                if line.strip():
                    p = json.loads(line)
                    desc = p.get("question_content", p.get("description", p.get("prompt", "")))
                    # Parse test cases
                    all_tc = []
                    pub_tc = p.get("public_test_cases", p.get("test_cases", p.get("tests", [])))
                    if isinstance(pub_tc, str):
                        try:
                            pub_tc = json.loads(pub_tc)
                        except json.JSONDecodeError:
                            pub_tc = []
                    if isinstance(pub_tc, list):
                        all_tc.extend(pub_tc)
                    # Also try private test cases
                    priv_tc = p.get("private_test_cases", [])
                    if isinstance(priv_tc, str):
                        try:
                            priv_tc = json.loads(priv_tc)
                        except json.JSONDecodeError:
                            priv_tc = []
                    if isinstance(priv_tc, list):
                        all_tc.extend(priv_tc)
                    
                    samples.append({
                        "_index": i,
                        "question_id": p.get("question_id", f"lcb_{i}"),
                        "description": desc,
                        "test_cases": all_tc,
                    })
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    total_samples = len(samples)
    print(f"Total samples: {total_samples}")

    # Split into shards
    shards = []
    shard_size = (total_samples + dp_size - 1) // dp_size
    for i in range(dp_size):
        start = i * shard_size
        end = min(start + shard_size, total_samples)
        if start < total_samples:
            shards.append(samples[start:end])

    actual_dp_size = len(shards)
    print(f"Split into {actual_dp_size} shards")

    # Launch subprocesses
    processes = []
    config_files = []
    output_files = []

    for gpu_idx in range(actual_dp_size):
        config_data = {
            "model_path": model_path,
            "benchmark_name": benchmark_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tensor_parallel_size": tensor_parallel_size,
            "shard_data": shards[gpu_idx],
            "gpu_id": gpu_idx,
        }
        config_path = f"/tmp/eval_code_config_{benchmark_name}_{os.getpid()}_{gpu_idx}.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, ensure_ascii=False)
        config_files.append(config_path)

        output_path = f"/tmp/eval_code_result_{benchmark_name}_{os.getpid()}_{gpu_idx}.json"
        output_files.append(output_path)

        progress_path = f"/tmp/eval_code_progress_{benchmark_name}_{os.getpid()}_{gpu_idx}.json"

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

    # Wait with progress
    from tqdm import tqdm
    print(f"\nWaiting for {actual_dp_size} workers to complete...")
    start_time = time.time()

    progress_files = [f"/tmp/eval_code_progress_{benchmark_name}_{os.getpid()}_{i}.json" for i in range(actual_dp_size)]
    pbar = tqdm(total=total_samples, desc=f"Generate {benchmark_name}", unit="sample")
    last_total_done = 0

    while True:
        all_done = all(proc.poll() is not None for proc in processes)
        total_done = 0
        for i in range(actual_dp_size):
            try:
                with open(progress_files[i]) as pf:
                    prog = json.load(pf)
                    total_done += prog["done"]
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        delta = total_done - last_total_done
        if delta > 0:
            pbar.update(delta)
            last_total_done = total_done

        if all_done:
            pbar.update(total_samples - last_total_done)
            break
        time.sleep(1)

    pbar.close()
    gen_time = time.time() - start_time
    print(f"All workers completed in {gen_time:.1f}s")

    # Cleanup
    os.unlink(script_path)
    for pf in progress_files:
        if os.path.exists(pf):
            os.unlink(pf)

    # Merge results
    all_results = []
    failed_gpus = []
    for gpu_idx in range(actual_dp_size):
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
        all_results.extend(shard_result["results"])

    # Sort by original index
    all_results.sort(key=lambda x: x["index"])

    # Now evaluate code correctness
    print(f"\nEvaluating code correctness ({len(all_results)} samples)...")
    correct = 0
    total = 0
    predictions = []

    eval_pbar = tqdm(total=len(all_results), desc=f"Test {benchmark_name}", unit="sample")

    for result in all_results:
        idx = result["index"]
        generated = result["generated"]
        sample = samples[idx]

        # Extract code
        code = _extract_code_block(generated)

        if benchmark_name == "mbpp":
            test_list = sample["test_list"]
            test_setup_code = sample.get("test_setup_code", "")
            passed = _run_mbpp_tests(code, test_list, test_setup_code)
            if passed:
                correct += 1
            total += 1
            predictions.append({
                "task_id": sample["task_id"],
                "text": sample["text"],
                "generated": generated,
                "extracted_code": code,
                "test_list": test_list,
                "passed": passed,
            })
        elif benchmark_name == "lcb":
            test_cases = sample.get("test_cases", [])
            if test_cases:
                num_passed, num_total = _run_lcb_tests(code, test_cases)
                all_passed = (num_passed == num_total) and num_total > 0
            else:
                num_passed, num_total = 0, 0
                all_passed = False
            if all_passed:
                correct += 1
            total += 1
            predictions.append({
                "question_id": sample["question_id"],
                "description": sample["description"][:200],
                "generated": generated,
                "extracted_code": code,
                "num_passed": num_passed,
                "num_total": num_total,
                "all_passed": all_passed,
            })

        eval_pbar.update(1)

    eval_pbar.close()

    accuracy = correct / total * 100 if total > 0 else 0.0
    score = correct / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Results for {benchmark_name}:")
    print(f"  pass@1: {score:.4f} ({accuracy:.2f}%)")
    print(f"  Passed: {correct}/{total}")
    print(f"  Wall time: {gen_time:.1f}s")
    if failed_gpus:
        print(f"  WARNING: Results incomplete due to {len(failed_gpus)} failed worker(s)")
    print(f"{'='*60}")

    return {
        "model_path": model_path,
        "benchmark": benchmark_name,
        "score": score,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "gen_time_s": gen_time,
        "dp_size": actual_dp_size,
        "failed_gpus": failed_gpus,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on code benchmarks (MBPP, LCB-v6)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default=None, help="Name for this model in results")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to dump results. Typically <experiment>/methods/<method>/results/")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help=f"Directory containing benchmark data (default: {DATA_DIR})")
    parser.add_argument("--benchmarks", type=str, default="mbpp,lcb",
                        help="Comma-separated benchmarks: mbpp, lcb (default: mbpp,lcb)")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="TP size")
    parser.add_argument("--dp_size", type=int, default=8, help="Data parallel size (number of GPUs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_name or Path(args.model_path).name
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    all_results = {}

    for bench in benchmarks:
        if bench == "mbpp":
            data_path = os.path.join(args.data_dir, "mbpp")
        elif bench == "lcb":
            data_path = os.path.join(args.data_dir, "live-code-bench-v6")
        else:
            print(f"Unknown benchmark: {bench}, skipping")
            continue

        if not os.path.exists(data_path):
            print(f"Data not found: {data_path}, skipping")
            print(f"  Please run the data preparation script first.")
            continue

        result = evaluate_code_with_vllm(
            model_path=args.model_path,
            benchmark_name=bench,
            data_path=data_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            dp_size=args.dp_size,
        )
        all_results[bench] = {
            "score": result["score"],
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
        }

        # Save detailed predictions
        detail_path = os.path.join(args.output_dir, f"{model_name}_{bench}_details.json")
        with open(detail_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "model_name": model_name,
        "model_path": args.model_path,
        "results": all_results,
    }
    summary_path = os.path.join(args.output_dir, f"{model_name}_code_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*60}")
    for bench, res in all_results.items():
        print(f"  {bench}: pass@1={res['score']:.4f} ({res['accuracy']:.2f}%, {res['correct']}/{res['total']})")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
