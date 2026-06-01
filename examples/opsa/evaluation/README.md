# OPSA Safety Evaluation

This module evaluates an OPSA-trained model on five public safety benchmarks
and reports the **Attack Success Rate (ASR)** scored by a guard model.

## Supported benchmarks

| Dataset       | HF source                  | Max samples | Notes                                      |
| ------------- | -------------------------- | ----------- | ------------------------------------------ |
| HarmBench     | `walledai/HarmBench`       | 400         | Subsets: `standard`, `contextual`, `copyright` |
| XSTest        | `walledai/XSTest`          | 450         | Over-refusal probe                         |
| WildJailbreak | `allenai/wildjailbreak`    | 10000       | `data_type=adversarial_harmful`            |
| StrongREJECT  | `walledai/StrongREJECT`    | 500         | Strong jailbreak set                       |
| WildBenign    | `allenai/wildjailbreak`    | 210         | `data_type=adversarial_benign`             |

## Quick start

```bash
# 1. Run on all 5 benchmarks with WildGuard
MODEL_PATH=/path/to/opsa/checkpoint bash examples/opsa/evaluation/run_eval.sh

# 2. Or call the Python entrypoint directly
python examples/opsa/evaluation/run_safety_eval.py \
    --model_path /path/to/opsa/checkpoint \
    --datasets harmbench xstest \
    --output_dir eval_results \
    --tp_size 1 \
    --guard_model wildguard
```

## Custom parameters

The shell wrapper exposes the most useful overrides via environment variables:

```bash
MODEL_PATH=/ckpt \
DATASETS="harmbench strongreject" \
OUTPUT_DIR=eval_out \
TP_SIZE=2 \
GPU_MEM=0.85 \
MAX_TOKENS=4096 \
GUARD_MODEL=llamaguard \
NUM_RUNS=3 \
TEMPERATURE=0.6 \
bash examples/opsa/evaluation/run_eval.sh
```

For full control, run `run_safety_eval.py --help`. Notable flags:

- `--guard_model {llamaguard,wildguard}`: classifier choice (default `wildguard`).
- `--guard_model_path`: override the default HF id of the guard model.
- `--num_runs`: sample multiple completions per prompt; an example counts as
  unsafe if **any** run is judged unsafe.
- `--temperature` / `--top_p`: generation sampling controls.
- `--max_model_len`: vLLM context length for the policy model.

## Guard models

| Guard      | HF id                          | Output parsed                                  |
| ---------- | ------------------------------ | ---------------------------------------------- |
| LlamaGuard | `meta-llama/Llama-Guard-3-8B`  | First line `safe` / `unsafe`                   |
| WildGuard  | `allenai/wildguard`            | `Harmful response: yes/no`                     |

Both are loaded with vLLM and exposed through a unified
`GuardModel.score_responses(prompts, responses) -> list[int]` API
(`0` = safe, `1` = unsafe). The factory
`create_guard_model("wildguard" | "llamaguard", ...)` is exported from
[`guard_model.py`](./guard_model.py) and can be reused standalone:

```bash
python examples/opsa/evaluation/guard_model.py --guard_type wildguard
```

## Output format

For every benchmark, a JSON file is written to `OUTPUT_DIR/<dataset>.json`:

```json
{
  "model_path": "/path/to/opsa/checkpoint",
  "dataset": "harmbench",
  "total_samples": 400,
  "unsafe_count": 12,
  "asr": 0.03,
  "num_runs": 1,
  "results": [
    {
      "prompt": "...",
      "response": "...",
      "is_unsafe": 0,
      "is_unsafe_per_run": [0],
      "category": "standard"
    }
  ]
}
```

A consolidated `summary.json` is also produced:

```json
{
  "model_path": "/path/to/opsa/checkpoint",
  "guard_model": "wildguard",
  "num_runs": 1,
  "datasets": {
    "harmbench":     {"total_samples": 400,   "unsafe_count": 12, "asr": 0.03},
    "xstest":        {"total_samples": 450,   "unsafe_count": 8,  "asr": 0.0178},
    "wildjailbreak": {"total_samples": 10000, "unsafe_count": 421, "asr": 0.0421}
  }
}
```

## Pipeline

1. Load each benchmark via `datasets.load_dataset` with the configured
   filters / subsets.
2. Load the policy model via vLLM and apply the model's chat template
   (`tokenizer.apply_chat_template`) to every prompt.
3. Generate `num_runs` completions per prompt with the requested sampling
   parameters.
4. Free policy-model memory, then load the guard model (also via vLLM) and
   classify every (prompt, response) pair.
5. Aggregate ASR and write per-dataset JSON plus a consolidated summary.
