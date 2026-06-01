"""OPSA safety evaluation entrypoint.

Run safety evaluation on five public benchmarks:

    * HarmBench       (``walledai/HarmBench``)
    * XSTest          (``walledai/XSTest``)
    * WildJailbreak   (``allenai/wildjailbreak``)
    * StrongREJECT    (``walledai/StrongREJECT``)
    * WildBenign      (``allenai/wildjailbreak`` benign split)

Generation uses vLLM and is scored by a guard model
(:class:`LlamaGuardModel` or :class:`WildGuardModel`).

Example::

    python examples/opsa/evaluation/run_safety_eval.py \\
        --model_path outputs/opsa/checkpoint \\
        --datasets harmbench xstest \\
        --output_dir eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local import (works whether evaluation/ is a package or a flat directory).
try:
    from .guard_model import create_guard_model
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from guard_model import create_guard_model  # type: ignore


logger = logging.getLogger("opsa.eval")


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "harmbench": {
        "source": "walledai/HarmBench",
        "max_samples": 400,
        "subsets": ["standard", "contextual", "copyright"],
    },
    "xstest": {
        "source": "walledai/XSTest",
        "split": "test",
        "max_samples": 450,
    },
    "wildjailbreak": {
        "source": "allenai/wildjailbreak",
        "split": "eval",
        "filter": {"data_type": "adversarial_harmful"},
        "max_samples": 10000,
        "prompt_key": "adversarial",
    },
    "strongreject": {
        "source": "walledai/StrongREJECT",
        "max_samples": 500,
    },
    "wildbenign": {
        "source": "allenai/wildjailbreak",
        "split": "eval",
        "filter": {"data_type": "adversarial_benign"},
        "max_samples": 210,
        "prompt_key": "adversarial",
    },
}


# Candidate keys that hold the prompt text (in priority order).
_PROMPT_KEYS = ("prompt", "question", "behavior", "input", "text", "instruction")
# Candidate keys that hold a category label.
_CATEGORY_KEYS = ("category", "type", "harm_category", "tag", "subset")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
@dataclass
class EvalSample:
    prompt: str
    category: str = ""


def _extract_prompt(row: Dict[str, Any], explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key and row.get(explicit_key):
        return str(row[explicit_key])
    for key in _PROMPT_KEYS:
        value = row.get(key)
        if value:
            return str(value)
    return None


def _extract_category(row: Dict[str, Any]) -> str:
    for key in _CATEGORY_KEYS:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _matches_filter(row: Dict[str, Any], filt: Optional[Dict[str, Any]]) -> bool:
    if not filt:
        return True
    return all(row.get(k) == v for k, v in filt.items())


def load_dataset_samples(name: str) -> List[EvalSample]:
    """Load a benchmark by name and return a list of :class:`EvalSample`."""
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset {name!r}. Choices: {list(DATASET_CONFIG)}")

    cfg = DATASET_CONFIG[name]
    source: str = cfg["source"]
    split: str = cfg.get("split", "train")
    subsets: Optional[List[str]] = cfg.get("subsets")
    filt: Optional[Dict[str, Any]] = cfg.get("filter")
    prompt_key: Optional[str] = cfg.get("prompt_key")
    max_samples: int = cfg.get("max_samples", 0) or 0

    from datasets import load_dataset  # type: ignore

    rows: List[Dict[str, Any]] = []
    if subsets:
        for subset in subsets:
            try:
                ds = load_dataset(source, subset, split=split if split else "train")
            except Exception:  # noqa: BLE001 - best-effort fallback
                ds = load_dataset(source, subset)
                ds = ds[next(iter(ds.keys()))]
            for row in ds:
                row = dict(row)
                row.setdefault("subset", subset)
                rows.append(row)
    else:
        try:
            ds = load_dataset(source, split=split)
        except Exception:  # noqa: BLE001
            ds = load_dataset(source)
            ds = ds[next(iter(ds.keys()))]
        rows.extend(dict(r) for r in ds)

    samples: List[EvalSample] = []
    for row in rows:
        if not _matches_filter(row, filt):
            continue
        prompt = _extract_prompt(row, prompt_key)
        if not prompt:
            continue
        samples.append(EvalSample(prompt=prompt, category=_extract_category(row)))

    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    logger.info("Loaded %d samples from %s", len(samples), name)
    return samples


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def build_chat_prompts(prompts: List[str], tokenizer) -> List[str]:
    """Apply the model's chat template to each user prompt."""
    formatted: List[str] = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted.append(text)
    return formatted


def generate_responses(
    llm,
    sampling_params,
    chat_prompts: List[str],
) -> List[str]:
    outputs = llm.generate(chat_prompts, sampling_params)
    # vLLM preserves request order in the returned list.
    return [out.outputs[0].text for out in outputs]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OPSA safety evaluation runner.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model id or local path.")
    parser.add_argument("--datasets", nargs="+",
                        default=["harmbench", "xstest", "wildjailbreak", "strongreject", "wildbenign"],
                        help="Datasets to evaluate.")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--guard_model", type=str, default="wildguard",
                        choices=["llamaguard", "wildguard"])
    parser.add_argument("--guard_model_path", type=str, default=None,
                        help="Override the default guard model path.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_model_len", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load datasets up front so failures fail fast -----------------------
    datasets: Dict[str, List[EvalSample]] = {}
    for name in args.datasets:
        datasets[name] = load_dataset_samples(name)

    # ---- Load policy model (vLLM) ------------------------------------------
    from transformers import AutoTokenizer  # type: ignore
    from vllm import LLM, SamplingParams  # type: ignore

    logger.info("Loading policy model from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # ---- Generate responses for all datasets, then free the policy model ---
    generations: Dict[str, Dict[str, Any]] = {}
    for name, samples in datasets.items():
        if not samples:
            generations[name] = {"prompts": [], "categories": [], "runs": []}
            continue
        prompts = [s.prompt for s in samples]
        categories = [s.category for s in samples]
        chat_prompts = build_chat_prompts(prompts, tokenizer)
        runs: List[List[str]] = []
        for run_idx in range(args.num_runs):
            logger.info("[%s] generation run %d/%d", name, run_idx + 1, args.num_runs)
            runs.append(generate_responses(llm, sampling_params, chat_prompts))
        generations[name] = {"prompts": prompts, "categories": categories, "runs": runs}

    # Release policy model memory before loading the guard model.
    del llm
    try:
        import gc
        import torch  # type: ignore

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    # ---- Load guard model and score ----------------------------------------
    guard_kwargs: Dict[str, Any] = {
        "tp_size": args.tp_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.guard_model_path:
        guard_kwargs["model_path"] = args.guard_model_path

    logger.info("Loading guard model: %s", args.guard_model)
    guard = create_guard_model(args.guard_model, **guard_kwargs)

    summary: Dict[str, Dict[str, Any]] = {}
    for name, gen in generations.items():
        prompts = gen["prompts"]
        categories = gen["categories"]
        runs = gen["runs"]
        if not prompts:
            result_payload = {
                "model_path": args.model_path,
                "dataset": name,
                "total_samples": 0,
                "unsafe_count": 0,
                "asr": 0.0,
                "num_runs": args.num_runs,
                "results": [],
            }
        else:
            per_run_scores: List[List[int]] = []
            for run_idx, responses in enumerate(runs):
                logger.info("[%s] guard scoring run %d/%d", name, run_idx + 1, args.num_runs)
                per_run_scores.append(guard.score_responses(prompts, responses))

            results: List[Dict[str, Any]] = []
            unsafe_count = 0
            for i, prompt in enumerate(prompts):
                run_scores = [per_run_scores[r][i] for r in range(args.num_runs)]
                is_unsafe = int(any(run_scores))
                unsafe_count += is_unsafe
                first_responses = [runs[r][i] for r in range(args.num_runs)]
                results.append(
                    {
                        "prompt": prompt,
                        "response": first_responses[0] if args.num_runs == 1 else first_responses,
                        "is_unsafe": is_unsafe,
                        "is_unsafe_per_run": run_scores,
                        "category": categories[i],
                    }
                )
            total = len(results)
            asr = unsafe_count / total if total else 0.0
            result_payload = {
                "model_path": args.model_path,
                "dataset": name,
                "total_samples": total,
                "unsafe_count": unsafe_count,
                "asr": asr,
                "num_runs": args.num_runs,
                "results": results,
            }

        out_path = output_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)
        logger.info(
            "[%s] ASR=%.4f (%d/%d) -> %s",
            name,
            result_payload["asr"],
            result_payload["unsafe_count"],
            result_payload["total_samples"],
            out_path,
        )
        summary[name] = {
            "total_samples": result_payload["total_samples"],
            "unsafe_count": result_payload["unsafe_count"],
            "asr": result_payload["asr"],
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "guard_model": args.guard_model,
                "num_runs": args.num_runs,
                "datasets": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
