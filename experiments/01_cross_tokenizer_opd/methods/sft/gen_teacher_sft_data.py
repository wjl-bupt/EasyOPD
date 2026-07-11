#!/usr/bin/env python3
"""
Generate teacher SFT data aligned with KDFlow pipeline.

Pipeline (matching KDFlow exactly):
1. Load raw dataset (mixed_math_code_10k_with_source) with ground truth labels
2. Generate N trajectories per question via SGLang /v1/chat/completions API
3. Verify answer correctness (math: number/boxed matching; code: format check)
4. Apply quality filters (length, repetition)
5. Select shortest correct response per question
6. Output parquet with 'messages' column for verl MultiTurnSFTDataset
"""

import argparse
import asyncio
import aiohttp
import json
import os
import re
import sys
import time
import logging
from collections import defaultdict
from typing import List, Dict, Optional

import importlib.util
import pandas as pd
from datasets import load_from_disk
from tqdm.asyncio import tqdm_asyncio

# Direct import of reward_score modules (avoid verl/__init__.py which needs tensordict)
EASYOPD_ROOT = "/path/to/EasyOPD"

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_math_mod = _load_module('math_score', os.path.join(EASYOPD_ROOT, 'verl/utils/reward_score/math.py'))
_gsm8k_mod = _load_module('gsm8k_score', os.path.join(EASYOPD_ROOT, 'verl/utils/reward_score/gsm8k.py'))
math_compute_score = _math_mod.compute_score
gsm8k_compute_score = _gsm8k_mod.compute_score

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Source type classification (aligned with KDFlow)
# ============================================================================
MATH_SOURCES = {"gsm8k", "math_minus_500", "open_math_instruct", "orca_math", "openr1_math"}
CODE_SOURCES = {"open_code_instruct", "kodcode", "taco", "code_contests", "apps"}
GSM8K_SOURCES = {"gsm8k"}
BOXED_MATH_SOURCES = {"math_minus_500", "open_math_instruct", "openr1_math"}
ORCA_MATH_SOURCES = {"orca_math"}


def is_math_source(source: str) -> bool:
    return source in MATH_SOURCES


def is_code_source(source: str) -> bool:
    return source in CODE_SOURCES


# ============================================================================
# Answer verification (aligned with KDFlow)
# ============================================================================

def _extract_ground_truth_answer(ground_truth: str) -> tuple:
    """Extract answer from ground_truth. Returns (answer_str, format_type)."""
    if not ground_truth:
        return "", "math"

    # GSM8K format: #### <number>
    gsm_match = re.findall(r'####\s*(\-?[0-9.,]+)', ground_truth)
    if gsm_match:
        return gsm_match[-1].replace(',', ''), "gsm8k"

    # MATH format: \boxed{...}
    boxed_match = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', ground_truth)
    if boxed_match:
        return boxed_match[-1], "math"

    # Short string (likely already the answer)
    if len(ground_truth) < 50 and '\n' not in ground_truth:
        if re.match(r'^-?\d+\.?\d*$', ground_truth.strip()):
            return ground_truth.strip(), "gsm8k"
        return ground_truth.strip(), "math"

    # Fallback: last number
    numbers = re.findall(r'(\-?[0-9.,]+)', ground_truth)
    if numbers:
        return numbers[-1].replace(',', ''), "gsm8k"

    return ground_truth, "math"


def verify_response(response: str, label: str, source: str) -> bool:
    """Verify if a response is correct given the ground truth label.

    For math: checks numerical/symbolic equivalence
    For code: format check only (has code block)
    """
    if not response or not response.strip():
        return False

    if is_code_source(source):
        # For code: just check it has some code content
        has_code = '```' in response or 'def ' in response or 'class ' in response
        return has_code and len(response.strip()) > 50

    if not label:
        # No label to verify against - accept all for code, reject for math
        return is_code_source(source)

    answer, format_type = _extract_ground_truth_answer(label)

    if format_type == "gsm8k":
        score = gsm8k_compute_score(response, answer, method="flexible")
    else:
        score = math_compute_score(response, answer)

    return score > 0.5


# ============================================================================
# Quality filters (aligned with KDFlow)
# ============================================================================

def compute_repetition_ratio(text: str, ngram_size: int = 10) -> float:
    """Compute ratio of repeated n-grams. Higher = more repetitive."""
    if not text or len(text) < ngram_size * 2:
        return 0.0
    words = text.split()
    if len(words) < ngram_size * 2:
        return 0.0
    ngrams = [tuple(words[i:i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def is_quality_response(response: str, source: str,
                        min_length: int = 20,
                        max_length: int = 4000,
                        max_rep_ratio: float = 0.30) -> tuple:
    """Check if response passes quality filters. Returns (passes, reason)."""
    if not response or not response.strip():
        return False, "empty"

    stripped = response.strip()
    length = len(stripped)

    if length < min_length:
        return False, "too_short"
    if length > max_length:
        return False, "too_long"

    rep_ratio = compute_repetition_ratio(stripped)
    if rep_ratio > max_rep_ratio:
        return False, f"repetitive({rep_ratio:.2f})"

    return True, "ok"


# ============================================================================
# SGLang async client (aligned with KDFlow's approach)
# ============================================================================

class SGLangClient:
    """Async SGLang client using /v1/chat/completions API."""

    def __init__(self, base_url: str, max_concurrent: int = 256, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.max_concurrent = max_concurrent
        self.timeout = timeout

    async def _post(self, session, payload, semaphore):
        max_retries = 3
        for attempt in range(max_retries):
            async with semaphore:
                try:
                    async with session.post(
                        self.chat_url, json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            logger.warning(f"HTTP {resp.status}: {text[:200]}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return {"error": text, "choices": []}
                        return await resp.json()
                except Exception as e:
                    logger.warning(f"Request error (attempt {attempt+1}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {"error": str(e), "choices": []}

    def chat_batch(
        self,
        messages_list: List[List[Dict]],
        max_new_tokens: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        n: int = 1,
        desc: str = "Generating",
    ) -> List[List[str]]:
        """Send batch chat requests. Returns list of list of generated texts."""
        return asyncio.run(self._chat_batch_async(
            messages_list, max_new_tokens, temperature, top_p, n, desc
        ))

    async def _chat_batch_async(self, messages_list, max_new_tokens, temperature, top_p, n, desc):
        semaphore = asyncio.Semaphore(self.max_concurrent)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent + 50)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for messages in messages_list:
                payload = {
                    "model": "default",
                    "messages": messages,
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "n": n,
                }
                tasks.append(self._post(session, payload, semaphore))

            results = await tqdm_asyncio.gather(*tasks, desc=desc)

        outputs = []
        for r in results:
            choices = r.get("choices", [])
            texts = [c.get("message", {}).get("content", "") for c in choices]
            # Pad if fewer than n
            while len(texts) < n:
                texts.append("")
            outputs.append(texts)
        return outputs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate teacher SFT data (KDFlow-aligned)")
    parser.add_argument("--raw_dataset", type=str, required=True,
                        help="Path to mixed_math_code_10k_with_source dataset")
    parser.add_argument("--output_parquet", type=str, required=True,
                        help="Output parquet path")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n_trajectories", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_concurrent", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--min_response_length", type=int, default=20)
    parser.add_argument("--max_response_length", type=int, default=4000)
    parser.add_argument("--max_repetition_ratio", type=float, default=0.30)
    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset from {args.raw_dataset}")
    ds = load_from_disk(args.raw_dataset)
    total_questions = len(ds)
    logger.info(f"  Total questions: {total_questions}")
    logger.info(f"  Features: {list(ds.features.keys())}")

    # Initialize client
    client = SGLangClient(
        base_url=args.base_url,
        max_concurrent=args.max_concurrent,
        timeout=300,
    )

    # Prepare messages for generation
    # KDFlow uses /v1/chat/completions which applies chat template server-side
    all_messages = []
    all_labels = []
    all_sources = []
    for i in range(total_questions):
        sample = ds[i]
        messages = sample["messages"]  # [{"role": "user", "content": "..."}]
        all_messages.append(messages)
        all_labels.append(sample.get("label", ""))
        all_sources.append(sample.get("source", "unknown"))

    logger.info(f"\n{'='*60}")
    logger.info(f"Teacher Response Generation (KDFlow-aligned)")
    logger.info(f"{'='*60}")
    logger.info(f"  Questions: {total_questions}")
    logger.info(f"  Trajectories per question: {args.n_trajectories}")
    logger.info(f"  Temperature: {args.temperature}, Top-p: {args.top_p}")
    logger.info(f"  Max new tokens: {args.max_new_tokens}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"{'='*60}\n")

    # Generate in batches
    all_outputs = [None] * total_questions  # List of List[str]
    num_batches = (total_questions + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total_questions)
        batch_messages = all_messages[batch_start:batch_end]

        logger.info(f"Batch {batch_idx+1}/{num_batches} (questions {batch_start}-{batch_end-1})")

        batch_outputs = client.chat_batch(
            messages_list=batch_messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            n=args.n_trajectories,
            desc=f"Batch {batch_idx+1}/{num_batches}",
        )

        for i, outputs in enumerate(batch_outputs):
            all_outputs[batch_start + i] = outputs

        elapsed = time.time() - start_time
        done = batch_end
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total_questions - done) / speed if speed > 0 else 0
        logger.info(f"  Done: {done}/{total_questions}, Speed: {speed:.1f} q/s, ETA: {eta/60:.1f} min")

    # --- Verify and filter responses ---
    logger.info(f"\n{'='*60}")
    logger.info(f"Verifying and filtering responses...")
    logger.info(f"{'='*60}")

    stats = {
        "total_responses": 0,
        "correct": 0,
        "quality_passed": 0,
        "selected": 0,
        "filter_reasons": defaultdict(int),
        "by_source": defaultdict(lambda: {"total": 0, "correct": 0, "selected": 0}),
    }

    selected_records = []

    for q_idx in range(total_questions):
        messages = all_messages[q_idx]
        label = all_labels[q_idx]
        source = all_sources[q_idx]
        trajectories = all_outputs[q_idx]

        best_response = None
        best_length = float('inf')

        for response in trajectories:
            stats["total_responses"] += 1
            stats["by_source"][source]["total"] += 1

            if not response or not response.strip():
                stats["filter_reasons"]["empty"] += 1
                continue

            # Step 1: Verify correctness
            is_correct = verify_response(response, label, source)
            if not is_correct:
                stats["filter_reasons"]["incorrect"] += 1
                continue

            stats["correct"] += 1
            stats["by_source"][source]["correct"] += 1

            # Step 2: Quality filter
            passes, reason = is_quality_response(
                response, source,
                min_length=args.min_response_length,
                max_length=args.max_response_length,
                max_rep_ratio=args.max_repetition_ratio,
            )
            if not passes:
                stats["filter_reasons"][reason] += 1
                continue

            stats["quality_passed"] += 1

            # Step 3: Select shortest correct response
            resp_len = len(response.strip())
            if resp_len < best_length:
                best_length = resp_len
                best_response = response.strip()

        if best_response is not None:
            stats["selected"] += 1
            stats["by_source"][source]["selected"] += 1
            # Build messages in verl format: [user_msg, assistant_msg]
            full_messages = list(messages) + [{"role": "assistant", "content": best_response}]
            selected_records.append({"messages": full_messages})

    # --- Print statistics ---
    logger.info(f"\n  Statistics:")
    logger.info(f"    Total responses: {stats['total_responses']}")
    logger.info(f"    Correct: {stats['correct']} ({stats['correct']/max(stats['total_responses'],1)*100:.1f}%)")
    logger.info(f"    Quality passed: {stats['quality_passed']} ({stats['quality_passed']/max(stats['total_responses'],1)*100:.1f}%)")
    logger.info(f"    Selected (best per question): {stats['selected']} ({stats['selected']/total_questions*100:.1f}% of questions)")

    logger.info(f"\n  Filter reasons:")
    for reason, count in sorted(stats["filter_reasons"].items(), key=lambda x: -x[1]):
        logger.info(f"    {reason}: {count}")

    # Per-source breakdown
    math_count = sum(1 for r in selected_records
                     if any(is_math_source(all_sources[i])
                            for i, m in enumerate(all_messages)
                            if m == r["messages"][:-1]))
    code_count = len(selected_records) - math_count

    logger.info(f"\n  Per-source summary:")
    for source in sorted(stats["by_source"].keys()):
        s = stats["by_source"][source]
        logger.info(f"    {source:<25} total={s['total']:>5}  correct={s['correct']:>5}  selected={s['selected']:>5}")

    logger.info(f"\n  Final dataset: {len(selected_records)} records (math≈{math_count}, code≈{code_count})")

    # --- Save output ---
    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
    df = pd.DataFrame(selected_records)
    df.to_parquet(args.output_parquet)
    logger.info(f"\n  Saved to: {args.output_parquet}")
    logger.info(f"  Shape: {df.shape}")

    # Save generation config
    config_path = args.output_parquet.replace(".parquet", "_config.json")
    config = {
        "raw_dataset": args.raw_dataset,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n_trajectories": args.n_trajectories,
        "max_new_tokens": args.max_new_tokens,
        "min_response_length": args.min_response_length,
        "max_response_length": args.max_response_length,
        "max_repetition_ratio": args.max_repetition_ratio,
        "total_questions": total_questions,
        "total_responses": stats["total_responses"],
        "correct_responses": stats["correct"],
        "quality_passed": stats["quality_passed"],
        "selected": stats["selected"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"  Config saved to: {config_path}")

    total_time = time.time() - start_time
    logger.info(f"\n  Total time: {total_time/60:.1f} min")
    logger.info(f"  Speed: {total_questions/total_time:.1f} questions/sec")


if __name__ == "__main__":
    main()
