#!/usr/bin/env python3
"""
Unified Model Evaluation Script for EasyOPD - powered by SGLang DP inference engine.
Supports: gsm8k, math500, mbpp, live-code-bench-v6

Architecture:
    - SGLang server provides high-throughput inference via OpenAI-compatible API
    - This script is a lightweight HTTP client that sends requests and evaluates results
    - Server must be started separately (see eval_model_sglang.sh for automation)

Prerequisites:
    # Start SGLang server (recommend DP=8 for high throughput)
    python -m sglang.launch_server --model-path <model_path> --dp-size 8 --tp-size 1 --port 30000

Usage:
    python evaluate_model_sglang.py --model_path /path/to/model --dataset gsm8k
    python evaluate_model_sglang.py --model_path /path/to/model --dataset math500
    python evaluate_model_sglang.py --model_path /path/to/model --dataset mbpp
    python evaluate_model_sglang.py --model_path /path/to/model --dataset live-code-bench-v6
"""

import argparse
import os
import sys
import re
import json
import time
import asyncio
import aiohttp
import logging
import multiprocessing
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from datasets import load_from_disk
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # EasyOPD root
DATASETS_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR), "eval_data"
)  # experiments/_shared/eval_data
DEFAULT_BASE_URL = "http://127.0.0.1:30000"

# Dataset metrics
DATASET_METRICS = {
    "gsm8k": "Accuracy",
    "math500": "Accuracy",
    "mbpp": "pass@1",
    "live-code-bench-v6": "pass@1",
}

# Standard file names
PREDICTIONS_FILE = "predictions.jsonl"
METRICS_FILE = "metrics.json"


# ============================================================================
# SGLang Client - async high-concurrency inference via OpenAI-compatible API
# ============================================================================
class SGLangClient:
    """SGLang inference client optimized for DP=8 data parallelism."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        max_concurrent: int = 256,
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.completions_url = f"{self.base_url}/v1/completions"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._supports_system_role = None
        self._logged_chat_template_kwargs_warning = False

    async def _post(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict,
        semaphore: asyncio.Semaphore,
    ) -> dict:
        """Send a single request with retry."""
        max_retries = 3
        for attempt in range(max_retries):
            async with semaphore:
                try:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            # If server rejects chat_template_kwargs, retry without it
                            if (
                                resp.status in (400, 422)
                                and "chat_template_kwargs" in payload
                                and ("chat_template_kwargs" in text or "Unexpected" in text)
                            ):
                                if not self._logged_chat_template_kwargs_warning:
                                    logger.warning(
                                        "Server does not recognise chat_template_kwargs; "
                                        "falling back to requests without it."
                                    )
                                    self._logged_chat_template_kwargs_warning = True
                                payload_without = {
                                    k: v for k, v in payload.items()
                                    if k != "chat_template_kwargs"
                                }
                                async with session.post(
                                    url,
                                    json=payload_without,
                                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                                ) as resp2:
                                    if resp2.status == 200:
                                        return await resp2.json()
                                    return {"error": await resp2.text(), "choices": []}
                            logger.warning(f"Request failed (HTTP {resp.status}): {text[:200]}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1)
                                continue
                            return {"error": text, "choices": []}
                        return await resp.json()
                except asyncio.TimeoutError:
                    logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return {"error": "timeout", "choices": []}
                except Exception as e:
                    logger.warning(f"Request error: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return {"error": str(e), "choices": []}
        return {"error": "max retries exceeded", "choices": []}

    async def _check_system_role_support(self, model: str = "default") -> bool:
        """Check if the served model supports system role in chat completions."""
        if self._supports_system_role is not None:
            return self._supports_system_role

        test_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi"},
            ],
            "max_tokens": 1,
            "temperature": 0,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_url,
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        self._supports_system_role = True
                        logger.info("System role support: ✓ supported")
                    else:
                        text = await resp.text()
                        if "System role not supported" in text:
                            self._supports_system_role = False
                            logger.info("System role support: ✗ not supported")
                        else:
                            self._supports_system_role = True
        except Exception as e:
            logger.warning(f"System role check failed: {e}, assuming supported")
            self._supports_system_role = True

        return self._supports_system_role

    def _merge_system_into_user(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Merge system role content into the first user message."""
        system_content = ""
        new_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n"
            else:
                new_messages.append(msg)

        if system_content and new_messages:
            first_user = new_messages[0]
            if first_user["role"] == "user":
                new_messages[0] = {
                    "role": "user",
                    "content": system_content.strip() + "\n\n" + first_user["content"],
                }

        return new_messages if new_messages else messages

    async def chat_batch_async(
        self,
        messages_list: List[List[Dict[str, str]]],
        model: str = "default",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """Batch async chat generation via /v1/chat/completions."""
        await self._check_system_role_support(model)

        if not self._supports_system_role:
            messages_list = [self._merge_system_into_user(msgs) for msgs in messages_list]

        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []

        for idx, messages in enumerate(messages_list):
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if temperature > 0 and top_p < 1.0:
                payload["top_p"] = top_p
            if stop:
                payload["stop"] = stop
            tasks.append(payload)

        results = [None] * len(messages_list)

        async with aiohttp.ClientSession() as session:
            async_tasks = [
                self._post(session, self.chat_url, payload, semaphore)
                for payload in tasks
            ]

            if show_progress:
                responses = await tqdm_asyncio.gather(
                    *async_tasks, desc="Inference(chat)", total=len(async_tasks)
                )
            else:
                responses = await asyncio.gather(*async_tasks)

            empty_count = 0
            for idx, resp in enumerate(responses):
                if resp.get("choices"):
                    msg = resp["choices"][0].get("message", {})
                    content = msg.get("content", "")
                    results[idx] = content
                    if not content or not content.strip():
                        empty_count += 1
                else:
                    results[idx] = ""
                    empty_count += 1

            total_count = len(responses)
            if total_count > 0 and empty_count / total_count > 0.5:
                logger.warning(
                    f"🚨 CRITICAL: {empty_count}/{total_count} ({empty_count/total_count*100:.1f}%) "
                    f"responses returned empty output! Check chat template compatibility."
                )

        return results

    def chat_batch(self, messages_list, **kwargs) -> List[str]:
        """Sync wrapper for chat completions."""
        return asyncio.run(self.chat_batch_async(messages_list=messages_list, **kwargs))

    def health_check(self) -> bool:
        """Check if SGLang server is available."""
        import urllib.request
        try:
            url = f"{self.base_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception:
            try:
                url = f"{self.base_url}/v1/models"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.status == 200
            except Exception:
                return False

    def get_model_name(self) -> str:
        """Get the model name served by SGLang."""
        import urllib.request
        try:
            url = f"{self.base_url}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                if data.get("data"):
                    return data["data"][0]["id"]
        except Exception:
            pass
        return "unknown"


# ============================================================================
# Utilities
# ============================================================================

def save_predictions(predictions: list, output_path: str):
    """Save predictions in JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    logger.info(f"Predictions saved to {output_path}")


def check_existing_results(output_dir: str) -> Optional[str]:
    """Check if prediction results already exist."""
    pred_path = os.path.join(output_dir, PREDICTIONS_FILE)
    if os.path.exists(pred_path) and os.path.getsize(pred_path) > 0:
        return pred_path
    return None


def strip_thinking_content(text: str) -> Tuple[str, bool, bool]:
    """Strip <think>...</think> content from model output.

    Returns: (stripped_text, had_thinking, was_truncated)
    """
    if not text:
        return text, False, False

    think_start = text.find("<think>")
    if think_start == -1:
        return text, False, False

    think_end = text.find("</think>")
    if think_end != -1:
        before = text[:think_start].strip()
        after = text[think_end + len("</think>"):].strip()
        stripped = (before + "\n" + after).strip() if before else after
        return stripped, True, False
    else:
        before = text[:think_start].strip()
        if before:
            return before, True, True
        thinking_content = text[think_start + len("<think>"):]
        tail = thinking_content[-500:] if len(thinking_content) > 500 else thinking_content
        return tail, True, True


def _extract_code_block(text: str) -> str:
    """Extract code from markdown code block if present, otherwise return as-is."""
    if not text:
        return ""
    pattern = re.compile(r'```(?:python)?\s*\n(.*?)```', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).rstrip()
    return text.rstrip()


def parse_verl_prompt(prompt: str) -> Tuple[str, str]:
    """Parse verl-format prompt (with Qwen chat template) to extract system and user content.

    Format: <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n

    Returns: (system_content, user_content)
    """
    # Extract user content between <|im_start|>user\n and <|im_end|>
    user_match = re.search(
        r'<\|im_start\|>user\n(.*?)<\|im_end\|>', prompt, re.DOTALL
    )
    system_match = re.search(
        r'<\|im_start\|>system\n(.*?)<\|im_end\|>', prompt, re.DOTALL
    )

    user_content = user_match.group(1).strip() if user_match else prompt
    system_content = system_match.group(1).strip() if system_match else ""

    return system_content, user_content


# ============================================================================
# Math Answer Extraction Utilities
# ============================================================================

def extract_number_from_answer(answer_text: str) -> Optional[float]:
    """Extract number from answer text, supports #### and \\boxed{} formats."""
    if not answer_text:
        return None
    answer_text = answer_text.replace(",", "")

    if "####" in answer_text:
        match = re.search(r'####\s*(-?\d+\.?\d*)', answer_text)
        if match:
            return float(match.group(1))

    boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer_text)
    if boxed_match:
        boxed_content = boxed_match.group(1).replace(",", "").strip()
        numbers = re.findall(r'-?\d+\.?\d*', boxed_content)
        if numbers:
            return float(numbers[-1])

    numbers = re.findall(r'-?\d+\.?\d*', answer_text)
    if numbers:
        return float(numbers[-1])

    return None


def normalize_answer_string(answer: str) -> str:
    """Normalize a LaTeX answer string for comparison."""
    if not answer:
        return ""
    s = answer.strip()
    s = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', s)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    s = s.replace('\\left', '').replace('\\right', '')
    s = re.sub(r'\\[,;:!]', '', s)
    s = s.replace('\\%', '%')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{...} content from text, handling nested braces."""
    idx = text.rfind('\\boxed')
    if idx == -1:
        return ""

    brace_start = text.find('{', idx)
    if brace_start == -1:
        return ""

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:i]

    return text[brace_start + 1:]


def try_parse_number(s: str) -> Optional[float]:
    """Try to parse a string as a number."""
    s = s.strip().replace(',', '')
    if s.endswith('.'):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def is_math_equivalent(pred: str, gold: str) -> bool:
    """Check if predicted and gold answers are mathematically equivalent.

    Multi-level comparison:
    1. Exact string match (after normalization)
    2. Numeric comparison (with tolerance)
    3. Sympy symbolic comparison (if available)
    """
    if not pred or not gold:
        return False

    # Level 1: Normalized string match
    norm_pred = normalize_answer_string(pred)
    norm_gold = normalize_answer_string(gold)
    if norm_pred == norm_gold:
        return True

    # Level 2: Numeric comparison
    num_pred = try_parse_number(norm_pred)
    num_gold = try_parse_number(norm_gold)
    if num_pred is not None and num_gold is not None:
        return abs(num_pred - num_gold) < 1e-6

    # Level 3: Sympy symbolic comparison
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify, N

        sym_pred = parse_latex(pred)
        sym_gold = parse_latex(gold)

        diff = simplify(sym_pred - sym_gold)
        if diff == 0:
            return True

        try:
            val_pred = complex(N(sym_pred))
            val_gold = complex(N(sym_gold))
            if abs(val_pred - val_gold) < 1e-6:
                return True
        except (TypeError, ValueError):
            pass
    except Exception:
        pass

    return False


# ============================================================================
# GSM8K System Prompt
# ============================================================================

GSM8K_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer after ####."
)

# ============================================================================
# MATH-500 System Prompt
# ============================================================================

MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


# ============================================================================
# GSM8K Evaluator
# ============================================================================

def evaluate_gsm8k(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate GSM8K dataset using 0-shot chat mode with native #### format."""
    logger.info(f"Loading GSM8K dataset ({split} split)...")
    logger.info(f"Eval mode: 0-shot chat (####)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    # Load from HuggingFace dataset (load_from_disk)
    dataset = load_from_disk(dataset_path)
    data = dataset[split]

    if debug:
        data = data.select(range(min(10, len(data))))
        logger.info(f"Debug mode: processing {len(data)} samples only")

    logger.info(f"Dataset size: {len(data)}")

    questions = []
    gold_answers = []
    for i in range(len(data)):
        sample = data[i]
        questions.append(sample['question'])
        gold_answers.append(sample['answer'])

    logger.info(f"Total {len(questions)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # Build chat messages: use our own system prompt (model-agnostic)
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for question in questions:
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": GSM8K_SYSTEM_PROMPT + "\n\n" + question},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ])

    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    # Evaluate
    predictions = []
    correct = 0
    total = 0
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        gold_answer = gold_answers[i]
        question = questions[i]

        stripped, had_thinking, was_truncated = strip_thinking_content(generated)
        pred_number = extract_number_from_answer(stripped)
        gold_number = extract_number_from_answer(gold_answer)

        is_correct = False
        if pred_number is not None and gold_number is not None:
            is_correct = abs(pred_number - gold_number) < 1e-6

        if is_correct:
            correct += 1
        total += 1

        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = pred_number is None and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            'question': question[:300],
            'gold_answer': gold_answer,
            'gold_number': gold_number,
            'generated': generated,
            'pred_number': pred_number,
            'is_correct': is_correct,
            'eval_mode': '0-shot chat (####)',
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults (0-shot chat):")
    logger.info(f"  Correct: {correct}/{total}")
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    return {
        'score': accuracy,
        'correct': correct,
        'total_samples': total,
        'eval_mode': '0-shot chat (####)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
    }


# ============================================================================
# MATH-500 Evaluator
# ============================================================================

def evaluate_math500(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate MATH-500 dataset using 0-shot chat mode with boxed format.

    MATH-500 answers are LaTeX strings; uses multi-level equivalence checking.
    """
    logger.info(f"Loading MATH-500 dataset...")
    logger.info(f"Eval mode: 0-shot chat (boxed)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    # Load from HuggingFace dataset (load_from_disk)
    # MATH-500 is stored as a single split (from HuggingFaceH4/MATH-500)
    dataset_disk_path = os.path.join(dataset_path, "dataset")
    if os.path.exists(dataset_disk_path):
        data = load_from_disk(dataset_disk_path)
    else:
        ds = load_from_disk(dataset_path)
        if isinstance(ds, dict) and split in ds:
            data = ds[split]
        else:
            data = ds

    if debug:
        data = data.select(range(min(10, len(data))))
        logger.info(f"Debug mode: processing {len(data)} samples only")

    logger.info(f"Dataset size: {len(data)}")

    problems = []
    gold_answers = []
    subjects = []
    levels = []
    unique_ids = []

    for i in range(len(data)):
        sample = data[i]
        problems.append(sample['problem'])
        gold_answers.append(sample['answer'])
        subjects.append(sample.get('subject', ''))
        levels.append(sample.get('level', 0))
        unique_ids.append(sample.get('unique_id', f'math500_{i}'))

    logger.info(f"Total {len(problems)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # Build chat messages
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for problem in problems:
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": MATH_SYSTEM_PROMPT + "\n\n" + problem},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": MATH_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ])

    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    # Evaluate
    predictions = []
    correct = 0
    total = 0
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        gold_answer = gold_answers[i]
        problem = problems[i]

        stripped, had_thinking, was_truncated = strip_thinking_content(generated)
        pred_answer = extract_boxed_answer(stripped)
        is_correct = is_math_equivalent(pred_answer, gold_answer)

        if is_correct:
            correct += 1
        total += 1

        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = not pred_answer and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            'problem': problem[:300],
            'gold_answer': gold_answer,
            'generated': generated,
            'pred_answer': pred_answer,
            'is_correct': is_correct,
            'eval_mode': '0-shot chat (boxed)',
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults (0-shot chat):")
    logger.info(f"  Correct: {correct}/{total}")
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    return {
        'score': accuracy,
        'correct': correct,
        'total_samples': total,
        'eval_mode': '0-shot chat (boxed)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
    }


# ============================================================================
# MBPP Evaluator
# ============================================================================

MBPP_SYSTEM_PROMPT = (
    "You are a Python programming assistant. Write a Python function to solve "
    "the given task. Your code must pass the provided test cases. "
    "Only output the Python code, without any explanation."
)


def _run_mbpp_tests(
    code: str, test_list: List[str], test_setup_code: str = "", timeout: float = 10.0
) -> bool:
    """Execute MBPP test cases against generated code."""
    full_code = ""
    if test_setup_code:
        full_code += test_setup_code + "\n"
    full_code += code + "\n"
    for test in test_list:
        full_code += test + "\n"

    def _exec_code(code_str, result_queue):
        try:
            exec_globals = {}
            exec(code_str, exec_globals)
            result_queue.put(True)
        except Exception:
            result_queue.put(False)

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_exec_code, args=(full_code, result_queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return False

    if result_queue.empty():
        return False

    return result_queue.get()


def evaluate_mbpp(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate MBPP dataset using 0-shot code generation + execution."""
    logger.info(f"Loading MBPP dataset ({split} split)...")
    logger.info(f"Eval mode: 0-shot chat with test cases (code generation)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, dict):
        data = dataset.get(split, dataset[list(dataset.keys())[0]])
    else:
        data = dataset

    if debug:
        data = data.select(range(min(10, len(data))))
        logger.info(f"Debug mode: processing {len(data)} samples only")

    logger.info(f"Dataset size: {len(data)}")

    task_ids = []
    texts = []
    codes = []
    test_lists = []
    test_setup_codes = []

    for i in range(len(data)):
        sample = data[i]
        task_ids.append(sample['task_id'])
        texts.append(sample['text'])
        codes.append(sample['code'])
        test_lists.append(sample['test_list'])
        test_setup_codes.append(sample.get('test_setup_code', ''))

    logger.info(f"Total {len(texts)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # Build chat messages: 0-shot with test cases in prompt
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for idx, text in enumerate(texts):
        test_cases_str = "\n".join(test_lists[idx])
        user_content = (
            f"{text}\n\n"
            f"Your code should pass these tests:\n"
            f"{test_cases_str}"
        )
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": MBPP_SYSTEM_PROMPT + "\n\n" + user_content},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": MBPP_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ])

    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    # Evaluate each sample by executing test cases
    predictions = []
    correct = 0
    total = 0
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    logger.info("Running test case execution...")
    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        task_id = task_ids[i]
        text = texts[i]
        test_list = test_lists[i]
        test_setup_code = test_setup_codes[i]

        stripped, had_thinking, was_truncated = strip_thinking_content(generated)
        code = _extract_code_block(stripped)
        passed = _run_mbpp_tests(code, test_list, test_setup_code)

        if passed:
            correct += 1
        total += 1

        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = not code.strip() and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            'task_id': task_id,
            'text': text,
            'gold_code': codes[i],
            'generated': generated,
            'extracted_code': code,
            'test_list': test_list,
            'passed': passed,
            'eval_mode': '0-shot chat with test cases (code generation)',
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i + 1}/{len(generated_texts)}, pass rate so far: {correct}/{total}")

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults (0-shot chat with test cases):")
    logger.info(f"  Passed: {correct}/{total}")
    logger.info(f"  pass@1: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    return {
        'score': accuracy,
        'correct': correct,
        'total_samples': total,
        'eval_mode': '0-shot chat with test cases (code generation)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
    }


# ============================================================================
# LiveCodeBench-v6 Evaluator
# ============================================================================

LCB_SYSTEM_PROMPT = (
    "You are a Python programming assistant. Solve the given competitive "
    "programming problem. Read from standard input and write to standard output. "
    "Only output the Python code, without any explanation."
)


def _run_lcb_tests(
    code: str, test_cases: List[Dict[str, str]], timeout: float = 10.0
) -> Tuple[int, int]:
    """Execute LiveCodeBench test cases (stdin/stdout) against generated code.

    Batches all test cases into a single subprocess call for efficiency.
    Returns (num_passed, num_total).
    """
    total = len(test_cases)
    if total == 0:
        return 0, 0

    test_inputs = [tc.get("input", "") for tc in test_cases]
    test_outputs = [tc.get("output", "").strip() for tc in test_cases]

    test_data_json = json.dumps({"inputs": test_inputs, "outputs": test_outputs})
    per_test_timeout = int(timeout)

    wrapper_code = '''import sys, io, json, signal

def _timeout_handler(signum, frame):
    raise TimeoutError()

test_data = json.loads("""__TEST_DATA__""")
inputs = test_data["inputs"]
expected = test_data["outputs"]

user_code = """__USER_CODE__"""
compiled_code = compile(user_code, "<solution>", "exec")

passed = 0
for i in range(len(inputs)):
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(__PER_TEST_TIMEOUT__)
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        sys.stdin = io.StringIO(inputs[i])
        capture = io.StringIO()
        sys.stdout = capture
        exec_globals = {"__name__": "__main__"}
        exec(compiled_code, exec_globals)
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        signal.alarm(0)
        actual = capture.getvalue().strip()
        if actual == expected[i]:
            passed += 1
    except (TimeoutError, SystemExit):
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        signal.alarm(0)
    except Exception:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        signal.alarm(0)

print(passed, file=sys.stderr)
'''

    safe_test_data = test_data_json.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
    safe_user_code = code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
    wrapper_code = wrapper_code.replace('__TEST_DATA__', safe_test_data)
    wrapper_code = wrapper_code.replace('__USER_CODE__', safe_user_code)
    wrapper_code = wrapper_code.replace('__PER_TEST_TIMEOUT__', str(per_test_timeout))

    total_timeout = min(timeout * total, 120.0)

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper_code)
            f.flush()
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=total_timeout,
        )
        try:
            passed = int(result.stderr.strip().split('\n')[-1])
        except (ValueError, IndexError):
            passed = 0
    except subprocess.TimeoutExpired:
        passed = 0
    except Exception:
        passed = 0
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return passed, total


def _run_lcb_tests_for_problem(args: Tuple) -> Tuple[int, int, int, bool]:
    """Wrapper for parallel execution."""
    idx, code, test_cases = args
    if test_cases:
        num_passed, num_total = _run_lcb_tests(code, test_cases)
        all_passed = (num_passed == num_total) and num_total > 0
    else:
        num_passed, num_total = 0, 0
        all_passed = False
    return idx, num_passed, num_total, all_passed


def evaluate_lcb(
    client: SGLangClient,
    model_name: str,
    dataset_path: str,
    split: str = "test",
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_concurrent: int = 256,
    output_dir: str = "./results",
    debug: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate LiveCodeBench-v6 dataset using code generation + execution."""
    logger.info(f"Loading LiveCodeBench-v6 dataset...")
    logger.info(f"Eval mode: 0-shot chat (competitive programming)")
    logger.info(f"Params: temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

    # Load dataset - support both JSONL and HF dataset format
    data_file = os.path.join(dataset_path, "problems.jsonl")
    if os.path.exists(data_file):
        problems = []
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
    else:
        dataset = load_from_disk(dataset_path)
        if isinstance(dataset, dict):
            data = dataset.get(split, dataset[list(dataset.keys())[0]])
        else:
            data = dataset
        problems = [data[i] for i in range(len(data))]

    if debug:
        problems = problems[:min(5, len(problems))]
        logger.info(f"Debug mode: processing {len(problems)} samples only")

    logger.info(f"Dataset size: {len(problems)}")

    # Extract problem descriptions and test cases
    problem_ids = []
    descriptions = []
    test_cases_list = []

    for p in problems:
        pid = p.get('question_id', p.get('task_id', p.get('id', '')))
        problem_ids.append(str(pid))
        desc = p.get('question_content', p.get('description', p.get('prompt', '')))
        descriptions.append(desc)

        # Collect all test cases: public + private
        all_tc = []

        pub_tc = p.get('public_test_cases', p.get('test_cases', p.get('tests', [])))
        if isinstance(pub_tc, str):
            try:
                pub_tc = json.loads(pub_tc)
            except json.JSONDecodeError:
                pub_tc = []
        if isinstance(pub_tc, list):
            all_tc.extend(pub_tc)

        # Decode private test cases (base64 -> zlib -> pickle -> JSON)
        priv_raw = p.get('private_test_cases', '')
        if priv_raw and isinstance(priv_raw, str):
            try:
                import base64, zlib, pickle
                decoded = base64.b64decode(priv_raw)
                decompressed = zlib.decompress(decoded)
                tc_str = pickle.loads(decompressed)
                priv_tc = json.loads(tc_str) if isinstance(tc_str, str) else tc_str
                if isinstance(priv_tc, list):
                    all_tc.extend(priv_tc)
            except Exception as e:
                logger.warning(f"  Failed to decode private_test_cases for {pid}: {e}")

        test_cases_list.append(all_tc)

    logger.info(
        f"Test cases loaded: avg {sum(len(tc) for tc in test_cases_list) / max(len(test_cases_list), 1):.1f} per problem"
    )

    logger.info(f"Total {len(descriptions)} requests, starting concurrent inference...")
    client.max_concurrent = max_concurrent

    # Build chat messages
    no_system_prompt = kwargs.get('no_system_prompt', False)
    messages_list = []
    for desc in descriptions:
        if no_system_prompt:
            messages_list.append([
                {"role": "user", "content": LCB_SYSTEM_PROMPT + "\n\n" + desc},
            ])
        else:
            messages_list.append([
                {"role": "system", "content": LCB_SYSTEM_PROMPT},
                {"role": "user", "content": desc},
            ])

    generated_texts = client.chat_batch(
        messages_list=messages_list,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        show_progress=True,
    )

    # Parallel test execution
    thinking_flags = []
    codes = []
    for i in range(len(generated_texts)):
        stripped, had_thinking, was_truncated = strip_thinking_content(generated_texts[i])
        thinking_flags.append((had_thinking, was_truncated))
        codes.append(_extract_code_block(stripped))

    tasks = [(i, codes[i], test_cases_list[i]) for i in range(len(generated_texts))]
    num_workers = min(multiprocessing.cpu_count(), len(tasks), 64)
    logger.info(f"Running test case execution... (parallel, {num_workers} workers)")

    results_map = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_run_lcb_tests_for_problem, t): t[0] for t in tasks}
        done_count = 0
        try:
            for future in as_completed(futures, timeout=1200):
                try:
                    idx, num_passed, num_total, all_passed = future.result(timeout=180)
                except Exception as e:
                    idx = futures[future]
                    logger.warning(f"  Problem {idx} test execution failed: {e}")
                    num_passed, num_total, all_passed = 0, 0, False
                results_map[idx] = (num_passed, num_total, all_passed)
                done_count += 1
                if done_count % 20 == 0:
                    logger.info(f"  Test execution progress: {done_count}/{len(tasks)}")
        except TimeoutError:
            logger.warning(f"  Global timeout reached after 1200s, {len(futures) - done_count} futures still pending")
        # Handle any remaining futures that timed out or were not completed
        for future, idx in futures.items():
            if idx not in results_map:
                logger.warning(f"  Problem {idx} timed out, marking as failed")
                future.cancel()
                results_map[idx] = (0, 0, False)

    # Assemble predictions
    predictions = []
    correct = 0
    total = 0
    truncated_count = 0
    extraction_failed_count = 0
    empty_output_count = 0

    for i in range(len(generated_texts)):
        generated = generated_texts[i]
        pid = problem_ids[i]
        desc = descriptions[i]
        code = codes[i]
        had_thinking, was_truncated = thinking_flags[i]
        num_passed, num_total, all_passed = results_map[i]

        if all_passed:
            correct += 1
        total += 1

        is_empty_output = not generated or not generated.strip()
        is_extraction_failed = not code.strip() and not is_empty_output
        if was_truncated:
            truncated_count += 1
        if is_extraction_failed:
            extraction_failed_count += 1
        if is_empty_output:
            empty_output_count += 1

        pred_dict = {
            'problem_id': pid,
            'description': desc[:500],
            'generated': generated,
            'extracted_code': code,
            'num_tests_passed': num_passed,
            'num_tests_total': num_total,
            'all_passed': all_passed,
            'eval_mode': '0-shot chat (competitive programming)',
        }
        if was_truncated:
            pred_dict['truncated'] = True
        if is_extraction_failed:
            pred_dict['extraction_failed'] = True
        if is_empty_output:
            pred_dict['empty_output'] = True
        if had_thinking:
            pred_dict['had_thinking'] = True
        predictions.append(pred_dict)

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"\nResults (0-shot chat):")
    logger.info(f"  Passed: {correct}/{total}")
    logger.info(f"  pass@1: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    pred_file = os.path.join(output_dir, PREDICTIONS_FILE)
    save_predictions(predictions, pred_file)

    return {
        'score': accuracy,
        'correct': correct,
        'total_samples': total,
        'eval_mode': '0-shot chat (competitive programming)',
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
    }


# ============================================================================
# Evaluator registry
# ============================================================================

DATASET_EVALUATORS = {
    "gsm8k": evaluate_gsm8k,
    "math500": evaluate_math500,
    "mbpp": evaluate_mbpp,
    "live-code-bench-v6": evaluate_lcb,
}

# Dataset data file mapping
DATASET_DATA_FILES = {
    "gsm8k": "gsm8k",  # HF dataset directory (load_from_disk)
    "math500": "math500",  # HF dataset directory (load_from_disk)
    "mbpp": "mbpp",  # HF dataset directory
    "live-code-bench-v6": "live-code-bench-v6",  # directory with problems.jsonl
}


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EasyOPD Model Evaluation (SGLang DP inference engine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start SGLang server first (in another terminal or via eval_model_sglang.sh)
    python -m sglang.launch_server \\
        --model-path /path/to/model --dp-size 8 --tp-size 1 --port 30000

    # Math evaluation
    python evaluate_model_sglang.py --model_path /path/to/model --dataset gsm8k
    python evaluate_model_sglang.py --model_path /path/to/model --dataset math500

    # Code evaluation
    python evaluate_model_sglang.py --model_path /path/to/model --dataset mbpp
    python evaluate_model_sglang.py --model_path /path/to/model --dataset live-code-bench-v6
        """,
    )

    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model directory (used as model identifier)",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_EVALUATORS.keys()),
        help=f"Dataset name: {list(DATASET_EVALUATORS.keys())}",
    )
    parser.add_argument(
        "--base_url", type=str, default=DEFAULT_BASE_URL,
        help=f"SGLang server URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=256,
        help="Max concurrent requests (default: 256)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048,
        help="Max new tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Generation temperature (default: 0.6, aligned with KDFlow). "
             "Use 0.0 for greedy, but note that phi-4-mini is prone to "
             "repetition loops under greedy decoding on MATH500.",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Nucleus sampling top_p (default: 0.95, aligned with KDFlow).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help=f"Dataset root directory (default: {DATASETS_DIR})",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode, process only a few samples",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--no-system-prompt", action="store_true",
        help="Disable system prompt (merge into user message)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    model_name = os.path.basename(args.model_path.rstrip('/'))
    datasets_dir = args.data_dir if args.data_dir else DATASETS_DIR

    if args.output_dir:
        eval_output_dir = args.output_dir
    else:
        eval_output_dir = os.path.join(SCRIPT_DIR, "..", "results", model_name, args.dataset)
    os.makedirs(eval_output_dir, exist_ok=True)

    print("=" * 80)
    print("EasyOPD Model Evaluation (SGLang DP inference engine)")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Model name: {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Metric: {DATASET_METRICS.get(args.dataset, 'N/A')}")
    print(f"SGLang server: {args.base_url}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top_p: {args.top_p}")
    print(f"Output dir: {eval_output_dir}")
    print("=" * 80)

    # Step 1: Check if results already exist
    existing_pred = check_existing_results(eval_output_dir)
    if existing_pred:
        print(f"\n✓ Results already exist: {existing_pred}")
        print("  Skipping. Delete the file to re-run.")
        return

    # Step 2: Check dataset path
    data_file = DATASET_DATA_FILES[args.dataset]
    dataset_path = os.path.join(datasets_dir, data_file)
    if not os.path.exists(dataset_path):
        print(f"Error: dataset path not found: {dataset_path}")
        sys.exit(1)

    # Step 3: Check SGLang server
    print("\nChecking SGLang server...")
    client = SGLangClient(
        base_url=args.base_url,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
    )

    if not client.health_check():
        print(f"Error: SGLang server unavailable: {args.base_url}")
        print("Please start SGLang server first:")
        print(f"  python -m sglang.launch_server \\")
        print(f"      --model-path {args.model_path} \\")
        print(f"      --dp-size 8 --tp-size 1 --port 30000")
        sys.exit(1)

    served_model = client.get_model_name()
    print(f"✓ SGLang server ready, model: {served_model}")

    # Step 4: Run evaluation
    print(f"\nStarting {args.dataset} evaluation...")
    start_time = time.time()

    evaluator = DATASET_EVALUATORS[args.dataset]
    results = evaluator(
        client=client,
        model_name=served_model,
        dataset_path=dataset_path,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_concurrent=args.max_concurrent,
        output_dir=eval_output_dir,
        debug=args.debug,
        no_system_prompt=getattr(args, 'no_system_prompt', False),
    )

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Metric: {DATASET_METRICS[args.dataset]}")
    print(f"Score: {results['score']:.4f} ({results['score'] * 100:.2f}%)")
    print(f"Correct: {results['correct']}/{results['total_samples']}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"Throughput: {results['total_samples'] / elapsed:.1f} samples/s")

    # Diagnostics
    total_samples = results['total_samples']
    truncated_count = results.get('truncated_count', 0)
    extraction_failed_count = results.get('extraction_failed_count', 0)
    empty_output_count = results.get('empty_output_count', 0)

    if truncated_count or extraction_failed_count or empty_output_count:
        print(f"\nDiagnostics:")
        print(f"  Truncated: {truncated_count}/{total_samples}")
        print(f"  Extraction failed: {extraction_failed_count}/{total_samples}")
        print(f"  Empty output: {empty_output_count}/{total_samples}")

    if total_samples > 0 and empty_output_count / total_samples > 0.5:
        print(f"\n🚨 CRITICAL: Empty output ratio > 50%! Check chat template compatibility.")
        print(f"   Try: --no-system-prompt")

    print("=" * 80)

    # Save metrics
    result_file = os.path.join(eval_output_dir, METRICS_FILE)
    metrics_data = {
        'model': model_name,
        'model_path': args.model_path,
        'served_model': served_model,
        'dataset': args.dataset,
        'metric': DATASET_METRICS.get(args.dataset, 'N/A'),
        'score': results['score'],
        'correct': results['correct'],
        'total_samples': total_samples,
        'elapsed_seconds': elapsed,
        'throughput_sps': total_samples / elapsed if elapsed > 0 else 0.0,
        'eval_mode': results.get('eval_mode', 'default'),
        'truncated_count': truncated_count,
        'extraction_failed_count': extraction_failed_count,
        'empty_output_count': empty_output_count,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'base_url': args.base_url,
            'max_concurrent': args.max_concurrent,
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'split': args.split,
        },
    }
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
