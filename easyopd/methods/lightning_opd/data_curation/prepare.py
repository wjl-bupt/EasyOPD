"""Prepare Lightning-OPD parquet from student rollout data.

Phase 1 – tokenize (CPU-friendly):
    Reads student rollout parquet, builds prompt via chat template,
    tokenizes responses, truncates to ``--max-response-len``, writes
    intermediate parquet WITHOUT teacher logprobs.

Phase 2 – precompute teacher logprobs (requires teacher vLLM server):
    Reads the intermediate parquet from Phase 1, sends each
    (prompt + response) sequence to the teacher server, stores
    per-token response logprobs back into the parquet.

Usage (Phase 1, CPU):
    python3 -m easyopd.methods.lightning_opd.data_curation.prepare \\
        --tokenizer-path <sft_checkpoint> \\
        --input-parquet <rollouts.parquet> \\
        --output-dir <output_dir>

Usage (Phase 2, GPU with teacher vLLM running):
    python3 -m easyopd.methods.lightning_opd.data_curation.prepare \\
        --tokenizer-path <sft_checkpoint> \\
        --input-parquet <rollouts.parquet> \\
        --output-dir <output_dir> \\
        --compute-teacher-logprobs \\
        --teacher-url http://127.0.0.1:8000/v1/completions
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import pandas as pd

from .http_utils import post_json

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - exercised only in minimal envs
    AutoTokenizer = None


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Prepare Lightning-OPD parquet data (tokenize + optional teacher logprobs)."
    )
    parser.add_argument(
        "--tokenizer-path", type=str, required=True,
        help="Path to HuggingFace tokenizer (e.g. the student SFT checkpoint).",
    )
    parser.add_argument(
        "--input-parquet", type=str, required=True,
        help="Path to student rollout parquet. Expected columns: messages (list[dict]).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory where intermediate and final parquet files are written.",
    )
    parser.add_argument(
        "--max-response-len", type=int, default=4096,
        help="Maximum response token length (default: 4096).",
    )
    parser.add_argument(
        "--compute-teacher-logprobs", action="store_true",
        help="Run Phase 2: compute teacher logprobs via a running vLLM server.",
    )
    parser.add_argument(
        "--teacher-url", type=str, default="http://127.0.0.1:8000/v1/completions",
        help="Teacher vLLM completions endpoint URL.",
    )
    parser.add_argument(
        "--sft-teacher-id", type=str, default=None,
        help="SFT teacher identifier for consistency check.",
    )
    parser.add_argument(
        "--opd-teacher-id", type=str, default=None,
        help="OPD teacher identifier for consistency check.",
    )
    parser.add_argument(
        "--allow-teacher-mismatch", action="store_true",
        help="Allow SFT/OPD teacher mismatch (debug only).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=64,
        help="Number of concurrent requests to teacher server (default: 64).",
    )
    return parser.parse_args(args)


# ── Phase 1: tokenize ────────────────────────────────────────────────────────

def phase1_tokenize(args, intermediate_path: Path) -> None:
    """Tokenize student rollout data and write intermediate parquet."""
    if AutoTokenizer is None:
        raise ImportError("transformers is required for phase1_tokenize")
    intermediate_path = Path(intermediate_path)

    logger.info("[Phase 1] Loading tokenizer from %s", args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    logger.info("[Phase 1] Loading input parquet: %s", args.input_parquet)
    df = pd.read_parquet(args.input_parquet)
    logger.info("[Phase 1] Total rows: %d", len(df))

    rows_out = []
    truncated = 0
    skipped = 0

    for _, row in df.iterrows():
        messages = row["messages"] if "messages" in row.index else None
        if messages is None:
            skipped += 1
            continue

        user_messages = [m for m in messages if m.get("role") != "assistant"]
        prompt_str = tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )

        assistant_msg = None
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_msg = msg["content"]
                break
        if assistant_msg is None:
            skipped += 1
            continue

        response_ids = tokenizer.encode(assistant_msg, add_special_tokens=False)

        if len(response_ids) > args.max_response_len:
            truncated += 1
            response_ids = response_ids[: args.max_response_len]

        rows_out.append({
            "prompt": prompt_str,
            "label": "0",
            "response_tokens": response_ids,
            "response_length": len(response_ids),
            "metadata": {
                "sft_teacher_id": args.sft_teacher_id or "",
                "opd_teacher_id": args.opd_teacher_id or "",
            },
        })

    logger.info(
        "[Phase 1] Rows written: %d, truncated: %d, skipped: %d",
        len(rows_out), truncated, skipped,
    )
    df_out = pd.DataFrame(rows_out)
    intermediate_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(intermediate_path, index=False)
    logger.info("[Phase 1] Saved to %s", intermediate_path)


# ── Phase 2: precompute teacher logprobs ─────────────────────────────────────

async def _fetch_logprobs_vllm(
    teacher_url: str,
    prompt_ids: list[int],
    response_ids: list[int],
) -> list[float]:
    """Call teacher vLLM server and return per-token logprobs for the response."""
    full_text_ids = prompt_ids + response_ids
    payload = {
        "prompt": full_text_ids,
        "max_tokens": 0,
        "logprobs": True,
        "temperature": 0,
        "echo": True,
    }
    ret = await asyncio.to_thread(post_json, teacher_url, payload)

    # vLLM echo mode returns logprobs for all input tokens
    prompt_len = len(prompt_ids)
    resp_len = len(response_ids)

    # Extract logprobs for the response portion
    all_lps = ret.get("prompt_logprobs", [])
    if all_lps:
        # prompt_logprobs is a list of dicts per token
        response_lps = []
        for lp_dict in all_lps[prompt_len:]:
            if lp_dict:
                # Take the logprob of the token that was actually generated
                first_key = next(iter(lp_dict))
                response_lps.append(lp_dict[first_key] if isinstance(lp_dict[first_key], (int, float)) else lp_dict[first_key].get("logprob", 0.0))
            else:
                response_lps.append(0.0)
        return response_lps[:resp_len]

    # Fallback: try completion_logprobs
    comp_lps = ret.get("choices", [{}])[0].get("logprobs", {}).get("token_logprobs", [])
    if comp_lps:
        return comp_lps[-resp_len:]

    raise RuntimeError(f"Could not extract logprobs from vLLM response: {list(ret.keys())}")


async def _process_all(args, tokenizer, rows: list[dict]) -> list[list[float]]:
    """Process all rows concurrently, preserving order."""
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[list[float] | None] = [None] * len(rows)

    async def bounded_fetch(idx: int, prompt_ids: list[int], response_ids: list[int]):
        async with semaphore:
            result = await _fetch_logprobs_vllm(args.teacher_url, prompt_ids, response_ids)
        results[idx] = result

    tasks = []
    for idx, row in enumerate(rows):
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        response_ids = [int(x) for x in row["response_tokens"]]
        tasks.append(bounded_fetch(idx, prompt_ids, response_ids))
    await asyncio.gather(*tasks)

    return [r if r is not None else [] for r in results]


def phase2_logprobs(args, intermediate_path: Path, output_path: Path) -> None:
    """Compute teacher logprobs and write final parquet."""
    if AutoTokenizer is None:
        raise ImportError("transformers is required for phase2_logprobs")
    intermediate_path = Path(intermediate_path)
    output_path = Path(output_path)

    logger.info("[Phase 2] Loading intermediate parquet: %s", intermediate_path)
    df = pd.read_parquet(intermediate_path)
    rows = df.to_dict(orient="records")
    logger.info("[Phase 2] Total rows: %d", len(rows))

    logger.info("[Phase 2] Loading tokenizer from %s", args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    logger.info(
        "[Phase 2] Computing teacher logprobs via %s (concurrency=%d)",
        args.teacher_url, args.concurrency,
    )
    all_logprobs = asyncio.run(_process_all(args, tokenizer, rows))

    for row, lps in zip(rows, all_logprobs):
        row["teacher_log_probs"] = lps
        row["metadata"]["opd_teacher_id"] = args.opd_teacher_id or ""

    df_out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    logger.info("[Phase 2] Saved to %s", output_path)

    # Sanity check
    df_check = pd.read_parquet(output_path)
    row0 = df_check.iloc[0]
    logger.info("[Phase 2] Sanity check row 0:")
    logger.info("  prompt[:80]:            %s", str(row0["prompt"])[:80])
    logger.info("  len(response_tokens):   %d", len(row0["response_tokens"]))
    logger.info("  len(teacher_log_probs): %d", len(row0["teacher_log_probs"]))


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    parsed = parse_args(args)

    # Teacher consistency check
    if parsed.sft_teacher_id and parsed.opd_teacher_id:
        from easyopd.methods.lightning_opd.teacher_consistency import check_teacher_consistency

        check_teacher_consistency(
            parsed.sft_teacher_id,
            parsed.opd_teacher_id,
            allow_mismatch=parsed.allow_teacher_mismatch,
        )
        logger.info("Teacher consistency: OK")

    output_dir = Path(parsed.output_dir)
    input_stem = Path(parsed.input_parquet).stem
    intermediate_path = output_dir / f"{input_stem}-lightning_opd.parquet"
    output_path = output_dir / f"{input_stem}-lightning_opd-precomputed.parquet"

    if parsed.compute_teacher_logprobs:
        if not intermediate_path.exists():
            logger.info("Intermediate parquet not found, running Phase 1 first.")
            phase1_tokenize(parsed, intermediate_path)
        phase2_logprobs(parsed, intermediate_path, output_path)
    else:
        phase1_tokenize(parsed, intermediate_path)
        logger.info(
            "To add teacher logprobs, re-run with --compute-teacher-logprobs "
            "after starting the teacher vLLM server."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
