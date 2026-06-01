"""Generate response parquet files for Lightning-OPD pipeline steps."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .http_utils import post_json

logger = logging.getLogger(__name__)

PostJson = Callable[[str, dict[str, Any], float], dict[str, Any]]


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Generate assistant responses from prompt JSONL/parquet and write a messages parquet."
    )
    parser.add_argument("--input-prompts", required=True, help="Input JSONL or parquet prompts.")
    parser.add_argument("--output-parquet", required=True, help="Output parquet with a messages column.")
    parser.add_argument("--model", required=True, help="Model identifier served by the endpoint.")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="OpenAI-compatible chat/completions endpoint.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum generated response tokens.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top-p.")
    parser.add_argument("--concurrency", type=int, default=16, help="Number of concurrent HTTP requests.")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds.")
    return parser.parse_args(args)


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list):
        messages = []
        for item in value:
            if isinstance(item, dict) and "role" in item and "content" in item:
                messages.append({"role": str(item["role"]), "content": str(item["content"])})
        if messages:
            return messages
    raise ValueError(f"Unsupported prompt format: {type(value)!r}")


def _load_prompts(path: str) -> list[list[dict[str, str]]]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        prompts = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompts.append(_normalize_messages(item.get("prompt", item.get("messages"))))
        return prompts

    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
        key = "prompt" if "prompt" in df.columns else "messages"
        return [_normalize_messages(value) for value in df[key].tolist()]

    raise ValueError(f"Unsupported input format for {path}; expected .jsonl or .parquet")


def _extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"Response did not contain choices: {response.keys()}")

    choice = choices[0]
    message = choice.get("message")
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"])
    if choice.get("text") is not None:
        return str(choice["text"])
    raise RuntimeError(f"Response choice did not contain text content: {choice.keys()}")


def _generate_one(
    messages: list[dict[str, str]],
    args,
    post_json_fn: PostJson,
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    response = post_json_fn(args.endpoint, payload, args.timeout)
    content = _extract_text(response)
    return {
        "messages": [*messages, {"role": "assistant", "content": content}],
        "prompt": messages,
        "response": content,
        "metadata": {"model": args.model, "endpoint": args.endpoint},
    }


def generate_responses(args, post_json_fn: PostJson = post_json) -> None:
    prompts = _load_prompts(args.input_prompts)
    logger.info("Loaded %d prompts from %s", len(prompts), args.input_prompts)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        rows = list(executor.map(lambda messages: _generate_one(messages, args, post_json_fn), prompts))

    output_path = Path(args.output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    logger.info("Wrote %d responses to %s", len(rows), output_path)


def main(args=None) -> None:
    parsed = parse_args(args)
    generate_responses(parsed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
