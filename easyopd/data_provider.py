# Copyright 2026 EasyOPD Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EasyOPD Automatic Data Provider.

This module handles automatic dataset downloading, conversion, and preparation.
Users only need to specify a HuggingFace dataset name (e.g., "openai/gsm8k") in
their config, and this module will:

1. Download the dataset from HuggingFace Hub (or ModelScope as fallback)
2. Convert it to the unified Parquet format expected by verl's RLHFDataset
3. Cache the converted data locally for reuse

Usage in config YAML::

    data:
      # Option 1: HuggingFace dataset name (auto-download + convert)
      dataset: "openai/gsm8k"
      dataset_split: "train"  # optional, default "train"

      # Option 2: Explicit local parquet files (legacy, still supported)
      train_files: ["path/to/train.parquet"]

      # Common options
      prompt_key: content
      max_prompt_length: 1024
      truncation: right

The framework will automatically resolve Option 1 into local parquet files
before passing to verl's data loading pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default cache directory
# ---------------------------------------------------------------------------
_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/easyopd/datasets")

# ---------------------------------------------------------------------------
# Predefined dataset recipes: method -> recommended datasets
# ---------------------------------------------------------------------------
DATASET_RECIPES: dict[str, dict[str, Any]] = {
    "echo_kd": {
        "dataset": "openai/gsm8k",
        "dataset_split": "train",
        "prompt_template": "math_qa",
        "description": "GSM8K for echo_kd demo distillation",
    },
    "gkd": {
        "dataset": "openai/gsm8k",
        "dataset_split": "train",
        "prompt_template": "math_qa",
        "description": "GSM8K math word problems for GKD distillation",
    },
    "sod": {
        "dataset": "jxu124/OpenAgentInstruct",
        "dataset_split": "train",
        "prompt_template": "agent",
        "description": "Agent instruction-following tasks for SOD",
    },
    "simple": {
        "dataset": "openai/gsm8k",
        "dataset_split": "train",
        "prompt_template": "math_qa",
        "description": "GSM8K for cross-tokenizer distillation",
    },
    "simct": {
        "dataset": "openai/gsm8k",
        "dataset_split": "train",
        "prompt_template": "math_qa",
        "description": "GSM8K for span-based cross-tokenizer KD",
    },
    "opcd": {
        "dataset": "openai/gsm8k",
        "dataset_split": "train",
        "prompt_template": "math_qa",
        "description": "GSM8K for on-policy context distillation",
    },
    "g_opd": {
        "dataset": "openai/gsm8k",
        "dataset_split": "train",
        "prompt_template": "math_qa",
        "description": "GSM8K for generalized on-policy distillation",
    },
    "gad": {
        "dataset": "openai/gsm8k",
        "dataset_split": "train",
        "prompt_template": "math_qa",
        "description": (
            "GAD (Generative Adversarial Distillation) repurposes the PPO "
            "critic as a Bradley-Terry discriminator that compares a student "
            "response against a teacher response. The dataset must therefore "
            "expose both a `prompt` field (student input) and a "
            "`teacher_response` field (teacher's reference output) for the "
            "BT comparison; on plain QA datasets such as GSM8K the answer "
            "column is used as the teacher response by the prompt template. "
            "Users with a paired student/teacher dataset on HuggingFace can "
            "override `data.dataset` accordingly."
        ),
    },
    "sdpo": {
        "dataset": "openai/gsm8k",
        "dataset_split": "train",
        "prompt_template": "math_qa",
        "description": "GSM8K for self-distilled policy optimization",
    },
    "opsa": {
        "dataset": "UWNSL/SafeChain",
        "dataset_split": "train",
        "prompt_template": "prompt_only",
        "description": "SafeChain dataset for OPSA safety self-distillation",
    },
    "ropd": {
        "dataset": "openai/gsm8k",
        "dataset_split": "train",
        "prompt_template": "math_qa",
        "description": (
            "Placeholder; replace with the rubric dataset used in ROPD "
            "experiments. ROPD is a black-box reward-manager method whose "
            "training signal is produced by a teacher + rubricator + "
            "verifier judge triple over each rollout, so it works on any "
            "instruction-style dataset that exposes a prompt field."
        ),
    },
    "lightning_opd": {
        "dataset": "open-thoughts/OpenThoughts3-1.2M",
        "dataset_split": "train",
        "prompt_template": "math_qa",
        "description": (
            "Lightning-OPD (offline on-policy distillation) consumes a "
            "parquet dataset that, in addition to a `prompt` field, "
            "carries a precomputed `teacher_log_probs` column produced "
            "by `examples/lightning_opd_trainer/tools/generate_sft_data.sh`. "
            "Per the paper (arXiv:2604.13010 §3), the SFT teacher and "
            "the OPD teacher MUST be the same model; the data-prep "
            "pipeline enforces this via "
            "`easyopd.methods.lightning_opd.teacher_consistency` and "
            "raises if `teacher_log_probs` is missing at training "
            "time. OpenThoughts3-1.2M is the default reasoning corpus "
            "used by the upstream NVIDIA-NeMo Lightning-OPD reference; "
            "users may override `data.dataset` to any HF dataset that "
            "carries a prompt and matching teacher log-probs column."
        ),
    },
    "vision_opd": {
        "dataset": "HuggingFaceM4/DocumentVQA",
        "dataset_split": "train",
        "prompt_template": "vision_qa",
        "description": "DocumentVQA for vision on-policy distillation",
    },
}

# ---------------------------------------------------------------------------
# Prompt templates: how to convert raw dataset fields into chat messages
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES: dict[str, dict[str, Any]] = {
    "math_qa": {
        "system_prompt": "Please reason step by step, and put your final answer within \\boxed{}.",
        "user_field": "question",
        "answer_field": "answer",
        "format": "chat",
    },
    "code": {
        "system_prompt": "You are a helpful coding assistant. Write clean, correct code.",
        "user_field": "instruction",
        "answer_field": "output",
        "format": "chat",
    },
    "agent": {
        "system_prompt": (
            "You are a helpful assistant that can use tools to solve tasks. "
            "Think step by step and use the available tools when needed."
        ),
        "user_field": "instruction",
        "answer_field": None,  # Agent tasks don't have pre-defined answers
        "format": "chat",
    },
    "vision_qa": {
        "system_prompt": "Answer the question based on the given image.",
        "user_field": "question",
        "answer_field": "answer",
        "image_field": "image",
        "format": "multimodal_chat",
    },
    "raw_chat": {
        # Dataset already has chat-format messages
        "messages_field": "messages",
        "format": "raw",
    },
    "prompt_only": {
        # Dataset has a single prompt field
        "prompt_field": "prompt",
        "format": "prompt",
    },
}


# ---------------------------------------------------------------------------
# Core: DataProvider class
# ---------------------------------------------------------------------------

class DataProvider:
    """Automatic dataset provider for EasyOPD.

    Handles downloading from HuggingFace/ModelScope, format conversion,
    and caching of datasets in verl-compatible Parquet format.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_data_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Resolve dataset references in config to local Parquet file paths.

        This is the main entry point. It takes a config dict and ensures
        that `data.train_files` and `data.val_files` point to valid local
        Parquet files, downloading and converting if necessary.

        Args:
            config: The full method config dict.

        Returns:
            Updated config with resolved `data.train_files` / `data.val_files`.
        """
        data_config = config.get("data", {})
        if not data_config:
            return config

        # If train_files already points to valid local files, skip
        train_files = data_config.get("train_files", [])
        if train_files and self._all_files_exist(train_files):
            logger.info("Data files already exist locally, skipping download.")
            return config

        # Check if a HuggingFace dataset is specified
        dataset_name = data_config.get("dataset")
        if not dataset_name:
            # Check if method has a default recipe
            method_name = config.get("method", {}).get("name")
            if method_name and method_name in DATASET_RECIPES:
                recipe = DATASET_RECIPES[method_name]
                dataset_name = recipe["dataset"]
                logger.info(
                    "No dataset specified, using default recipe for '%s': %s",
                    method_name, dataset_name
                )
            else:
                return config

        # Resolve the dataset
        train_split = data_config.get("dataset_split", "train")
        val_split = data_config.get("val_split", "test")
        prompt_template = data_config.get("prompt_template")
        subset = data_config.get("dataset_subset")
        max_samples = data_config.get("max_samples")

        # If no template specified, try to infer from method recipe
        if not prompt_template:
            method_name = config.get("method", {}).get("name")
            if method_name and method_name in DATASET_RECIPES:
                prompt_template = DATASET_RECIPES[method_name].get("prompt_template", "math_qa")
            else:
                prompt_template = "math_qa"

        # Download and convert
        train_path = self.prepare_dataset(
            dataset_name=dataset_name,
            split=train_split,
            subset=subset,
            prompt_template=prompt_template,
            max_samples=max_samples,
        )

        val_path = None
        try:
            val_path = self.prepare_dataset(
                dataset_name=dataset_name,
                split=val_split,
                subset=subset,
                prompt_template=prompt_template,
                max_samples=min(max_samples, 500) if max_samples else 500,
            )
        except Exception as e:
            logger.warning("Could not prepare validation split '%s': %s", val_split, e)

        # Update config
        config.setdefault("data", {})
        config["data"]["train_files"] = [str(train_path)]
        if val_path:
            config["data"]["val_files"] = [str(val_path)]
        else:
            config["data"].setdefault("val_files", [])

        # Set prompt_key if not already set
        config["data"].setdefault("prompt_key", "content")

        logger.info("Data resolved: train=%s, val=%s", train_path, val_path)
        return config

    def prepare_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
        prompt_template: str = "math_qa",
        max_samples: Optional[int] = None,
    ) -> Path:
        """Download and convert a HuggingFace dataset to Parquet format.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "openai/gsm8k").
            split: Dataset split to use (e.g., "train", "test").
            subset: Dataset subset/config name (e.g., "main" for gsm8k).
            prompt_template: Name of the prompt template to use for conversion.
            max_samples: Maximum number of samples to include (None = all).

        Returns:
            Path to the generated Parquet file.
        """
        # Check cache first
        cache_key = self._cache_key(dataset_name, split, subset, prompt_template, max_samples)
        cached_path = self.cache_dir / f"{cache_key}.parquet"

        if cached_path.exists():
            logger.info("Using cached dataset: %s", cached_path)
            return cached_path

        logger.info(
            "Preparing dataset: %s (split=%s, subset=%s, template=%s)",
            dataset_name, split, subset, prompt_template,
        )

        # Download from HuggingFace
        dataset = self._download_dataset(dataset_name, split, subset)

        # Limit samples if requested
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        # Convert to unified format
        converted = self._convert_dataset(dataset, prompt_template, dataset_name)

        # Save as Parquet
        converted.to_parquet(str(cached_path))
        logger.info("Dataset saved to: %s (%d samples)", cached_path, len(converted))

        return cached_path

    def _download_dataset(self, dataset_name: str, split: str, subset: Optional[str]):
        """Download dataset from HuggingFace Hub with multiple fallback strategies.

        Strategy order:
        1. Direct parquet download via requests (most reliable with proxies)
        2. datasets.load_dataset with HF_ENDPOINT
        3. datasets.load_dataset with hf-mirror.com
        4. ModelScope fallback
        """
        import datasets

        hf_error = None

        # Strategy 1: Direct parquet download via HuggingFace API + requests
        # This is the most reliable method when behind a proxy
        try:
            ds = self._download_via_api(dataset_name, split, subset)
            if ds is not None:
                return ds
        except Exception as e:
            logger.warning("Direct API download failed: %s", e)
            hf_error = e

        # Strategy 2: datasets.load_dataset (standard path)
        try:
            logger.info("Trying datasets.load_dataset for: %s", dataset_name)
            kwargs = {"path": dataset_name, "split": split, "trust_remote_code": True}
            if subset:
                kwargs["name"] = subset
            ds = datasets.load_dataset(**kwargs)
            return ds
        except Exception as e:
            if hf_error is None:
                hf_error = e
            logger.warning("datasets.load_dataset failed: %s", e)

        # Strategy 3: Try with explicit hf-mirror.com endpoint
        try:
            logger.info("Trying hf-mirror.com fallback for: %s", dataset_name)
            old_endpoint = os.environ.get("HF_ENDPOINT", "")
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            try:
                ds = datasets.load_dataset(
                    dataset_name, name=subset, split=split, trust_remote_code=True
                )
                return ds
            finally:
                if old_endpoint:
                    os.environ["HF_ENDPOINT"] = old_endpoint
                else:
                    os.environ.pop("HF_ENDPOINT", None)
        except Exception as mirror_error:
            logger.warning("hf-mirror fallback failed: %s", mirror_error)

        # Strategy 4: ModelScope
        try:
            logger.info("Trying ModelScope fallback for: %s", dataset_name)
            from modelscope.msdatasets import MsDataset
            ds = MsDataset.load(dataset_name, split=split, subset_name=subset)
            return ds.to_hf_dataset() if hasattr(ds, 'to_hf_dataset') else ds
        except ImportError:
            logger.debug("ModelScope not installed, skipping fallback.")
        except Exception as ms_error:
            logger.warning("ModelScope download failed: %s", ms_error)

        raise RuntimeError(
            f"Failed to download dataset '{dataset_name}'. "
            f"Please check your network connection or set HF_ENDPOINT environment variable. "
            f"Original error: {hf_error}"
        )

    def _download_via_api(self, dataset_name: str, split: str, subset: Optional[str]):
        """Download dataset parquet files directly via HTTP requests.

        This method queries the HuggingFace API to find parquet file URLs,
        then downloads them directly using requests (which respects HTTP_PROXY).
        """
        import tempfile

        import datasets
        import requests

        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

        # Query dataset info to find parquet files
        api_url = f"{hf_endpoint}/api/datasets/{dataset_name}"
        logger.info("Querying dataset info: %s", api_url)

        resp = requests.get(api_url, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"API request failed: {resp.status_code}")

        info = resp.json()
        siblings = info.get("siblings", [])

        # Determine the config/subset name
        config_name = subset
        if not config_name:
            # Try to infer from dataset info
            card_data = info.get("cardData", {})
            configs = card_data.get("configs", [])
            if configs:
                config_name = configs[0].get("config_name", "default")
            else:
                # Check if there's a 'main' or 'default' config
                for s in siblings:
                    fname = s.get("rfilename", "")
                    if fname.startswith("main/"):
                        config_name = "main"
                        break
                    elif fname.startswith("default/"):
                        config_name = "default"
                        break

        # Find parquet files for the requested split
        prefix = f"{config_name}/{split}" if config_name else split
        parquet_files = [
            s["rfilename"] for s in siblings
            if s["rfilename"].startswith(prefix) and s["rfilename"].endswith(".parquet")
        ]

        if not parquet_files:
            # Try without config prefix
            parquet_files = [
                s["rfilename"] for s in siblings
                if split in s["rfilename"] and s["rfilename"].endswith(".parquet")
            ]

        if not parquet_files:
            raise RuntimeError(
                f"No parquet files found for split '{split}' in dataset '{dataset_name}'. "
                f"Available files: {[s['rfilename'] for s in siblings if s['rfilename'].endswith('.parquet')]}"
            )

        logger.info("Found %d parquet file(s) for split '%s'", len(parquet_files), split)

        # Download parquet files
        local_files = []
        download_dir = self.cache_dir / "downloads" / dataset_name.replace("/", "_")
        download_dir.mkdir(parents=True, exist_ok=True)

        for pf in parquet_files:
            local_path = download_dir / os.path.basename(pf)
            if not local_path.exists():
                download_url = f"{hf_endpoint}/datasets/{dataset_name}/resolve/main/{pf}"
                logger.info("Downloading: %s", download_url)
                resp = requests.get(download_url, timeout=120, stream=True)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info("Downloaded: %s (%d bytes)", local_path, local_path.stat().st_size)
            else:
                logger.info("Using cached: %s", local_path)
            local_files.append(str(local_path))

        # Load as HuggingFace Dataset
        ds = datasets.load_dataset("parquet", data_files=local_files, split="train")
        return ds

    def _convert_dataset(self, dataset, prompt_template: str, dataset_name: str):
        """Convert a HuggingFace dataset to the unified EasyOPD format.

        The output format is a Parquet file with a 'content' column containing
        chat messages in the format: [{"role": "user", "content": "..."}]
        """
        import datasets as hf_datasets

        template = PROMPT_TEMPLATES.get(prompt_template)
        if template is None:
            raise ValueError(
                f"Unknown prompt template '{prompt_template}'. "
                f"Available: {list(PROMPT_TEMPLATES.keys())}"
            )

        fmt = template.get("format", "chat")

        if fmt == "raw":
            # Dataset already has chat messages
            messages_field = template["messages_field"]
            if messages_field in dataset.column_names:
                # Just rename to 'content' if needed
                if messages_field != "content":
                    dataset = dataset.rename_column(messages_field, "content")
                return dataset
            else:
                raise ValueError(
                    f"Dataset does not have expected field '{messages_field}'. "
                    f"Available columns: {dataset.column_names}"
                )

        elif fmt == "prompt":
            # Single prompt field
            prompt_field = template["prompt_field"]
            return dataset.map(
                lambda x: {"content": [{"role": "user", "content": x[prompt_field]}]},
                remove_columns=dataset.column_names,
            )

        elif fmt == "chat":
            # Convert question/instruction + optional system prompt to chat format
            system_prompt = template.get("system_prompt", "")
            user_field = template["user_field"]
            answer_field = template.get("answer_field")

            # Auto-detect field names if the expected ones don't exist
            user_field = self._resolve_field(dataset, user_field, ["question", "instruction", "input", "prompt", "problem"])
            if answer_field:
                answer_field = self._resolve_field(dataset, answer_field, ["answer", "output", "response", "solution", "target"], required=False)

            def convert_row(row):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": str(row[user_field])})
                result = {"content": messages}

                # Optionally keep the ground truth answer as metadata
                if answer_field and answer_field in row:
                    result["ground_truth"] = str(row[answer_field])

                return result

            # Determine columns to keep
            keep_cols = ["content", "ground_truth"] if answer_field else ["content"]
            converted = dataset.map(convert_row, remove_columns=dataset.column_names)

            # Filter out ground_truth column if it wasn't populated
            if "ground_truth" in converted.column_names:
                # Keep it as metadata for evaluation
                pass

            return converted

        elif fmt == "multimodal_chat":
            # Vision datasets with images
            system_prompt = template.get("system_prompt", "")
            user_field = template["user_field"]
            image_field = template.get("image_field", "image")

            user_field = self._resolve_field(dataset, user_field, ["question", "query", "instruction"])

            def convert_vision_row(row):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": str(row[user_field])})
                result = {"content": messages}

                # Keep image data
                if image_field in row and row[image_field] is not None:
                    result["images"] = [row[image_field]]

                return result

            return dataset.map(convert_vision_row, remove_columns=dataset.column_names)

        else:
            raise ValueError(f"Unknown format '{fmt}' in template '{prompt_template}'")

    def _resolve_field(
        self,
        dataset,
        preferred: str,
        alternatives: list[str],
        required: bool = True,
    ) -> Optional[str]:
        """Resolve a field name, trying alternatives if preferred doesn't exist."""
        if preferred in dataset.column_names:
            return preferred

        for alt in alternatives:
            if alt in dataset.column_names:
                logger.info("Field '%s' not found, using '%s' instead", preferred, alt)
                return alt

        if required:
            raise ValueError(
                f"Cannot find field '{preferred}' or alternatives {alternatives} "
                f"in dataset columns: {dataset.column_names}"
            )
        return None

    def _all_files_exist(self, files: list[str]) -> bool:
        """Check if all files in the list exist locally."""
        for f in files:
            if isinstance(f, str) and f.startswith("<"):
                # Placeholder path like "<path to training data>"
                return False
            if not os.path.exists(f):
                return False
        return True

    def _cache_key(
        self,
        dataset_name: str,
        split: str,
        subset: Optional[str],
        prompt_template: str,
        max_samples: Optional[int],
    ) -> str:
        """Generate a deterministic cache key for a dataset configuration."""
        key_parts = {
            "dataset": dataset_name,
            "split": split,
            "subset": subset,
            "template": prompt_template,
            "max_samples": max_samples,
        }
        key_str = json.dumps(key_parts, sort_keys=True)
        hash_str = hashlib.md5(key_str.encode()).hexdigest()[:12]
        # Human-readable prefix
        safe_name = dataset_name.replace("/", "_").replace(".", "_")
        return f"{safe_name}_{split}_{hash_str}"

    @classmethod
    def get_recipe(cls, method_name: str) -> Optional[dict[str, Any]]:
        """Get the recommended dataset recipe for a method."""
        return DATASET_RECIPES.get(method_name)

    @classmethod
    def list_recipes(cls) -> dict[str, str]:
        """List all available dataset recipes with descriptions."""
        return {
            name: recipe["description"]
            for name, recipe in DATASET_RECIPES.items()
        }


# ---------------------------------------------------------------------------
# Integration: hook into EasyOPD's config resolution
# ---------------------------------------------------------------------------

def resolve_data_in_config(config: dict[str, Any], cache_dir: Optional[str] = None) -> dict[str, Any]:
    """Convenience function to resolve data references in a config dict.

    This is the function that should be called during framework startup
    to ensure all data paths are resolved before training begins.

    Args:
        config: Full method configuration dict.
        cache_dir: Optional override for the dataset cache directory.

    Returns:
        Updated config with resolved data paths.
    """
    provider = DataProvider(cache_dir=cache_dir)
    return provider.resolve_data_config(config)
