# Copyright 2026 EasyOPD Contributors
#
# HuggingFace-based teacher engine for the `simple` cross-tokenizer KD method.
# This is a drop-in replacement for `sglang_engine.py` that uses HuggingFace
# transformers directly, allowing the simple method to work in environments
# where SGLang is not installed (e.g., vLLM-only setups).
#
# The engine loads the teacher model on a specific GPU and runs forward passes
# to extract hidden states at response positions.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class HFEngineConfig:
    """Configuration for the HuggingFace teacher engine."""

    model_path: str
    base_gpu_id: int = 0
    tp_size: int = 1
    mem_fraction_static: float = 0.6
    context_length: Optional[int] = None
    quantization: Optional[str] = None
    dtype: str = "bfloat16"


class HFTeacherEngine:
    """HuggingFace-based teacher engine that provides hidden states.

    This engine loads a teacher model using transformers and runs forward
    passes to extract hidden states. It provides the same functional
    interface as SGLangEngineService but without the SGLang dependency.
    """

    def __init__(self, config: HFEngineConfig) -> None:
        self.config = config
        self._started = False
        self.model = None
        self.tokenizer = None
        self.device = None

    def start(self) -> None:
        """Load the model and tokenizer onto the specified GPU."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device_str = f"cuda:{self.config.base_gpu_id}"
        self.device = torch.device(device_str)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        logger.info(
            "[HFTeacherEngine] Loading model %s on %s (dtype=%s)",
            self.config.model_path,
            device_str,
            self.config.dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch_dtype,
            device_map=device_str,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

        self._started = True
        logger.info(
            "[HFTeacherEngine] Model loaded successfully on %s", device_str
        )

    def generate(
        self,
        prompt: Optional[List[str]] = None,
        input_ids: Optional[List[List[int]]] = None,
        loss_masks: Optional[List[np.ndarray]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        return_hidden_states: bool = True,
    ) -> List[np.ndarray]:
        """Run forward pass and return hidden states at loss_mask positions.

        Args:
            prompt: List of prompt strings (used if input_ids is None).
            input_ids: Pre-tokenized input ids for each sample.
            loss_masks: Boolean masks indicating which positions to extract
                hidden states from.
            sampling_params: Ignored (kept for interface compatibility).
            return_hidden_states: Must be True for this engine.

        Returns:
            List of numpy arrays, each of shape [num_loss_tokens, hidden_dim].
        """
        if not self._started:
            raise RuntimeError("[HFTeacherEngine] Engine not started.")

        # Tokenize if input_ids not provided
        if input_ids is None:
            if prompt is None:
                raise ValueError("Either prompt or input_ids must be provided.")
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.context_length or 4096,
            )
            input_ids_tensor = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
        else:
            # Pad input_ids to same length
            max_len = max(len(ids) for ids in input_ids)
            pad_id = self.tokenizer.pad_token_id or 0
            padded_ids = []
            attention_masks = []
            for ids in input_ids:
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [pad_id] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)
            input_ids_tensor = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self.device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract last hidden state
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

        # Extract hidden states at loss_mask positions
        results: List[np.ndarray] = []
        for i in range(last_hidden.shape[0]):
            if loss_masks is not None and i < len(loss_masks):
                mask = np.asarray(loss_masks[i]).astype(bool)
                # Align mask length with actual sequence length
                seq_len = int(attention_mask[i].sum().item())
                if len(mask) > seq_len:
                    mask = mask[:seq_len]
                elif len(mask) < seq_len:
                    # Pad mask with False
                    mask = np.concatenate([mask, np.zeros(seq_len - len(mask), dtype=bool)])

                hidden_i = last_hidden[i, :seq_len][mask]
            else:
                # Return all non-padding hidden states
                seq_len = int(attention_mask[i].sum().item())
                hidden_i = last_hidden[i, :seq_len]

            results.append(hidden_i.cpu().to(torch.float16).numpy())

        return results

    def sleep(self, tags: Optional[str] = None) -> None:
        """Offload model to CPU to free GPU memory."""
        if self.model is not None:
            self.model.cpu()
            torch.cuda.empty_cache()
            logger.info("[HFTeacherEngine] Model offloaded to CPU.")

    def wakeup(self, tags: Optional[str] = None) -> None:
        """Move model back to GPU."""
        if self.model is not None:
            self.model.to(self.device)
            logger.info("[HFTeacherEngine] Model moved back to GPU.")

    def shutdown(self) -> None:
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        self._started = False
        logger.info("[HFTeacherEngine] Shutdown complete.")


__all__ = ["HFEngineConfig", "HFTeacherEngine"]
