# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs


def convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: str | list[str], tokenizer, config=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None
        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

    def __len__(self):
        return len(self.messages)

    def _process_message_tokens(
        self,
        messages: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        enable_thinking: Optional[bool] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process tokens for a single message or a group of messages.

        Args:
            messages: List of message dictionaries
            start_idx: Start index in messages list
            end_idx: End index in messages list
            is_assistant: Whether this is an assistant message
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (tokens, loss_mask, attention_mask)
        """
        if start_idx > 0:
            prev_applied_text = self.tokenizer.apply_chat_template(
                messages[:start_idx],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
                **self.apply_chat_template_kwargs,
            )
            if is_assistant:
                prev_applied_text_w_generation_prompt = self.tokenizer.apply_chat_template(
                    messages[:start_idx],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    tools=tools,
                    **self.apply_chat_template_kwargs,
                )

        else:
            prev_applied_text = ""

        cur_applied_text = self.tokenizer.apply_chat_template(
            messages[:end_idx],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            tools=tools,
            **self.apply_chat_template_kwargs,
        )
        # Get tokens for the current message only
        if is_assistant:
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text) :]
            generation_prompt_tokens = self.tokenizer.encode(
                generation_prompt_text,
                add_special_tokens=False,
            )
            _message_tokens = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text_w_generation_prompt) :],
                add_special_tokens=False,
            )
            message_tokens = generation_prompt_tokens + _message_tokens
            loss_mask = [0] * (len(generation_prompt_tokens)) + [1] * (
                len(message_tokens) - len(generation_prompt_tokens)
            )
        else:
            message_tokens = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text) :],
                add_special_tokens=False,
            )
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)

        return message_tokens, loss_mask, attention_mask

    def _validate_and_convert_tokens(
        self,
        full_tokens: Any,
        concat_tokens: list[int],
        concat_loss_mask: list[int],
        concat_attention_mask: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate tokenization and convert to tensors.

        Args:
            full_tokens: Full conversation tokens
            concat_tokens: Concatenated tokens
            concat_loss_mask: Concatenated loss mask
            concat_attention_mask: Concatenated attention mask

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask) as tensors
        """
        full_tokens_tensor = None

        # NOTE: tokenizer.apply_chat_template(...) may return different token container
        # types across tokenizer versions:
        # - torch.Tensor (when return_tensors="pt")
        # - tokenizers.Encoding / List[tokenizers.Encoding]
        # - List[int]
        if isinstance(full_tokens, torch.Tensor):
            if full_tokens.dim() > 1:
                full_tokens = full_tokens[0]
            full_tokens_tensor = full_tokens.to(dtype=torch.long)
            full_tokens_list = full_tokens_tensor.tolist()
        elif hasattr(full_tokens, "ids"):
            # tokenizers.Encoding
            full_tokens_list = list(full_tokens.ids)
            full_tokens_tensor = torch.tensor(full_tokens_list, dtype=torch.long)
        elif hasattr(full_tokens, "input_ids"):
            # transformers.BatchEncoding
            ids = full_tokens["input_ids"]
            if isinstance(ids, torch.Tensor):
                if ids.dim() > 1:
                    ids = ids[0]
                full_tokens_list = ids.tolist()
            elif isinstance(ids, np.ndarray):
                if ids.ndim > 1:
                    ids = ids[0]
                full_tokens_list = ids.tolist()
            elif isinstance(ids, (list, tuple)):
                if len(ids) > 0 and isinstance(ids[0], (list, tuple, np.ndarray, torch.Tensor)):
                    first = ids[0]
                    if isinstance(first, torch.Tensor):
                        full_tokens_list = first.tolist()
                    elif isinstance(first, np.ndarray):
                        full_tokens_list = first.tolist()
                    else:
                        full_tokens_list = list(first)
                else:
                    full_tokens_list = list(ids)
            else:
                raise TypeError(f"Unsupported BatchEncoding input_ids type: {type(ids)}")
            full_tokens_tensor = torch.tensor(full_tokens_list, dtype=torch.long)
        elif isinstance(full_tokens, np.ndarray):
            if full_tokens.ndim > 1:
                full_tokens = full_tokens[0]
            full_tokens_list = full_tokens.tolist()
            full_tokens_tensor = torch.tensor(full_tokens_list, dtype=torch.long)
        elif isinstance(full_tokens, (list, tuple)):
            if len(full_tokens) > 0 and hasattr(full_tokens[0], "ids"):
                # List[tokenizers.Encoding]
                full_tokens_list = list(full_tokens[0].ids)
            elif len(full_tokens) > 0 and isinstance(full_tokens[0], (list, tuple, np.ndarray, torch.Tensor)):
                # Batched token list, take first sample
                first = full_tokens[0]
                if isinstance(first, torch.Tensor):
                    full_tokens_list = first.tolist()
                elif isinstance(first, np.ndarray):
                    full_tokens_list = first.tolist()
                else:
                    full_tokens_list = list(first)
            else:
                full_tokens_list = list(full_tokens)
            full_tokens_tensor = torch.tensor(full_tokens_list, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported token type from apply_chat_template: {type(full_tokens)}")

        if len(concat_tokens) != len(full_tokens_list) or not all(
            a == b for a, b in zip(concat_tokens, full_tokens_list, strict=True)
        ):
            if len(concat_tokens) == len(full_tokens_list):
                # Same length but different tokens: use full_tokens (correct sequence)
                # with concat_loss_mask (correct loss boundaries).
                # This handles cases like Phi-4-mini where apply_chat_template with
                # add_generation_prompt=False appends <|endoftext|> but the full
                # conversation has <|assistant|> at that position instead.
                logging.warning(
                    f"Token mismatch detected (same length {len(full_tokens_list)}). "
                    f"Using full_tokens with concatenated loss_mask."
                )
                return (
                    full_tokens_tensor,
                    torch.tensor(concat_loss_mask, dtype=torch.long),
                    torch.tensor(concat_attention_mask, dtype=torch.long),
                )
            else:
                # Different lengths: fall back to concat_tokens (original behavior)
                logging.warning(
                    f"Token mismatch detected! Full tokenization length: {len(full_tokens_list)}, "
                    f"Concatenated tokens length: {len(concat_tokens)}. Using concatenated version."
                )
                return (
                    torch.tensor(concat_tokens, dtype=torch.long),
                    torch.tensor(concat_loss_mask, dtype=torch.long),
                    torch.tensor(concat_attention_mask, dtype=torch.long),
                )

        return (
            full_tokens_tensor,
            torch.tensor(concat_loss_mask, dtype=torch.long),
            torch.tensor(concat_attention_mask, dtype=torch.long),
        )

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None

        # Build tokens via per-message tokenization directly
        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []

        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                # Process assistant message
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, is_assistant=True, enable_thinking=enable_thinking, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            elif cur_messages["role"] == "tool":
                # Process consecutive tool messages
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, st, ed, enable_thinking=enable_thinking, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i = ed
            elif cur_messages["role"] in ["user", "system"]:
                # Process user or system message
                if cur_messages["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, enable_thinking=enable_thinking, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

        # Validate against full tokenization to catch template-induced mismatches.
        # For Phi-4-mini with transformers 4.57.1:
        #   - Per-message tokenization with add_generation_prompt=False produces
        #     <|endoftext|> where the full conversation has <|end|><|assistant|>
        #   - This causes concat_tokens to be missing <|assistant|> and have wrong tokens
        #   - Solution: use full_tokens (correct sequence) and rebuild loss_mask
        #     based on assistant message boundaries
        try:
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=enable_thinking, tools=tools,
                **self.apply_chat_template_kwargs,
            )
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            if full_tokens != concat_tokens:
                # Rebuild loss_mask from scratch using full_tokens
                # Strategy: find <|assistant|> tokens and mark everything after them
                # (until the next <|end|><|endoftext|> or <|user|>/<|system|>) as loss=1
                assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
                end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
                endoftext_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
                user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
                system_token_id = tokenizer.convert_tokens_to_ids("<|system|>")

                new_loss_mask = [0] * len(full_tokens)
                in_assistant = False
                for ti in range(len(full_tokens)):
                    if full_tokens[ti] == assistant_token_id:
                        in_assistant = True
                        continue  # <|assistant|> itself has loss=0
                    if in_assistant:
                        if full_tokens[ti] in (user_token_id, system_token_id):
                            in_assistant = False
                        else:
                            new_loss_mask[ti] = 1
                            # Stop at <|endoftext|> (but include it in loss)
                            if full_tokens[ti] == endoftext_token_id:
                                in_assistant = False

                concat_tokens = full_tokens
                concat_loss_mask = new_loss_mask
                concat_attention_mask = [1] * len(full_tokens)
                if item < 3:
                    logging.info(
                        f"[Sample {item}] Token mismatch fixed: using full_tokens "
                        f"(len={len(full_tokens)}) with rebuilt loss_mask "
                        f"(loss_tokens={sum(new_loss_mask)})"
                    )
        except Exception as e:
            if item < 3:
                logging.warning(f"[Sample {item}] Full tokenization failed: {e}")
            pass  # If full tokenization fails, use concat_tokens as-is

        # === FIX: Loss mask off-by-one correction for ALL assistant messages ===
        # In fsdp_sft_trainer.py, loss is computed as:
        #   loss_mask_shifted = loss_mask[:, :-1]
        #   loss[i] = CE(logits[i], input_ids[i+1]) * loss_mask_shifted[i]
        # So loss_mask[i]=1 means model learns to predict input_ids[i+1].
        #
        # For Phi-4-mini, the token sequence is:
        #   ... <|end|> <|assistant|> The answer is 4. <|end|> <|endoftext|>
        #   loss:  0       0          1   1     1  1    1       1
        #
        # After trainer's [:, :-1] shift, loss_mask[<|assistant|> pos]=0 means
        # the model does NOT learn to predict "The" (first response token).
        #
        # Fix: For each 0->1 transition in loss_mask, set the position before
        # the first loss=1 to also be 1. This makes the model learn to predict
        # the first response token given the <|assistant|> context.
        for j in range(1, len(concat_loss_mask)):
            if concat_loss_mask[j] == 1 and concat_loss_mask[j - 1] == 0:
                concat_loss_mask[j - 1] = 1
        # === END FIX ===

        # === DEBUG LOGGING: Print loss_mask details for first few samples ===
        # This helps verify:
        # 1. Token sequence is correct (has <|assistant|>, not <|endoftext|> in middle)
        # 2. Loss mask boundaries are correct
        # 3. After the off-by-one fix, first response token IS learned
        if item < 3:
            non_pad_len = len(concat_tokens)
            loss_start_idx = next((j for j, m in enumerate(concat_loss_mask) if m == 1), -1)
            loss_end_idx = non_pad_len - 1 - next((j for j, m in enumerate(reversed(concat_loss_mask)) if m == 1), -1) if 1 in concat_loss_mask else -1
            total_loss_tokens = sum(concat_loss_mask)
            logging.info(
                f"[Sample {item}] Loss mask debug: "
                f"total_tokens={non_pad_len}, loss_tokens={total_loss_tokens}, "
                f"loss_range=[{loss_start_idx}:{loss_end_idx+1}]"
            )
            # Show tokens around the loss boundary (where loss transitions from 0 to 1)
            if loss_start_idx >= 0:
                boundary_start = max(0, loss_start_idx - 1)
                boundary_end = min(non_pad_len, loss_start_idx + 4)
                boundary_info = []
                for bi in range(boundary_start, boundary_end):
                    tok_id = concat_tokens[bi]
                    tok_text = tokenizer.decode([tok_id])
                    mask_val = concat_loss_mask[bi]
                    boundary_info.append(f"  idx={bi}: id={tok_id} loss={mask_val} text={repr(tok_text)}")
                logging.info(
                    f"[Sample {item}] Loss boundary tokens:\n" + "\n".join(boundary_info)
                )
                # After fix: loss_mask[loss_start_idx]=1 means model predicts input_ids[loss_start_idx+1]
                # loss_start_idx should now be at <|assistant|> position
                first_predicted = concat_tokens[loss_start_idx + 1] if loss_start_idx + 1 < non_pad_len else None
                first_predicted_text = tokenizer.decode([first_predicted]) if first_predicted is not None else "N/A"
                logging.info(
                    f"[Sample {item}] First token model learns to predict: "
                    f"input_ids[{loss_start_idx+1}] = {repr(first_predicted_text)} "
                    f"(given context up to input_ids[{loss_start_idx}] = {repr(tokenizer.decode([concat_tokens[loss_start_idx]]))})"
                )
        # === END DEBUG LOGGING ===

        # Convert concatenated tokens to tensors directly
        input_ids = torch.tensor(concat_tokens, dtype=torch.long)
        loss_mask = torch.tensor(concat_loss_mask, dtype=torch.long)
        attention_mask = torch.tensor(concat_attention_mask, dtype=torch.long)

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
