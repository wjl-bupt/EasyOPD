# Copyright 2026 EasyOPD Contributors
#
# Monkey patch for SGLang scheduler's `process_batch_result_prefill` method.
# Ported verbatim from KDFlow (`kdflow/backend/sglang/monkey_patch.py`) to
# remove the cross-repo dependency. Verified against SGLang 0.5.9.
#
# Key change relative to upstream SGLang: use `.numpy()` instead of
# `.tolist()` for hidden_states tensor transfer, which is dramatically
# faster (often 10-100x) for the large logits/hidden tensors produced by
# teacher prefill.

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )

logger = logging.getLogger(__name__)

# Flag to prevent multiple patch applications.
_PATCH_APPLIED = False


def process_batch_result_prefill_patched(
    self: "Scheduler",
    batch: "ScheduleBatch",
    result: Union["GenerationBatchResult", "EmbeddingBatchResult"],
):
    """Patched version of `process_batch_result_prefill`.

    Key change: use `.numpy()` instead of `.tolist()` for hidden_states
    (much faster for large tensors).
    """
    from sglang.srt.environ import envs
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.mem_cache.common import release_kv_cache

    skip_stream_req = None

    if self.is_generation:
        if result.copy_done is not None:
            result.copy_done.synchronize()

        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
        )

        # Move next_token_ids and logprobs to cpu
        next_token_ids = next_token_ids.tolist()
        if batch.return_logprob:
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.tolist()
                )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )

        hidden_state_offset = 0

        # Check finish conditions
        logprob_pt = 0

        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req.finished() or req.is_retracted:
                # decode req in mixed batch or retracted req
                continue

            if req.is_chunked <= 0:
                if hasattr(req.time_stats, "prefill_finished_ts"):
                    if req.time_stats.prefill_finished_ts == 0.0:
                        req.time_stats.prefill_finished_ts = time.time()
                else:
                    req.time_stats.set_prefill_finished_time()

                # req output_ids are set here
                req.output_ids.append(next_token_id)
                req.check_finished()

                if req.finished():
                    self.maybe_collect_routed_experts(req)
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.completion_time = time.perf_counter()
                elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                    # This updates radix so others can match
                    self.tree_cache.cache_unfinished_req(req)

                self.maybe_collect_customized_info(i, req, logits_output)

                if batch.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]

                    num_input_logprobs = self._calculate_num_input_logprobs(
                        req, extend_input_len, extend_logprob_start_len
                    )

                    if req.return_logprob:
                        self.add_logprob_return_values(
                            i,
                            req,
                            logprob_pt,
                            next_token_ids,
                            num_input_logprobs,
                            logits_output,
                        )
                    logprob_pt += num_input_logprobs

                # === KEY CHANGE: Use .numpy() instead of .tolist() ===
                if (
                    req.return_hidden_states
                    and logits_output.hidden_states is not None
                ):
                    req.hidden_states.append(
                        logits_output.hidden_states[
                            hidden_state_offset : (
                                hidden_state_offset := hidden_state_offset
                                + len(req.origin_input_ids)
                            )
                        ]
                        .half()
                        .cpu()
                        .numpy()
                    )

                if req.grammar is not None:
                    # FIXME: this try-except block is for handling unexpected xgrammar issue.
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        # Grammar accept_token can raise ValueError if the token is not in the grammar.
                        # This can happen if the grammar is not set correctly or the token is invalid.
                        logger.error(
                            f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        )
                        self.abort_request(AbortReq(rid=req.rid))
                    req.grammar.finished = req.finished()

            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1
                # There is only at most one request being currently chunked.
                # Because this request does not finish prefill,
                # we don't want to stream the request currently being chunked.
                skip_stream_req = req

                # Incrementally update input logprobs.
                if batch.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        # Update input logprobs.
                        num_input_logprobs = self._calculate_num_input_logprobs(
                            req, extend_input_len, extend_logprob_start_len
                        )
                        if req.return_logprob:
                            self.add_input_logprob_return_values(
                                i,
                                req,
                                logits_output,
                                logprob_pt,
                                num_input_logprobs,
                                last_prefill_chunk=False,
                            )
                        logprob_pt += num_input_logprobs

    else:  # embedding or reward model
        if result.copy_done is not None:
            result.copy_done.synchronize()

        is_sparse = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()

        embeddings = result.embeddings

        if is_sparse:
            batch_ids, token_ids = embeddings.indices()
            values = embeddings.values()

            embeddings = [{} for _ in range(embeddings.size(0))]
            for i in range(batch_ids.shape[0]):
                embeddings[batch_ids[i].item()][token_ids[i].item()] = values[
                    i
                ].item()
        else:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.tolist()
            else:
                embeddings = [tensor.tolist() for tensor in embeddings]

        # Check finish conditions
        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue

            req.embedding = embeddings[i]
            if req.is_chunked <= 0:
                # Dummy output token for embedding models
                req.output_ids.append(0)
                req.check_finished()

                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                else:
                    self.tree_cache.cache_unfinished_req(req)
            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1

    self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

    if self.current_scheduler_metrics_enabled:
        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        self.log_prefill_stats(
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )


def apply_patch() -> bool:
    """Apply the monkey patch to SGLang's `SchedulerOutputProcessorMixin`.

    This function is idempotent — calling it multiple times is safe.
    Returns True if the patch was applied (or already applied), False
    otherwise.
    """
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return True

    try:
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            SchedulerOutputProcessorMixin,
        )

        # Check if already patched (e.g. by another mechanism such as
        # sitecustomize, or a previous EasyOPD subprocess).
        current_method = getattr(
            SchedulerOutputProcessorMixin, "process_batch_result_prefill", None
        )
        if current_method is not None and getattr(
            current_method, "_easyopd_patched", False
        ):
            _PATCH_APPLIED = True
            print(
                f"[easyopd.simple.sglang_monkey_patch] Already applied, "
                f"PID={os.getpid()}",
                flush=True,
            )
            return True

        # Mark the patched function so we (or others) can detect it later.
        process_batch_result_prefill_patched._easyopd_patched = True

        # Apply patch.
        SchedulerOutputProcessorMixin.process_batch_result_prefill = (
            process_batch_result_prefill_patched
        )

        _PATCH_APPLIED = True
        print(
            f"[easyopd.simple.sglang_monkey_patch] SUCCESS: "
            f"process_batch_result_prefill patched, PID={os.getpid()}",
            flush=True,
        )
        return True

    except ImportError as e:
        print(
            f"[easyopd.simple.sglang_monkey_patch] Cannot import SGLang: {e}",
            flush=True,
        )
        return False
    except Exception as e:
        print(
            f"[easyopd.simple.sglang_monkey_patch] Error applying patch: {e}",
            flush=True,
        )
        import traceback

        traceback.print_exc()
        return False


def is_patch_applied() -> bool:
    """Return whether the patch has been applied in this process."""
    return _PATCH_APPLIED
