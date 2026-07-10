# Copyright 2026 EasyOPD Contributors
#
# SGLang teacher engine subprocess service. Ported from KDFlow
# (`kdflow/backend/sglang/sglang_engine.py`) so EasyOPD does not need to
# import KDFlow at runtime.
#
# The service spawns SGLang's PatchedEngine in a separate process, exposes
# generate / sleep / wakeup / update_weights_from_tensor over an
# mp.Queue, and streams hidden_states back via a ZMQ PUSH/PULL IPC socket
# (zero-copy numpy transfer).
#
# Hidden-state transfer is fast because the in-process scheduler patch
# (`sglang_monkey_patch.apply_patch`) replaces SGLang's default
# `.tolist()` with `.numpy()`.

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import queue
from dataclasses import dataclass
from multiprocessing import Queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import zmq
from sglang.srt.entrypoints.engine import Engine as _SglEngine
from sglang.srt.managers.scheduler import (
    run_scheduler_process as _original_run_scheduler_process,
)


os.environ["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "true"


def _patched_run_scheduler_process(*args, **kwargs):
    try:
        from easyopd.methods.simple.sglang_monkey_patch import apply_patch

        apply_patch()
    except Exception as e:
        print(
            f"[PatchedEngine] WARNING: Failed to apply monkey patch "
            f"(PID={os.getpid()}): {e}",
            flush=True,
        )
    return _original_run_scheduler_process(*args, **kwargs)


class PatchedEngine(_SglEngine):
    """SGLang Engine that applies the EasyOPD monkey patch in scheduler subprocesses.

    Motivation: SGLang Engine supports returning hidden states, but the
    upstream implementation calls `.tolist()` to move hidden_states from
    GPU tensor to a Python list, which is very slow. The patch replaces
    that with `.numpy()` for an order-of-magnitude speedup.
    """

    run_scheduler_process_func = staticmethod(_patched_run_scheduler_process)


@dataclass
class EngineConfig:
    """Configuration for the SGLang Engine."""

    model_path: str
    tp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    chunked_prefill_size: int = -1
    disable_radix_cache: bool = True
    enable_memory_saver: bool = False
    mem_fraction_static: float = 0.8
    context_length: Optional[int] = None
    quantization: Optional[str] = None
    offload_tags: Optional[str] = "all"
    base_gpu_id: int = 0
    # for multi-node tp/pp
    nnodes: int = 1
    node_rank: int = 0
    dist_init_addr: Optional[str] = None


def _engine_worker(
    config: EngineConfig, request_queue: Queue, response_queue: Queue
) -> None:
    import asyncio

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    if config.nnodes > 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    # Ensure an asyncio event loop exists in this spawned subprocess.
    # SGLang 0.4.6.post3 uses asyncio.get_event_loop() internally for
    # release_memory_occupation / resume_memory_occupation / generate etc.
    # uvloop (if installed) raises RuntimeError if no loop is set.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    engine = None
    zmq_ctx = None

    try:
        zmq_ctx = zmq.Context()
        data_socket = zmq_ctx.socket(zmq.PUSH)
        zmq_ipc_addr = f"ipc:///tmp/sglang_hs_{os.getpid()}"
        data_socket.bind(zmq_ipc_addr)

        engine_kwargs = dict(
            model_path=config.model_path,
            tp_size=config.tp_size,
            ep_size=config.ep_size,
            chunked_prefill_size=config.chunked_prefill_size,
            disable_radix_cache=config.disable_radix_cache,
            quantization=config.quantization,
            mem_fraction_static=config.mem_fraction_static,
            base_gpu_id=config.base_gpu_id,
            nnodes=config.nnodes,
            node_rank=config.node_rank,
            dist_init_addr=config.dist_init_addr,
        )
        if config.context_length is not None:
            engine_kwargs["context_length"] = config.context_length
            # Allow overriding context length to prevent SGLang from rejecting
            os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        engine = PatchedEngine(**engine_kwargs)

        response_queue.put(
            {
                "type": "init_done",
                "success": True,
                "zmq_ipc_addr": zmq_ipc_addr,
            }
        )

        while True:
            request = request_queue.get()
            if request is None:
                break

            req_type = request.get("type")

            try:
                if req_type == "generate":
                    _handle_generate(
                        engine, request, data_socket, request_queue, response_queue
                    )
                elif req_type == "sleep":
                    _handle_sleep(engine, request, config, response_queue)
                elif req_type == "wakeup":
                    _handle_wakeup(engine, request, config, response_queue)
                elif req_type == "update_weights_from_tensor":
                    _handle_update_weights_from_tensor(engine, request, response_queue)
                else:
                    response_queue.put(
                        {
                            "type": req_type,
                            "success": False,
                            "error": f"Unknown request type: {req_type}",
                        }
                    )
            except Exception:
                import traceback

                response_queue.put(
                    {
                        "type": req_type,
                        "success": False,
                        "error": traceback.format_exc(),
                    }
                )

    except Exception:
        import traceback

        response_queue.put(
            {
                "type": "init_done",
                "success": False,
                "error": traceback.format_exc(),
            }
        )
    finally:
        if zmq_ctx:
            try:
                data_socket.close()
                zmq_ctx.term()
            except Exception:
                pass
        if engine:
            try:
                engine.shutdown()
            except Exception:
                pass


def _normalize_tags(tags):
    """Convert tags to the format SGLang expects (None, or list of strings)."""
    if tags is None or tags == "all":
        return None
    if isinstance(tags, str):
        return [tags]
    return tags


def _handle_generate(engine, request, data_socket, request_queue, response_queue):
    """Run inference and send hidden states via ZMQ."""
    kwargs = request["kwargs"]

    generate_kwargs = {
        "sampling_params": kwargs["sampling_params"],
        "return_hidden_states": kwargs.get("return_hidden_states", True),
    }
    if kwargs.get("input_ids") is not None:
        generate_kwargs["prompt"] = None
        generate_kwargs["input_ids"] = kwargs["input_ids"]
    else:
        generate_kwargs["prompt"] = kwargs["prompt"]
    if kwargs.get("image_data") is not None:
        generate_kwargs["image_data"] = kwargs["image_data"]

    outputs = engine.generate(**generate_kwargs)

    num_samples = len(outputs)

    response_queue.put(
        {
            "type": "generate",
            "success": True,
            "num_samples": num_samples,
        }
    )

    for i, (output, mask) in enumerate(zip(outputs, kwargs["loss_masks"])):
        hs_np = output["meta_info"]["hidden_states"][0]
        hs_np = np.asarray(hs_np)
        mask = np.asarray(mask).astype(bool)
        hs_np = hs_np[: mask.shape[0]]  # loss_mask may have been truncated
        hs_np = hs_np[mask]
        if not hs_np.flags["C_CONTIGUOUS"]:
            hs_np = np.ascontiguousarray(hs_np)

        meta = pickle.dumps({"shape": hs_np.shape, "dtype": str(hs_np.dtype)})
        data_socket.send(meta, flags=zmq.SNDMORE)
        data_socket.send(hs_np, copy=False)


def _handle_sleep(engine, request, config, response_queue):
    """Offload GPU memory."""
    import asyncio
    tags = request.get("tags", config.offload_tags)
    torch.cuda.empty_cache()
    # SGLang 0.4.6.post3 uses asyncio.get_event_loop() internally;
    # in a spawned subprocess there may be no current loop.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    engine.release_memory_occupation()
    response_queue.put({"type": "sleep", "success": True, "tags": tags})


def _handle_wakeup(engine, request, config, response_queue):
    """Restore GPU memory."""
    import asyncio
    tags = request.get("tags", config.offload_tags)
    torch.cuda.empty_cache()
    # SGLang 0.4.6.post3 uses asyncio.get_event_loop() internally;
    # in a spawned subprocess there may be no current loop.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    engine.resume_memory_occupation()
    response_queue.put({"type": "wakeup", "success": True, "tags": tags})


def _handle_update_weights_from_tensor(engine, request, response_queue):
    """Update weights from student (for self-distillation)."""
    serialized_named_tensors = request["kwargs"]["serialized_named_tensors"]
    load_format = request["kwargs"]["load_format"]
    flush_cache = request["kwargs"]["flush_cache"]
    engine.update_weights_from_tensor(
        named_tensors=serialized_named_tensors,
        load_format=load_format,
        flush_cache=flush_cache,
    )
    response_queue.put({"type": "update_weights_from_tensor", "success": True})


class SGLangEngineService:
    """Manages an SGLang Engine in a subprocess with ZMQ communication."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.process: Optional[mp.Process] = None
        self.request_queue: Optional[Queue] = None
        self.response_queue: Optional[Queue] = None
        self._started = False
        self._zmq_ctx: Optional[zmq.Context] = None
        self._data_socket = None

    def start(self, timeout: float = 1800.0) -> None:
        """Start the SGLang Engine in a subprocess."""
        if self._started:
            raise RuntimeError("Service already started")

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        self.request_queue = mp.Queue()
        self.response_queue = mp.Queue()

        self.process = mp.Process(
            target=_engine_worker,
            args=(self.config, self.request_queue, self.response_queue),
        )
        self.process.start()

        try:
            response = self.response_queue.get(timeout=timeout)
            if response.get("type") == "init_done" and response.get("success"):
                self._started = True
                self._zmq_ctx = zmq.Context()
                self._data_socket = self._zmq_ctx.socket(zmq.PULL)
                self._data_socket.connect(response["zmq_ipc_addr"])
            else:
                raise RuntimeError(f"Init failed: {response.get('error')}")
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Engine initialization failed: {e}")

    def generate(
        self,
        prompt: Optional[List[str]],
        loss_masks: List[np.ndarray],
        sampling_params: Dict[str, Any],
        return_hidden_states: bool = True,
        image_data=None,
        input_ids: Optional[List[List[int]]] = None,
    ) -> List[np.ndarray]:
        """Run generation and return hidden states via ZMQ.

        Args:
            prompt: List of raw text prompts. SGLang handles tokenization
                internally. Set to None when using pre-tokenized input_ids.
            input_ids: Optional list of pre-tokenized teacher input ids. When
                provided, these ids are sent directly to SGLang and prompt text
                is ignored.
            loss_masks: Pre-computed boolean masks for selecting response
                hidden states.
            sampling_params: Sampling parameters (e.g. ``max_new_tokens=0``
                for prefill-only).
            return_hidden_states: Whether to return hidden states.
            image_data: Optional list of image data for multimodal models.
        """
        if not self._started:
            raise RuntimeError("Service not started")

        # Check if subprocess is still alive before sending request
        if self.process and not self.process.is_alive():
            raise RuntimeError(
                f"[SGLangEngineService] Engine subprocess "
                f"(PID={self.process.pid}) is dead! "
                f"exitcode={self.process.exitcode}"
            )

        kwargs = {
            "prompt": prompt,
            "input_ids": input_ids,
            "loss_masks": loss_masks,
            "sampling_params": sampling_params,
            "return_hidden_states": return_hidden_states,
        }
        if image_data is not None:
            kwargs["image_data"] = image_data

        self.request_queue.put({"type": "generate", "kwargs": kwargs})

        response = self._get_response(req_type="generate", timeout=600)
        if not response.get("success"):
            raise RuntimeError(f"Generate failed: {response.get('error')}")

        # Read hidden states via ZMQ.
        num_samples = response["num_samples"]
        hidden_states: List[np.ndarray] = []
        for _ in range(num_samples):
            if self._data_socket.poll(timeout=120_000) == 0:
                raise RuntimeError("ZMQ recv timeout while receiving hidden states")
            meta_bytes = self._data_socket.recv()
            data_bytes = self._data_socket.recv()
            meta = pickle.loads(meta_bytes)
            hs = np.frombuffer(data_bytes, dtype=np.dtype(meta["dtype"])).reshape(
                meta["shape"]
            )
            hidden_states.append(hs.copy())  # copy because zmq buffer will be reused

        return hidden_states

    def sleep(self, tags: Optional[str] = "all"):
        """Release GPU memory."""
        if not self._started:
            return None
        self.request_queue.put({"type": "sleep", "tags": tags})
        response = self._get_response(req_type="sleep", timeout=300)
        if not response.get("success"):
            raise RuntimeError(f"Sleep failed: {response.get('error')}")
        return response.get("tags")

    def wakeup(self, tags: Optional[str] = "all"):
        """Resume GPU memory."""
        if not self._started:
            return None
        self.request_queue.put({"type": "wakeup", "tags": tags})
        response = self._get_response(req_type="wakeup", timeout=300)
        if not response.get("success"):
            raise RuntimeError(f"Wakeup failed: {response.get('error')}")
        return response.get("tags")

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        kwargs = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        self.request_queue.put(
            {"type": "update_weights_from_tensor", "kwargs": kwargs}
        )
        response = self._get_response(
            req_type="update_weights_from_tensor", timeout=300
        )
        if not response.get("success"):
            raise RuntimeError(
                f"update_weights_from_tensor failed: {response.get('error')}"
            )

    def _get_response(self, req_type: str = "unknown", timeout: int = 600,
                      check_interval: int = 10):
        elapsed = 0
        while elapsed < timeout:
            try:
                return self.response_queue.get(timeout=check_interval)
            except queue.Empty:
                elapsed += check_interval
                if self.process and not self.process.is_alive():
                    raise RuntimeError(
                        f"Engine subprocess (PID={self.process.pid}) died "
                        f"during '{req_type}'! "
                        f"exitcode={self.process.exitcode}"
                    )
        raise RuntimeError(
            f"Response timeout after {timeout}s during '{req_type}'"
        )

    def shutdown(self) -> None:
        """Shutdown the subprocess gracefully."""
        if not self._started:
            return
        self._started = False
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up subprocess, queues and shared memory."""
        if self._data_socket:
            try:
                self._data_socket.close()
            except Exception:
                pass
            self._data_socket = None
        if self._zmq_ctx:
            try:
                self._zmq_ctx.term()
            except Exception:
                pass
            self._zmq_ctx = None

        if self.request_queue:
            try:
                self.request_queue.put(None)
            except Exception:
                pass

        if self.process:
            self.process.join(timeout=30)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()

        self.process = None
        self.request_queue = None
        self.response_queue = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
