# EasyOPD `simple` Method — Cross-Tokenizer Knowledge Distillation

`simple` is the EasyOPD port of [KDFlow](../../../) `simple_ctkd`. It enables
on-policy knowledge distillation across two **different tokenizers** (e.g.
Qwen3 → Llama3.2) by computing KL on the *overlap sub-vocabulary* between
student and teacher tokenizers, on character-level aligned response
positions only.

## High-level pipeline

1. **Overlap discovery** (`alignment.find_overlap_tokens`)
   - Normalize vocab keys (`Ġ` → `▁`).
   - Take the intersection of the two vocabs.
   - Append EOS ids on both sides as a fallback.
2. **Sequence alignment** (`alignment.align_sequences`)
   - Greedy character-level cumulative match of teacher / student tokens.
   - Same-tokenizer fast path returns the identity index list.
   - EOS positions are always treated as a legal match.
3. **Teacher forward** (`teacher_forward.TeacherCrossTokenizerForward`)
   - Decode student response text with student tokenizer.
   - Re-encode with teacher tokenizer → `teacher_input_ids` /
     `teacher_response_mask`.
   - Run a plain HuggingFace forward and column-crop the response-segment
     logits to `teacher_overlap_token_ids` → `teacher_overlap_logits`.
   - **Note:** the teacher does NOT reuse the verl vllm/sglang path; it runs
     an independent HF forward. This keeps the engine-side code untouched.
4. **Loss** (`losses.compute_distillation_loss_simple_cross_tokenizer`)
   - For each sample, align teacher / student response tokens.
   - At aligned positions, gather student / teacher logits restricted to
     overlap columns (shape-checked equal).
   - Compute forward or reverse KL on this shared sub-vocab.
   - Return a `[B, resp_len]` tensor (zeros at unaligned positions) so the
     upstream `distillation_loss` helper handles both PG and supervised
     paths uniformly.

## Modifications to verl

All verl-side edits are wrapped in
`# [EasyOPD:simple] ... # [EasyOPD:simple] End` comment markers so they are
trivially auditable / revertible.

| File | Change |
| --- | --- |
| `verl/trainer/distillation/losses.py` | Add `use_cross_tokenizer` field to `DistillationLossSettings`; relax mutual-exclusivity to "exactly one of three"; tail-import that calls `register_simple_loss()`. |
| `verl/workers/config/distillation.py` | `validate_and_prepare_for_distillation` accepts `use_cross_tokenizer` and skips top-k validation + `response_length=1` rewrite; `DistillationConfig.__post_init__` forwards the flag. |

No other verl files are touched. With `loss_mode != "simple"` (the default),
behaviour of all existing distillation losses (`forward_kl_topk`, `kl`, `k1`,
`abs`, `mse`, `k2`, `low_var_kl`, `k3`) is bit-identical to upstream.

## How to run

See [`examples/simple/run_simple.sh`](../../../examples/simple/run_simple.sh)
and the [`easyopd/config/simple.yaml`](../../config/simple.yaml) configuration
template.

```bash
bash examples/simple/run_simple.sh
```

Set `+distillation.distillation_loss.loss_mode=simple` to opt in.

## Status

| Task | Status |
| --- | --- |
| 1. Skeleton & method metadata | ✅ |
| 2. Overlap & sequence alignment | ✅ |
| 3. Extend `DistillationLossSettings` | ✅ |
| 4. Extend `DistillationConfig` | ✅ |
| 5. Teacher HF forward | ✅ |
| 6. simple loss + register | ✅ |
| 7. Config & launch script | ✅ |
| 8. End-to-end integration tests | ✅ (CPU-runnable subset; full multi-GPU runs require live model checkpoints) |
