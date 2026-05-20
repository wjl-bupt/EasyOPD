# EasyOPD `simple` — Cross-Tokenizer KD Example

This example runs the EasyOPD `simple` cross-tokenizer knowledge-distillation
method on top of verl's on-policy distillation framework.

## What is `simple`?

`simple` is the EasyOPD port of [KDFlow](../../) `simple_ctkd`. It enables
KD across two **different tokenizers** (e.g. Qwen3 → Llama3.2) by:
1. Discovering the overlap sub-vocabulary between student / teacher tokenizers.
2. Greedy-character-aligning the response token sequences.
3. Computing KL on the overlap-cropped logits at aligned positions only.

See [`easyopd/methods/simple/README.md`](../../easyopd/methods/simple/README.md)
for the algorithm-side documentation.

## Quick start

```bash
bash examples/simple/run_simple.sh
```

Default setup:
- Student: `Qwen/Qwen3-8B`
- Teacher: `meta-llama/Llama-3.2-3B-Instruct`
- Loss: `simple` (forward KL on overlap sub-vocab)
- 1 node × 8 GPUs (override via env vars)

## Tunable env vars

| Variable | Default | Description |
| --- | --- | --- |
| `STUDENT_MODEL` | `Qwen/Qwen3-8B` | HF model id or local path |
| `TEACHER_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | teacher path (any tokenizer) |
| `KL_DIRECTION` | `forward` | `forward` (KL(s‖t)) or `reverse` |
| `USE_POLICY_GRADIENT` | `False` | `True` enables thinkingmachines-style PG path |
| `TEACHER_WORLD_SIZE` | `2` | GPUs reserved for the teacher pool |
| `MAX_PROMPT_LENGTH` | `1024` | |
| `MAX_RESPONSE_LENGTH` | `1024` | |

Pass any extra hydra overrides at the end:
```bash
bash examples/simple/run_simple.sh \
    +data.shuffle=True \
    trainer.total_epochs=10
```

## Same-tokenizer sanity check

For a quick dry-run without an actual cross-tokenizer setup, point the
student and teacher at the same model. Alignment falls into the identity
fast path so `simple` ≈ full-vocab forward KL on the response:
```bash
TEACHER_MODEL="$STUDENT_MODEL" bash examples/simple/run_simple.sh
```

## Expected outputs

In addition to verl's standard PPO / distillation metrics, `simple` reports:
- `distillation/loss` — aggregated KD loss
- `distillation/align_ratio` — fraction of student response tokens that aligned
  to a teacher token. Sustained values < 0.5 trigger a console WARNING.
- `distillation/overlap_vocab_size` — size of the overlap sub-vocabulary (constant)
- `distillation/loss_min` / `distillation/loss_max` — min/max KD per token

## Files involved

```
easyopd/methods/simple/{__init__,alignment,losses,teacher_forward}.py
easyopd/methods/simple/README.md
easyopd/config/simple.yaml
examples/simple/run_simple.sh
examples/simple/README.md            ← you are here
```

## Verl-side touchpoints

All changes are bracketed by `# [EasyOPD:simple]` markers and limited to:
- `verl/trainer/distillation/losses.py`
- `verl/workers/config/distillation.py`

When `loss_mode != "simple"`, behavior is bit-identical to upstream verl.
