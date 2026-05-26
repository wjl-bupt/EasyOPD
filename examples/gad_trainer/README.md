# GAD trainer entrypoint

This directory holds the shell launcher for the GAD adversarial-training
stage. It composes the base config at `easyopd/config/gad/base.yaml`
with user-supplied overrides.

## Data preparation

Each training sample must provide four teacher-side tensor batch keys:

| Key | Shape | Meaning |
|-----|-------|---------|
| `teacher_input_ids` | `[B, T_full_t]` | Full critic input (prompt + teacher response) |
| `teacher_attention_mask` | `[B, T_full_t]` | Attention mask for the full input |
| `teacher_position_ids` | `[B, T_full_t]` | Position ids for the full input |
| `teacher_response` | `[B, T_resp_t]` | Teacher response tokens only (used to derive response_length) |

These are tensor batch entries and survive verl's `_get_gen_batch` pop
list because that list only contains the three student-side keys.

To reproduce the paper:

1. Download the LMSYS-Chat-GPT-5-Chat-Response dataset from
   https://huggingface.co/datasets/ytz20/LMSYS-Chat-GPT-5-Chat-Response.
2. Convert to parquet using the upstream tool at
   `microsoft/LMOps/gad/tools/export_lmsys_parquet.py`.
3. Point `data.train_files` at the resulting parquet path.

## Discriminator checkpoint

The discriminator is a separately trained model (typically the result
of SFT on teacher responses, i.e. the upstream "warmup" stage). EasyOPD
does NOT provide warmup tooling — bring your own checkpoint.

Set `gad.discriminator_init_path=<path>` and `critic.model.path=${gad.discriminator_init_path}`.

## Launching

```bash
bash examples/gad_trainer/train_gad.sh \
  gad.discriminator_init_path=/data/disc.ckpt \
  data.train_files=/data/lmsys_gpt5_chat.parquet \
  actor_rollout_ref.model.path=/data/qwen2.5-7b-instruct \
  critic.model.path=/data/disc.ckpt
```
