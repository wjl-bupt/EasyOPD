# GAD: Generative Adversarial Distillation

## Method summary

GAD repurposes the verl PPO critic as a Bradley-Terry discriminator over
student vs teacher responses. The discriminator's last-token score is
consumed by the standard PPO advantage / actor path as a token-level
reward. See `docs/algo/gad.md` for the full method description, and
`docs/superpowers/specs/2026-05-27-gad-migration-to-easyopd-design.md`
for the integration design.

Paper: https://arxiv.org/abs/2511.10643

## verl files modified

| File | What changes | Why |
|------|--------------|-----|
| `verl/workers/critic/dp_critic.py` | `_forward_micro_batch` accepts `compute_teacher` kwarg, swaps input keys to the four teacher_* keys when set, reduces output to last-token-only | Critic becomes a discriminator; the only relevant scalar is the seq-level score at the last token |
| `verl/workers/critic/dp_critic.py` | `update_critic` dispatches to `easyopd.methods.gad.critic_update.update_critic_step` when `gad.enable=true` | Replaces MSE value loss with Bradley-Terry pairwise loss |

All edits are wrapped in `# ============ [EasyOPD:GAD] ... # ============ [EasyOPD:GAD] End ============` comments.

## Data contract

Training samples must provide four extra tensor batch keys:

- `teacher_input_ids`: full critic input for the teacher (prompt + teacher response), shape `[B, T_full_t]`
- `teacher_attention_mask`: attention mask matching `teacher_input_ids`, shape `[B, T_full_t]`
- `teacher_position_ids`: position ids matching `teacher_input_ids`, shape `[B, T_full_t]`
- `teacher_response`: just the teacher response tokens (used to derive `response_length`), shape `[B, T_resp_t]`

All four shapes are independent of the student's `T_s`. The trainer's
existing `_get_gen_batch` only pops `input_ids / attention_mask /
position_ids`, so these extra tensor keys survive automatically into
the `batch` consumed by `compute_values` and `update_critic`.

## Reproduction outline

1. Prepare a parquet dataset whose rows include the four teacher-side
   tensor fields (`teacher_input_ids`, `teacher_attention_mask`,
   `teacher_position_ids`, `teacher_response`). The upstream paper used GPT-5-Chat responses on LMSYS prompts
   (see `https://huggingface.co/datasets/ytz20/LMSYS-Chat-GPT-5-Chat-Response`
   and `microsoft/LMOps/gad/tools/export_lmsys_parquet.py` for an example).
2. Provide a pretrained discriminator checkpoint and pass its path via
   `gad.discriminator_init_path=<path>`. Use a discriminator that has
   already been warmed up (e.g. SFT on teacher responses).
3. Launch with `bash examples/gad_trainer/train_gad.sh ...`.
