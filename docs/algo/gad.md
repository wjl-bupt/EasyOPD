# GAD: Generative Adversarial Distillation

**Paper:** [arXiv:2511.10643](https://arxiv.org/abs/2511.10643)
**Source code:** `easyopd/methods/gad/`
**Launch:** `examples/gad_trainer/train_gad.sh`

## TL;DR

GAD repurposes verl's PPO critic as a Bradley-Terry discriminator over
student vs teacher responses. The discriminator emits a sequence-level
score (placed at the last response token) which the standard PPO
advantage path consumes as a token-level reward. The actor update is
unchanged.

## How it sits inside verl

| Component | Verl behavior | GAD override |
|-----------|---------------|--------------|
| Rollout | Generates student responses | Unchanged |
| Critic forward (`_forward_micro_batch`) | Per-token value head | Same forward; reduced to last-token-only when `gad.enable=true`; can also forward teacher inputs via `compute_teacher=True` |
| `compute_values` | Returns per-token values | Returns last-token-only score (via the forward override) |
| Advantage computation | GAE/GRPO from token-level rewards | Unchanged; token_level_scores = critic values |
| `update_critic` | MSE value regression | Replaced with Bradley-Terry pairwise loss against teacher responses |
| Actor update | Standard PPO | Unchanged |

All overrides live in `easyopd/methods/gad/`. The only verl file with
EasyOPD-side edits is `verl/workers/critic/dp_critic.py` (3 comment-
wrapped insertion points, ~19 lines).

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

1. Pull the LMSYS-Chat-GPT-5-Chat-Response dataset from HuggingFace
   (`ytz20/LMSYS-Chat-GPT-5-Chat-Response`).
2. Convert with the upstream `microsoft/LMOps/gad/tools/export_lmsys_parquet.py`.
3. Point `data.train_files` at the resulting parquet.

## Discriminator setup

Bring your own discriminator checkpoint (typically an SFT model trained
on teacher responses — i.e. the upstream "warmup" stage, which is out
of scope for this EasyOPD integration). Set:

```yaml
gad:
  enable: true
  discriminator_init_path: /path/to/discriminator.ckpt
critic:
  model:
    path: ${gad.discriminator_init_path}
```

## Reference

```bibtex
@article{ye2025blackboxonpolicydistillationlarge,
  title={Black-Box On-Policy Distillation of Large Language Models},
  author={Tianzhu Ye and Li Dong and Zewen Chi and Xun Wu and Shaohan Huang and Furu Wei},
  journal={arXiv preprint arXiv:2511.10643},
  year={2025},
  url={https://arxiv.org/abs/2511.10643}
}
```
