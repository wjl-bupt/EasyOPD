"""
Analyze why span logits are much larger than overlap logits in SimCT.

Core question: span_logit = mean(model_logits[correct_token_id]) for each token
in the span. This is the model's confidence in the correct next token.
How does this compare to overlap_logits = model_logits[overlap_token_ids]?

Usage:
    CUDA_VISIBLE_DEVICES=0 python analyze_span_logits.py
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD")
from easyopd.methods.simple.alignment import find_overlap_tokens

# ============================================================
# Config
# ============================================================
STUDENT_MODEL = "/root/workspace/models/runs/01_cross_tokenizer_opd/sft/sft_phi4mini/hf/global_step_78"
TEACHER_MODEL = "/root/workspace/models/Qwen2.5-7B-Instruct"

# A simple math prompt + response to analyze
TEST_TEXT = """<|user|>
Solve the equation: 2x + 3 = 7. Show your work step by step.
<|assistant|>
To solve 2x + 3 = 7:

Step 1: Subtract 3 from both sides
2x + 3 - 3 = 7 - 3
2x = 4

Step 2: Divide both sides by 2
x = 4/2
x = 2

Therefore, x = 2."""


def main():
    print("=" * 80)
    print("SimCT Span Logit Analysis")
    print("=" * 80)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load tokenizers
    print("\n[1] Loading tokenizers...")
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)

    # Find overlap tokens
    print("[2] Finding overlap tokens...")
    student_overlap_ids_list, teacher_overlap_ids_list = find_overlap_tokens(student_tokenizer, teacher_tokenizer)
    num_overlap = len(student_overlap_ids_list)
    print(f"    Overlap vocab size: {num_overlap}")
    print(f"    Student vocab size: {student_tokenizer.vocab_size}")
    print(f"    Teacher vocab size: {teacher_tokenizer.vocab_size}")

    # Load student model only (enough to demonstrate the issue)
    print("\n[3] Loading student model (phi4-mini) on cuda:0...")
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    student_model.eval()

    # Tokenize
    print("\n[4] Tokenizing...")
    input_ids = student_tokenizer.encode(TEST_TEXT, return_tensors="pt").to("cuda:0")
    seq_len = input_ids.shape[1]
    print(f"    Sequence length: {seq_len}")

    # Find where response starts (after <|assistant|>)
    assistant_token = student_tokenizer.encode("<|assistant|>", add_special_tokens=False)
    # Find the position after the assistant token
    input_ids_list = input_ids[0].tolist()
    response_start = None
    for i in range(len(input_ids_list) - len(assistant_token)):
        if input_ids_list[i:i+len(assistant_token)] == assistant_token:
            response_start = i + len(assistant_token)
            break
    if response_start is None:
        # Fallback: assume first 30 tokens are prompt
        response_start = 30
    print(f"    Response starts at position: {response_start}")

    # Forward pass
    print("\n[5] Running forward pass...")
    with torch.no_grad():
        outputs = student_model(input_ids)
        logits = outputs.logits[0].float()  # [seq_len, vocab_size]

    # Analyze response positions (logits at pos i predict token at pos i+1)
    student_overlap_ids_t = torch.tensor(student_overlap_ids_list, dtype=torch.long, device="cuda:0")

    print("\n" + "=" * 80)
    print("ANALYSIS: Overlap Logits vs Correct-Token Logit (Span Logit)")
    print("=" * 80)
    print(f"\nFor each response position:")
    print(f"  - overlap_logits = logits[overlap_token_ids]  (what SimCT uses for overlap dims)")
    print(f"  - correct_logit = logits[next_token_id]       (what SimCT uses for span dim)")
    print()

    all_overlap_means = []
    all_overlap_maxs = []
    all_overlap_stds = []
    all_overlap_medians = []
    all_correct_logits = []
    all_correct_ranks = []
    all_top1_logits = []

    num_positions = seq_len - response_start - 1  # -1 because last position has no next token

    print(f"{'Pos':>4} {'Next Token':>25} {'Ovlp Mean':>10} {'Ovlp Max':>9} {'Ovlp Std':>9} {'Correct':>8} {'Top1':>8} {'Rank':>5}")
    print("-" * 95)

    for i in range(num_positions):
        pos = response_start + i  # position in sequence
        logits_at_pos = logits[pos]  # [vocab_size], predicts token at pos+1
        next_token_id = input_ids_list[pos + 1]

        # Overlap logits
        overlap_logits = logits_at_pos[student_overlap_ids_t]

        # Correct (next-token) logit. NOTE: in original SimCT this fed the
        # extra "span_dim" of build_virtual_vocab_logits; in SimCT-v2 the span
        # dimension is removed and this logit is only retained at its natural
        # overlap-vocab position (no separate dim).
        correct_logit = logits_at_pos[next_token_id].item()

        # Top-1 logit for reference
        top1_logit = logits_at_pos.max().item()

        # Rank of correct token
        rank = (logits_at_pos > correct_logit).sum().item() + 1

        overlap_mean = overlap_logits.mean().item()
        overlap_max = overlap_logits.max().item()
        overlap_std = overlap_logits.std().item()
        overlap_median = overlap_logits.median().item()

        all_overlap_means.append(overlap_mean)
        all_overlap_maxs.append(overlap_max)
        all_overlap_stds.append(overlap_std)
        all_overlap_medians.append(overlap_median)
        all_correct_logits.append(correct_logit)
        all_correct_ranks.append(rank)
        all_top1_logits.append(top1_logit)

        if i < 30:  # Print first 30 positions
            token_str = student_tokenizer.decode([next_token_id]).replace('\n', '\\n')[:20]
            print(f"{i:>4} {token_str:>25} {overlap_mean:>10.3f} {overlap_max:>9.3f} {overlap_std:>9.3f} {correct_logit:>8.3f} {top1_logit:>8.3f} {rank:>5}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (across all response positions)")
    print("=" * 80)

    print(f"\n{'Metric':>30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
    print("-" * 85)
    print(f"{'Overlap logit mean':>30} {np.mean(all_overlap_means):>10.3f} {np.std(all_overlap_means):>10.3f} {np.min(all_overlap_means):>10.3f} {np.max(all_overlap_means):>10.3f} {np.median(all_overlap_means):>10.3f}")
    print(f"{'Overlap logit max':>30} {np.mean(all_overlap_maxs):>10.3f} {np.std(all_overlap_maxs):>10.3f} {np.min(all_overlap_maxs):>10.3f} {np.max(all_overlap_maxs):>10.3f} {np.median(all_overlap_maxs):>10.3f}")
    print(f"{'Overlap logit std':>30} {np.mean(all_overlap_stds):>10.3f} {np.std(all_overlap_stds):>10.3f} {np.min(all_overlap_stds):>10.3f} {np.max(all_overlap_stds):>10.3f} {np.median(all_overlap_stds):>10.3f}")
    print(f"{'Correct token logit':>30} {np.mean(all_correct_logits):>10.3f} {np.std(all_correct_logits):>10.3f} {np.min(all_correct_logits):>10.3f} {np.max(all_correct_logits):>10.3f} {np.median(all_correct_logits):>10.3f}")
    print(f"{'Top-1 logit':>30} {np.mean(all_top1_logits):>10.3f} {np.std(all_top1_logits):>10.3f} {np.min(all_top1_logits):>10.3f} {np.max(all_top1_logits):>10.3f} {np.median(all_top1_logits):>10.3f}")
    print(f"{'Correct token rank':>30} {np.mean(all_correct_ranks):>10.1f} {np.std(all_correct_ranks):>10.1f} {np.min(all_correct_ranks):>10.1f} {np.max(all_correct_ranks):>10.1f} {np.median(all_correct_ranks):>10.1f}")

    # Key gap analysis
    gap = np.mean(all_correct_logits) - np.mean(all_overlap_means)
    gap_vs_max = np.mean(all_correct_logits) - np.mean(all_overlap_maxs)
    print(f"\n*** KEY GAPS ***")
    print(f"  correct_logit - overlap_mean = {gap:.3f}")
    print(f"  correct_logit - overlap_max  = {gap_vs_max:.3f}")
    print(f"  correct_logit - overlap_median = {np.mean(all_correct_logits) - np.mean(all_overlap_medians):.3f}")

    # ============================================================
    # Simulate softmax distribution in virtual vocab
    # ============================================================
    print("\n" + "=" * 80)
    print("SIMULATION: Softmax Distribution in Virtual Vocab")
    print("=" * 80)

    # Pick 5 representative positions
    positions_to_sim = [0, num_positions//4, num_positions//2, 3*num_positions//4, num_positions-1]

    for sim_pos_idx in positions_to_sim:
        pos = response_start + sim_pos_idx
        logits_at_pos = logits[pos]
        next_token_id = input_ids_list[pos + 1]

        overlap_logits = logits_at_pos[student_overlap_ids_t]
        correct_logit = logits_at_pos[next_token_id]

        # Simulate: assume 10 spans in batch, current is span #0
        num_spans = 10
        span_dims = torch.full((num_spans,), -1e9, device="cuda:0")
        span_dims[0] = correct_logit

        virtual_logits = torch.cat([overlap_logits, span_dims])
        virtual_probs = torch.softmax(virtual_logits, dim=0)

        overlap_prob = virtual_probs[:num_overlap].sum().item()
        span_prob = virtual_probs[num_overlap].item()

        token_str = student_tokenizer.decode([next_token_id]).replace('\n', '\\n')[:15]
        print(f"\n  Pos {sim_pos_idx}: token='{token_str}', correct_logit={correct_logit.item():.2f}, overlap_max={overlap_logits.max().item():.2f}")
        print(f"    -> Span prob: {span_prob*100:.1f}%, Overlap total: {overlap_prob*100:.1f}%")

    # ============================================================
    # Experiment: Different scaling strategies
    # ============================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT: Span Logit Scaling Strategies")
    print("=" * 80)

    # Use middle position for demonstration
    mid_pos = response_start + num_positions // 2
    logits_at_pos = logits[mid_pos]
    next_token_id = input_ids_list[mid_pos + 1]
    overlap_logits = logits_at_pos[student_overlap_ids_t]
    correct_logit = logits_at_pos[next_token_id]

    ovlp_mean = overlap_logits.mean()
    ovlp_std = overlap_logits.std()
    ovlp_max = overlap_logits.max()
    ovlp_p95 = torch.quantile(overlap_logits, 0.95)
    ovlp_p99 = torch.quantile(overlap_logits, 0.99)

    print(f"\n  Reference position: correct_logit={correct_logit.item():.3f}")
    print(f"  Overlap stats: mean={ovlp_mean.item():.3f}, std={ovlp_std.item():.3f}, max={ovlp_max.item():.3f}, p95={ovlp_p95.item():.3f}, p99={ovlp_p99.item():.3f}")

    strategies = {
        "No scaling (current)": correct_logit.item(),
        "overlap_max": ovlp_max.item(),
        "mean + 1*std": (ovlp_mean + 1 * ovlp_std).item(),
        "mean + 2*std": (ovlp_mean + 2 * ovlp_std).item(),
        "mean + 3*std": (ovlp_mean + 3 * ovlp_std).item(),
        "percentile 95": ovlp_p95.item(),
        "percentile 99": ovlp_p99.item(),
        "percentile 99.9": torch.quantile(overlap_logits, 0.999).item(),
    }

    num_spans = 10
    print(f"\n  {'Strategy':>25} {'Span Logit':>10} {'Span Prob%':>10} {'Ovlp Prob%':>10}")
    print("  " + "-" * 60)

    for name, scaled_val in strategies.items():
        span_dims = torch.full((num_spans,), -1e9, device="cuda:0")
        span_dims[0] = scaled_val
        vl = torch.cat([overlap_logits, span_dims])
        vp = torch.softmax(vl, dim=0)
        span_p = vp[num_overlap].item()
        ovlp_p = vp[:num_overlap].sum().item()
        print(f"  {name:>25} {scaled_val:>10.3f} {span_p*100:>9.1f}% {ovlp_p*100:>9.1f}%")

    # ============================================================
    # KL divergence comparison
    # ============================================================
    print("\n" + "=" * 80)
    print("KL DIVERGENCE: Virtual Vocab vs Overlap-Only")
    print("=" * 80)

    # Simulate student-teacher difference
    # Teacher is typically more confident -> higher correct_logit
    teacher_gap_values = [0.5, 1.0, 2.0, 3.0, 5.0]

    print(f"\n  Simulating teacher having higher span logit by various amounts.")
    print(f"  Student span logit = {correct_logit.item():.3f}")
    print(f"\n  {'Teacher Gap':>12} {'KL(virtual)':>12} {'KL(overlap)':>12} {'Ratio':>8}")
    print("  " + "-" * 50)

    # Overlap-only KL (baseline, what Simple does)
    # Simulate teacher overlap = student overlap + small noise
    teacher_overlap = overlap_logits + torch.randn_like(overlap_logits) * 0.3
    s_logp_ovlp = F.log_softmax(overlap_logits.unsqueeze(0), dim=-1)
    t_logp_ovlp = F.log_softmax(teacher_overlap.unsqueeze(0), dim=-1)
    t_p_ovlp = t_logp_ovlp.exp()
    kl_overlap_only = (t_p_ovlp * (t_logp_ovlp - s_logp_ovlp)).sum().item()

    for teacher_gap in teacher_gap_values:
        # Virtual vocab with span
        num_spans = 10
        s_span_dims = torch.full((num_spans,), -1e9, device="cuda:0")
        s_span_dims[0] = correct_logit
        t_span_dims = torch.full((num_spans,), -1e9, device="cuda:0")
        t_span_dims[0] = correct_logit + teacher_gap

        s_virtual = torch.cat([overlap_logits, s_span_dims]).unsqueeze(0)
        t_virtual = torch.cat([teacher_overlap, t_span_dims]).unsqueeze(0)

        s_logp = F.log_softmax(s_virtual, dim=-1)
        t_logp = F.log_softmax(t_virtual, dim=-1)
        t_p = t_logp.exp()
        kl_virtual = (t_p * (t_logp - s_logp)).sum().item()

        ratio = kl_virtual / max(kl_overlap_only, 1e-10)
        print(f"  {teacher_gap:>12.1f} {kl_virtual:>12.6f} {kl_overlap_only:>12.6f} {ratio:>7.2f}x")

    # ============================================================
    # Final insight
    # ============================================================
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"""
Key findings:
1. Correct token logit (span dim) mean = {np.mean(all_correct_logits):.3f}
   Overlap logit mean = {np.mean(all_overlap_means):.3f}
   GAP = {gap:.3f}

2. This gap means the span dimension dominates the softmax distribution.
   The span dim absorbs a large fraction of probability mass.

3. The reason span logit is high: it's model_logits[correct_token_id],
   i.e., the model's confidence in the correct next token. For a well-trained
   model, this is typically rank 1-5, so its logit is near the maximum.

4. The reason overlap logits are low on average: they include ALL {num_overlap}
   overlap tokens, most of which are irrelevant at any given position.
   Only a few overlap tokens are plausible next tokens.

5. The correct token logit is close to overlap_max (gap = {gap_vs_max:.3f}),
   because the correct token is usually among the top overlap tokens.
   But it's FAR above overlap_mean (gap = {gap:.3f}).

6. This means: the span dimension's logit is essentially the same as the
   top-1 or top-2 overlap logit. After softmax in the virtual vocab,
   the span dim gets roughly the same probability as the top overlap token.
   
   BUT: if student and teacher disagree on the span logit (e.g., teacher
   is more confident by 2-3 logit units), this creates a large KL because
   the span dim has high probability mass on the teacher side.
""")


if __name__ == "__main__":
    main()
