"""Generate summary table from benchmark evaluation results."""

import json
import os
from pathlib import Path

RESULTS_DIR = "/apdcephfs_cq8/share_1324356/shinejiesun/workspace/EasyOPD/experiments/benchmark/results"

# Expected methods in order
METHODS = [
    "base_qwen2.5-1.5b",
    "sft",
    "grpo",
    "gkd",
    "sod",
    "opcd",
    "g_opd",
    "sdpo",
    "opsa",
    "ropd",
    "vision_opd",
    "simple",
    "simct",
]

METHOD_DISPLAY_NAMES = {
    "base_qwen2.5-1.5b": "Qwen2.5-1.5B (Base)",
    "sft": "SFT (Teacher KD)",
    "grpo": "GRPO (No Distill)",
    "gkd": "GKD",
    "sod": "SOD",
    "opcd": "OPCD",
    "g_opd": "G-OPD",
    "sdpo": "SDPO",
    "opsa": "OPSA",
    "ropd": "ROPD",
    "vision_opd": "Vision-OPD",
    "simple": "Simple (Cross-Tok KD)",
    "simct": "SimCT (Span Cross-Tok)",
}


def load_results():
    """Load all summary JSON files from results directory."""
    results = {}
    if not os.path.exists(RESULTS_DIR):
        return results

    for f in Path(RESULTS_DIR).glob("*_summary.json"):
        with open(f) as fp:
            data = json.load(fp)
        model_name = data["model_name"]
        results[model_name] = data.get("results", {})

    return results


def generate_markdown_table(results):
    """Generate a markdown table from results."""
    header = "| Method | MATH-500 (%) | GSM8K (%) |"
    separator = "|--------|:---:|:---:|"

    rows = [header, separator]

    for method in METHODS:
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        res = results.get(method, {})

        math500 = res.get("math500", {}).get("accuracy", None)
        gsm8k = res.get("gsm8k", {}).get("accuracy", None)

        math500_str = f"{math500:.2f}" if math500 is not None else "-"
        gsm8k_str = f"{gsm8k:.2f}" if gsm8k is not None else "-"

        rows.append(f"| {display_name} | {math500_str} | {gsm8k_str} |")

    return "\n".join(rows)


def generate_csv(results):
    """Generate CSV from results."""
    lines = ["method,math500,gsm8k"]
    for method in METHODS:
        res = results.get(method, {})
        math500 = res.get("math500", {}).get("accuracy", "")
        gsm8k = res.get("gsm8k", {}).get("accuracy", "")
        lines.append(f"{method},{math500},{gsm8k}")
    return "\n".join(lines)


def main():
    results = load_results()

    if not results:
        print("No results found yet. Run experiments first.")
        return

    # Generate markdown table
    table = generate_markdown_table(results)
    print("\n" + "=" * 60)
    print("EasyOPD Benchmark Results")
    print("=" * 60)
    print(f"\nStudent: Qwen2.5-1.5B-Instruct")
    print(f"Teacher: Qwen2.5-7B-Instruct")
    print(f"Training: 200 steps, GRPO, mixed_math_code_10k")
    print(f"Evaluation: greedy decoding, max_tokens=2048")
    print()
    print(table)

    # Save to file
    output_path = os.path.join(RESULTS_DIR, "benchmark_table.md")
    with open(output_path, "w") as f:
        f.write("# EasyOPD Benchmark Results\n\n")
        f.write("**Student Model**: Qwen2.5-1.5B-Instruct\n")
        f.write("**Teacher Model**: Qwen2.5-7B-Instruct\n")
        f.write("**Training Data**: mixed_math_code_10k (9900 samples)\n")
        f.write("**Training Steps**: 200\n")
        f.write("**Algorithm**: GRPO\n")
        f.write("**Evaluation**: greedy decoding, max_tokens=2048\n\n")
        f.write(table)
        f.write("\n")
    print(f"\nTable saved to: {output_path}")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(csv_path, "w") as f:
        f.write(generate_csv(results))
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
