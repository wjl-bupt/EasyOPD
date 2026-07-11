"""Generate summary table from benchmark evaluation results.

Usage:
    python generate_table.py <experiment_dir> [--out <dir>]

Scans <experiment_dir>/methods/*/results/*_summary.json and aggregates them
into a markdown table + csv. Output files are written into --out
(default: <experiment_dir>/results/).
"""

import argparse
import json
import os
from pathlib import Path

EXPERIMENTS_ROOT = "/path/to/EasyOPD/experiments"

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


def load_results(experiment_dir: str):
    """Load all <method>/results/*_summary.json files under experiment_dir/methods/."""
    results = {}
    methods_root = Path(experiment_dir) / "methods"
    if not methods_root.exists():
        print(f"[WARN] methods dir not found: {methods_root}")
        return results

    for method_dir in methods_root.iterdir():
        if not method_dir.is_dir():
            continue
        results_dir = method_dir / "results"
        if not results_dir.exists():
            continue
        for f in results_dir.glob("*_summary.json"):
            with open(f) as fp:
                data = json.load(fp)
            model_name = data.get("model_name", f.stem.replace("_summary", ""))
            # Tag each entry with which method directory it came from so multiple
            # ckpts from the same method don't collide.
            results[model_name] = {
                "method_dir": method_dir.name,
                "results": data.get("results", {}),
                "summary_path": str(f),
            }

    return results


def generate_markdown_table(results):
    """Generate a markdown table. One row per discovered model_name."""
    header = "| Method dir | Model | MATH-500 (%) | GSM8K (%) |"
    separator = "|---|---|:---:|:---:|"

    rows = [header, separator]

    # Group by method_dir, ordered by METHODS list (unknown methods appended at end).
    def method_order(name):
        return METHODS.index(name) if name in METHODS else len(METHODS)

    grouped = sorted(
        results.items(),
        key=lambda kv: (method_order(kv[1]["method_dir"]), kv[0]),
    )

    for model_name, entry in grouped:
        method_dir = entry["method_dir"]
        display = METHOD_DISPLAY_NAMES.get(method_dir, method_dir)
        res = entry["results"]

        math500 = res.get("math500", {}).get("accuracy", None)
        gsm8k = res.get("gsm8k", {}).get("accuracy", None)

        math500_str = f"{math500:.2f}" if math500 is not None else "-"
        gsm8k_str = f"{gsm8k:.2f}" if gsm8k is not None else "-"

        rows.append(f"| {display} | `{model_name}` | {math500_str} | {gsm8k_str} |")

    return "\n".join(rows)


def generate_csv(results):
    """Generate CSV from results."""
    lines = ["method_dir,model_name,math500,gsm8k"]
    for model_name, entry in sorted(results.items()):
        method_dir = entry["method_dir"]
        res = entry["results"]
        math500 = res.get("math500", {}).get("accuracy", "")
        gsm8k = res.get("gsm8k", {}).get("accuracy", "")
        lines.append(f"{method_dir},{model_name},{math500},{gsm8k}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", nargs="?",
                        default=os.path.join(EXPERIMENTS_ROOT, "01_cross_tokenizer_opd"),
                        help="Path to <experiment> dir (the one containing methods/).")
    parser.add_argument("--out", default=None,
                        help="Output dir for benchmark_table.md / benchmark_results.csv "
                             "(default: <experiment_dir>/results/).")
    args = parser.parse_args()

    experiment_dir = os.path.abspath(args.experiment_dir)
    out_dir = args.out or os.path.join(experiment_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    results = load_results(experiment_dir)

    if not results:
        print(f"No results found under {experiment_dir}/methods/*/results/. Run experiments first.")
        return

    table = generate_markdown_table(results)
    print("\n" + "=" * 60)
    print(f"Benchmark Results -- {os.path.basename(experiment_dir)}")
    print("=" * 60)
    print(table)

    output_path = os.path.join(out_dir, "benchmark_table.md")
    with open(output_path, "w") as f:
        f.write(f"# Benchmark Results -- {os.path.basename(experiment_dir)}\n\n")
        f.write(table)
        f.write("\n")
    print(f"\nTable saved to: {output_path}")

    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    with open(csv_path, "w") as f:
        f.write(generate_csv(results))
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
