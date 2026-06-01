#!/usr/bin/env python3
# Copyright 2026 EasyOPD Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EasyOPD Unified Training Entry Point.

This script provides a single entry point for running any EasyOPD method.
It handles method discovery, configuration loading, and training launch.

Usage::

    python scripts/run_easyopd.py --method gkd --config easyopd/config/gkd.yaml
    python scripts/run_easyopd.py --method simple --config my_custom_config.yaml
    python scripts/run_easyopd.py --method sod  # uses default config

    # List available methods:
    python scripts/run_easyopd.py --list-methods
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="EasyOPD: Unified On-Policy Distillation Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_easyopd.py --method gkd --config easyopd/config/gkd.yaml
  python scripts/run_easyopd.py --method simple
  python scripts/run_easyopd.py --list-methods
        """,
    )

    parser.add_argument(
        "--method",
        type=str,
        help="Name of the OPD method to run (e.g. simple, gkd, sod, g_opd, opcd, vision_opd, sdpo)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to method config YAML file. If not specified, uses default config.",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List all available OPD methods and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and validate without starting training.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    # Allow additional hydra-style overrides
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional config overrides in key=value format.",
    )

    return parser.parse_args()


def parse_overrides(override_list: list[str]) -> dict:
    """Parse key=value override strings into a dict.

    Supports nested keys with dot notation (e.g. distillation.loss_mode=gkd).
    """
    overrides = {}
    for item in override_list:
        if "=" not in item:
            print(f"Warning: Ignoring invalid override '{item}' (expected key=value format)")
            continue
        key, value = item.split("=", 1)

        # Try to parse value as Python literal
        try:
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # Keep as string

        # Handle nested keys (dot notation)
        keys = key.split(".")
        d = overrides
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return overrides


def list_methods_and_exit():
    """Print all available methods and exit."""
    from easyopd import EasyOPD

    methods = EasyOPD.list_methods()
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║           EasyOPD: Available OPD Methods                    ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    for name in methods:
        method_cls = EasyOPD.get_method(name)
        desc = getattr(method_cls, "description", "No description")
        paper = getattr(method_cls, "paper_url", "N/A")
        print(f"║  {name:<12} │ {desc[:42]:<42} ║")

    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nTotal: {len(methods)} methods available.")
    print("\nUsage: python scripts/run_easyopd.py --method <name> [--config <path>]")
    sys.exit(0)


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # List methods mode
    if args.list_methods:
        list_methods_and_exit()

    # Validate method argument
    if args.method is None:
        print("Error: --method is required. Use --list-methods to see available methods.")
        sys.exit(1)

    # Import EasyOPD
    from easyopd import EasyOPD
    from easyopd.config_loader import load_method_config, validate_config

    # Parse overrides
    overrides = parse_overrides(args.overrides) if args.overrides else {}

    # Load configuration
    print(f"\n{'='*60}")
    print(f"  EasyOPD Training: method={args.method}")
    print(f"{'='*60}")

    try:
        config = load_method_config(
            config_path=args.config,
            method_name=args.method,
            overrides=overrides,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Validate configuration
    try:
        warnings = validate_config(config, method_name=args.method)
        for w in warnings:
            print(f"  Warning: {w}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load method
    try:
        instance = EasyOPD.from_hparams(
            args.method,
            config_path=args.config,
            **overrides,
        )
    except Exception as e:
        print(f"Error loading method '{args.method}': {e}")
        sys.exit(1)

    print(f"  Method: {instance.method_name}")
    print(f"  Description: {instance.description}")
    if instance.paper_url:
        print(f"  Paper: {instance.paper_url}")
    print(f"  Config keys: {list(instance.config.keys())}")
    print(f"{'='*60}\n")

    # Dry run mode
    if args.dry_run:
        print("Dry run complete. Configuration is valid.")
        print(f"  Method: {instance.method_name}")
        print(f"  Config: {instance.config}")
        sys.exit(0)

    # Start training
    print("Starting training...")
    print("Note: Full training integration requires verl hook dispatch setup.")
    print("Please use the method-specific run scripts in examples/ for now,")
    print("or configure verl with easyopd.method.name in your training config.")


if __name__ == "__main__":
    main()
