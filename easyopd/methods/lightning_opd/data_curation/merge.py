"""Merge Arrow IPC files into a single parquet.

Recursively searches the input directory for ``.arrow`` files,
concatenates them with PyArrow, and writes a single parquet.

Usage:
    python3 -m easyopd.methods.lightning_opd.data_curation.merge \\
        --input-dir data/rollouts/raw \\
        --output data/rollouts/rollouts.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Merge Arrow IPC files into a single parquet.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with .arrow files.")
    parser.add_argument("--output", type=str, required=True, help="Output parquet path.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Discard rows exceeding this token count.")
    return parser.parse_args(args)


def main(args=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parsed = parse_args(args)
    input_dir = Path(parsed.input_dir)
    arrow_files = sorted(input_dir.rglob("*.arrow"))

    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {input_dir}")

    logger.info("Found %d arrow files in %s", len(arrow_files), input_dir)

    tables = []
    for f in arrow_files:
        reader = pa.ipc.open_file(str(f))
        table = reader.read_all()
        tables.append(table)

    merged = pa.concat_tables(tables)
    logger.info("Merged rows: %d", merged.num_rows)

    if parsed.max_tokens is not None and "tokens" in merged.column_names:
        mask = pa.compute.less_equal(merged.column("tokens"), parsed.max_tokens)
        merged = merged.filter(mask)
        logger.info("After filtering (max_tokens=%d): %d rows", parsed.max_tokens, merged.num_rows)

    output_path = Path(parsed.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, str(output_path))
    logger.info("Written to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
