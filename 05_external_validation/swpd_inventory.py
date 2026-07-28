#!/usr/bin/env python3
"""Inventory the read-only SWPD development subject (sub-01 only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from whisper_ecog_ext.swpd.nwb import (  # noqa: E402
    ConfirmatoryDataLocked,
    NWBLayoutError,
    PILOT_SUBJECT,
    inventory_pilot,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--subject",
        default=PILOT_SUBJECT,
        choices=[PILOT_SUBJECT],
        help="Confirmatory participants are deliberately unavailable",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    inventory = inventory_pilot(args.data_root).to_dict()
    rendered = json.dumps(inventory, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
        print(f"[saved] {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfirmatoryDataLocked, NWBLayoutError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
