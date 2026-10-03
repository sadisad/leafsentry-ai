#!/usr/bin/env python3
"""Run the pinned LeafSentry model against checksum-verified upstream samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leafsentry.model_smoke import run_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/model/sample_manifest.json"),
    )
    parser.add_argument("--sample-dir", type=Path, default=Path(".cache/model-smoke/samples"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_smoke(args.manifest, args.sample_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["all_top1_matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
