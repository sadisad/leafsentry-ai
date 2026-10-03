#!/usr/bin/env python3
"""Fail-closed advisory audit of locked runtime, including official CPU wheels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from leafsentry.dependency_audit import advisory_requirements, verify_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requirements", type=Path, help="uv export with --no-hashes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    normalized = advisory_requirements(args.requirements.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Delete stale evidence, so a failed auditor cannot reuse a previous green result.
    args.output.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="leafsentry-audit-") as directory:
        requirements = Path(directory) / "advisory-versions.txt"
        requirements.write_text(normalized, encoding="utf-8")
        # Fixed interpreter/module; paths are separate argv elements, never shell commands.
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--no-deps",
                "--disable-pip",
                "-r",
                str(requirements),
                "--format",
                "json",
                "--output",
                str(args.output),
            ],
            check=False,
            timeout=180,
        )
    if result.returncode != 0:
        return result.returncode
    report = json.loads(args.output.read_text(encoding="utf-8"))
    count = verify_audit(report)
    print(f"Audited {count} upstream release versions: zero skips and known advisories.")
    print("CPU local suffix mapping is for advisory lookup only; install hashes are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
