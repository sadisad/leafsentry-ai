"""Command-line evaluation utilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from leafsentry.evaluation import EvaluationRow, evaluate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leafsentry")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluation = subparsers.add_parser("evaluate", help="Evaluate JSONL probabilities")
    evaluation.add_argument("predictions", type=Path)
    evaluation.add_argument("--min-confidence", type=float, default=0.75)
    evaluation.add_argument("--min-margin", type=float, default=0.20)
    evaluation.add_argument("--ece-bins", type=int, default=10)
    return parser


def _load_rows(path: Path) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                target = payload["target"]
                probabilities = payload["probabilities"]
                if not isinstance(target, str) or not isinstance(probabilities, dict):
                    raise TypeError
                rows.append(
                    EvaluationRow(
                        target=target,
                        probabilities={
                            str(key): float(value) for key, value in probabilities.items()
                        },
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid JSONL row {line_number}") from exc
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        report = evaluate(
            _load_rows(args.predictions),
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            ece_bins=args.ece_bins,
        )
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")
