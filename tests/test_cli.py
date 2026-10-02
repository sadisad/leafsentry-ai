import json

from leafsentry.cli import main


def test_evaluate_cli_writes_deterministic_json(tmp_path, capsys) -> None:
    rows = [
        {"target": "healthy", "probabilities": {"healthy": 0.9, "rust": 0.1}},
        {"target": "rust", "probabilities": {"healthy": 0.6, "rust": 0.4}},
    ]
    input_path = tmp_path / "predictions.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    exit_code = main(
        [
            "evaluate",
            str(input_path),
            "--min-confidence",
            "0.75",
            "--min-margin",
            "0.2",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["samples"] == 2
    assert payload["accepted"] == 1
    assert payload["coverage"] == 0.5
    assert payload["selective_accuracy"] == 1.0
