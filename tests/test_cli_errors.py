import json

import pytest

from leafsentry.cli import main
from leafsentry.evaluation import EvaluationRow, evaluate


@pytest.mark.parametrize("payload", ["not-json", '{"target":1,"probabilities":{}}', "{}"])
def test_cli_rejects_malformed_rows_with_line_number(tmp_path, payload):
    path = tmp_path / "bad.jsonl"
    path.write_text(payload)
    with pytest.raises(ValueError, match="row 1"):
        main(["evaluate", str(path)])


def test_cli_skips_blank_lines(tmp_path, capsys):
    path = tmp_path / "good.jsonl"
    path.write_text('\n{"target":"a","probabilities":{"a":0.9,"b":0.1}}\n\n')
    assert main(["evaluate", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 1


def test_evaluation_rejects_empty_data():
    with pytest.raises(ValueError):
        evaluate([], min_confidence=0, min_margin=0)


def test_evaluation_rejects_unknown_target():
    with pytest.raises(ValueError):
        evaluate([EvaluationRow("missing", {"a": 0.5, "b": 0.5})], min_confidence=0, min_margin=0)


def test_evaluation_rejects_invalid_ece_bins():
    with pytest.raises(ValueError):
        evaluate(
            [EvaluationRow("a", {"a": 0.5, "b": 0.5})], min_confidence=0, min_margin=0, ece_bins=0
        )
