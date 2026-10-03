from pathlib import Path

import pytest

from leafsentry.model_smoke import run_smoke


@pytest.mark.model
def test_pinned_model_matches_checksum_verified_examples(tmp_path: Path) -> None:
    report = run_smoke(
        Path("tests/model/sample_manifest.json"),
        tmp_path / "samples",
    )

    assert report["all_top1_matched"] is True
    assert [sample["expected_label"] for sample in report["samples"]] == [
        "angular_leaf_spot",
        "bean_rust",
        "healthy",
    ]
    assert all(sample["matched"] for sample in report["samples"])
