import pytest

from leafsentry.dependency_audit import advisory_requirements, verify_audit


def test_cpu_wheels_map_to_upstream_release_without_losing_markers() -> None:
    source = (
        "torch==2.13.0+cpu ; sys_platform == 'linux'\n"
        "torchvision==0.28.0+cpu ; sys_platform == 'linux'\n"
        "torch==2.13.0 ; sys_platform == 'darwin'\n"
        "transformers==5.10.1\n"
    )
    output = advisory_requirements(source)

    assert "torch==2.13.0 ; sys_platform == 'linux'" in output
    assert "torchvision==0.28.0 ; sys_platform == 'linux'" in output
    assert "torch==2.13.0 ; sys_platform == 'darwin'" in output
    assert "transformers==5.10.1" in output
    assert "+cpu" not in output


def test_unknown_local_build_fails_instead_of_being_silently_rewritten() -> None:
    with pytest.raises(ValueError, match="unknown local-build"):
        advisory_requirements("torch==2.13.0+untrusted\n")


def test_skipped_packages_fail_audit_even_when_no_advisories_are_returned() -> None:
    report = {"dependencies": [{"name": "torch", "skip_reason": "not found on PyPI"}]}
    with pytest.raises(ValueError, match="skipped"):
        verify_audit(report)


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"dependencies": []},
        {"dependencies": ["invalid"]},
        {"dependencies": [{}]},
        {"dependencies": [{"name": "torch"}]},
        {"dependencies": [{"name": "torch", "vulns": []}]},
        {"dependencies": [{"name": "torch", "version": "", "vulns": []}]},
        {"dependencies": [{"name": "torch", "version": 213, "vulns": []}]},
        {"dependencies": [{"name": "torch", "version": "2.13.0", "vulns": "invalid"}]},
        {
            "dependencies": [
                {"name": "torch", "version": "2.13.0", "vulns": [{"id": "known-advisory"}]}
            ]
        },
    ],
)
def test_incomplete_or_vulnerable_audit_report_is_rejected(report) -> None:
    with pytest.raises(ValueError):
        verify_audit(report)


def test_complete_audit_counts_packages_without_skips() -> None:
    report = {
        "dependencies": [
            {"name": "torch", "version": "2.13.0", "vulns": []},
            {"name": "torchvision", "version": "0.28.0", "vulns": []},
        ]
    }
    assert verify_audit(report) == 2
