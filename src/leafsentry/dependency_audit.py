"""Advisory lookup for official CPU builds, without silently skipping packages."""

from __future__ import annotations

import re
from typing import Any

_CPU_RELEASE = re.compile(r"^(torch(?:vision)?==)([0-9]+(?:\.[0-9]+)*)\+cpu(?=\s|$)")
_LOCAL_RELEASE = re.compile(r"^[A-Za-z0-9_.-]+==[^\s;]+\+")


def advisory_requirements(source: str) -> str:
    """Map only official torch/torchvision +cpu builds to upstream advisory versions.

    This changes the advisory-query input, never the installation lock or hashes.
    Platform markers and unrelated requirements are preserved.
    """
    lines: list[str] = []
    for line in source.splitlines():
        normalized = _CPU_RELEASE.sub(r"\1\2", line)
        if _LOCAL_RELEASE.match(normalized):
            raise ValueError("unknown local-build version cannot be mapped for advisory lookup")
        lines.append(normalized)
    return "\n".join(lines) + "\n"


def verify_audit(report: dict[str, Any]) -> int:
    """Require nonempty audit evidence with no skipped packages or advisories."""
    dependencies = report.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("audit report must contain dependencies")
    for dependency in dependencies:
        if not isinstance(dependency, dict) or not dependency.get("name"):
            raise ValueError("audit report contains an invalid dependency")
        if dependency.get("skip_reason"):
            raise ValueError(f"audit skipped {dependency['name']}")
        if not isinstance(dependency.get("version"), str) or not dependency["version"].strip():
            raise ValueError("audit report omits dependency version")
        if "vulns" not in dependency or not isinstance(dependency["vulns"], list):
            raise ValueError("audit report omits vulnerability evidence")
        if dependency["vulns"]:
            raise ValueError(f"audit found known advisories for {dependency['name']}")
    return len(dependencies)
