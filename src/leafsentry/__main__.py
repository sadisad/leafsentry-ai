"""Run the LeafSentry API with Uvicorn."""

from __future__ import annotations

import uvicorn


def main() -> None:
    # The container must be reachable through its published port; deployment controls exposure.
    uvicorn.run(
        "leafsentry.api:create_app",
        factory=True,
        host="0.0.0.0",  # noqa: S104 -- required for container networking
        port=8000,
    )


if __name__ == "__main__":
    main()
