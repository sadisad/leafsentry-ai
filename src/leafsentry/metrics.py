"""Low-cardinality Prometheus instrumentation."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

_ALLOWED_OUTCOMES = frozenset({"accepted", "abstained", "error"})


class Metrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.registry = registry
        self.predictions = Counter(
            "leafsentry_predictions_total",
            "Prediction requests by bounded outcome.",
            ("outcome",),
            registry=registry,
        )
        self.latency = Histogram(
            "leafsentry_prediction_duration_seconds",
            "End-to-end prediction request duration.",
            ("outcome",),
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=registry,
        )

    def observe(self, *, outcome: str, duration_seconds: float) -> None:
        if outcome not in _ALLOWED_OUTCOMES:
            raise ValueError("unsupported metrics outcome")
        self.predictions.labels(outcome=outcome).inc()
        self.latency.labels(outcome=outcome).observe(max(0.0, duration_seconds))

    def snapshot(self) -> dict[str, int]:
        """Prediction counts by outcome, for the public stats endpoint.

        Read-only view over the same counters exported at /metrics; values
        reset when the process restarts, mirroring Prometheus semantics.
        """
        counts: dict[str, int] = {}
        for label_values, child in self.predictions._metrics.items():
            outcome = label_values[0] if isinstance(label_values, tuple) else label_values
            counts[str(outcome)] = int(child._value.get())  # type: ignore[attr-defined]
        return counts
