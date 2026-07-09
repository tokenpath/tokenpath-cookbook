"""Latency and cost aggregation for the results table.

Latency lives in the results table (per the plan), not a separate section. We
report p50 and p95 wall-clock per method, plus mean $/query. All inputs are real
measurements collected at call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MethodCost:
    """Accumulates per-query latency and dollar cost for one method."""

    name: str
    latencies: list[float] = field(default_factory=list)  # seconds per query
    costs: list[float] = field(default_factory=list)  # USD per query

    def add(self, seconds: float, usd: float) -> None:
        self.latencies.append(seconds)
        self.costs.append(usd)

    def summary(self) -> dict:
        lat = np.array(self.latencies) if self.latencies else np.array([0.0])
        cost = np.array(self.costs) if self.costs else np.array([0.0])
        return {
            "n": len(self.latencies),
            "latency_p50_s": round(float(np.percentile(lat, 50)), 3),
            "latency_p95_s": round(float(np.percentile(lat, 95)), 3),
            "usd_per_query_mean": round(float(cost.mean()), 6),
            "usd_total": round(float(cost.sum()), 4),
        }


def tokenpath_cost_usd(attributed_tokens: int, usd_per_mtok: float) -> float:
    """TokenPath bills per attributed token; answer+document token count feeds this."""
    return attributed_tokens / 1_000_000 * usd_per_mtok
