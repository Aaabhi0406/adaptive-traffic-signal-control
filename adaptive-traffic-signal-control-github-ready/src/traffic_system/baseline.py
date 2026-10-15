"""
baseline.py — fixed-timer signal baseline + delay estimation (Phase 1).

Every claim like "adaptive beats fixed-timer by X%" needs a fixed-timer
baseline to beat, and a delay metric that isn't just eyeballing green_time
numbers. This module gives both:

  FixedTimerBaseline   — the classical, non-adaptive control: same green
                          time every cycle regardless of measured demand.
  estimate_delay()     — a simplified vehicle-seconds-of-delay estimate from
                          a per-frame count/green-time trace, used to compare
                          any two controllers (adaptive vs fixed, or later,
                          RF-driven vs RL-driven) on the same trace.

The delay model is intentionally simple (documented, not hidden) — it is a
discrete queueing approximation, not a calibrated traffic-engineering model.
Treat the *relative* comparison between controllers on the same trace as the
useful number, not the absolute delay figure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FixedTimerBaseline:
    """Cycles through a fixed green time regardless of measured demand.
    This is the classical control every adaptive claim should be measured
    against."""

    green_time_s: int = 25

    def get_signal_time(self, vehicle_count: int) -> tuple[int, str]:
        # Deliberately ignores vehicle_count — that's the point of a
        # fixed-timer baseline. Level label kept for interface parity with
        # signal_logic.get_signal_time().
        return self.green_time_s, "FIXED"


def estimate_delay(
    counts: list[int],
    green_times: list[int],
    saturation_flow_per_s: float = 0.5,
    avg_wait_s: float = 30.0,
) -> float:
    """Rough total vehicle-seconds-of-delay estimate for a trace.

    For each sample, treat `vehicle_count` as the queue present when the
    signal is red, and estimate how much of that queue clears once green
    starts (at `saturation_flow_per_s` vehicles/sec, a standard traffic-
    engineering rule-of-thumb for a single lane). Vehicles that don't clear
    carry over to the next cycle's queue.

    Delay is charged per unserved vehicle at a fixed `avg_wait_s` (a nominal
    wait unit, not the green time itself) — charging it against green_s
    directly would perversely *increase* the delay figure for a longer green
    that serves more vehicles, since the same small leftover queue would be
    multiplied by a bigger number. Fixing the wait unit keeps the metric
    monotonic: more effective green time never reports as worse.

    This is a discrete approximation (not a calibrated micro-simulation) —
    useful for *comparing* two controllers on the identical trace, not for
    quoting an absolute real-world delay figure.
    """
    if len(counts) != len(green_times):
        raise ValueError("counts and green_times must be the same length")

    total_delay = 0.0
    carryover = 0.0

    for count, green_s in zip(counts, green_times):
        queue = carryover + count
        served = min(queue, saturation_flow_per_s * green_s)
        unserved = max(0.0, queue - served)

        total_delay += unserved * avg_wait_s
        carryover = unserved

    return total_delay


def compare_controllers(
    counts: list[int],
    adaptive_green_times: list[int],
    fixed_green_time: int = 25,
) -> dict:
    """Run the same count trace through the adaptive green-time sequence
    already computed by the pipeline, and through a fixed-timer baseline,
    and report both delays plus the percentage improvement.

    Returns a dict suitable for logging or dumping to a benchmark report —
    see scripts/benchmark.py.
    """
    fixed_green_times = [fixed_green_time] * len(counts)

    adaptive_delay = estimate_delay(counts, adaptive_green_times)
    fixed_delay = estimate_delay(counts, fixed_green_times)

    improvement_pct = (
        100.0 * (fixed_delay - adaptive_delay) / fixed_delay if fixed_delay > 0 else 0.0
    )

    return {
        "adaptive_delay_vehicle_seconds": adaptive_delay,
        "fixed_delay_vehicle_seconds": fixed_delay,
        "improvement_pct": improvement_pct,
        "n_samples": len(counts),
    }
