"""Tests for baseline.py — fixed-timer control + delay estimation, the
Phase 1 comparison point every adaptive claim needs."""

import pytest

from traffic_system.baseline import (
    FixedTimerBaseline,
    compare_controllers,
    estimate_delay,
)


class TestFixedTimerBaseline:
    def test_ignores_vehicle_count(self):
        baseline = FixedTimerBaseline(green_time_s=25)
        g1, _ = baseline.get_signal_time(0)
        g2, _ = baseline.get_signal_time(500)
        assert g1 == g2 == 25

    def test_level_label_is_fixed(self):
        baseline = FixedTimerBaseline()
        _, level = baseline.get_signal_time(10)
        assert level == "FIXED"


class TestEstimateDelay:
    def test_empty_trace_has_zero_delay(self):
        assert estimate_delay([], []) == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            estimate_delay([1, 2], [10])

    def test_light_traffic_fully_served_has_zero_carryover_delay(self):
        # 1 vehicle, 25s green, 0.5 veh/s saturation flow -> easily cleared.
        delay = estimate_delay([1], [25])
        assert delay == 0.0

    def test_heavy_traffic_produces_positive_delay(self):
        # Far more vehicles than a short green can clear.
        delay = estimate_delay([100], [10])
        assert delay > 0.0

    def test_more_green_time_never_increases_delay(self):
        counts = [40, 40, 40]
        short_green = estimate_delay(counts, [10, 10, 10])
        long_green = estimate_delay(counts, [40, 40, 40])
        assert long_green <= short_green


class TestCompareControllers:
    def test_adaptive_matching_demand_beats_undersized_fixed_timer(self):
        counts = [5, 30, 5, 30, 5]
        adaptive_green_times = [15, 40, 15, 40, 15]  # tracks demand
        result = compare_controllers(counts, adaptive_green_times, fixed_green_time=15)
        assert result["improvement_pct"] > 0

    def test_result_has_expected_keys(self):
        result = compare_controllers([5, 5], [15, 15], fixed_green_time=15)
        assert set(result.keys()) == {
            "adaptive_delay_vehicle_seconds",
            "fixed_delay_vehicle_seconds",
            "improvement_pct",
            "n_samples",
        }

    def test_n_samples_matches_input_length(self):
        result = compare_controllers([1, 2, 3], [15, 15, 15])
        assert result["n_samples"] == 3
