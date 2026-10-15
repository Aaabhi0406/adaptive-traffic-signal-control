"""Tests for the pure-logic 4-way allocator (intersection.allocate_cycle).
No video, no YOLO, no DeepSort — just the Webster-style split."""

import pytest

from traffic_system.intersection import (
    MAX_GREEN_S,
    MIN_GREEN_S,
    TOTAL_CYCLE_S,
    compute_is_demo_mode,
    allocate_cycle,
    fuse_demand,
)


class TestAllocateCycle:
    def test_equal_demand_splits_evenly(self):
        densities = {"N": 10.0, "E": 10.0, "S": 10.0, "W": 10.0}
        allocation = allocate_cycle(densities)
        values = list(allocation.values())
        assert all(v == values[0] for v in values)

    def test_allocation_sums_to_total_cycle(self):
        densities = {"N": 30.0, "E": 5.0, "S": 12.0, "W": 8.0}
        allocation = allocate_cycle(densities)
        assert sum(allocation.values()) == TOTAL_CYCLE_S

    def test_zero_demand_everywhere_splits_evenly(self):
        densities = {"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0}
        allocation = allocate_cycle(densities)
        values = list(allocation.values())
        assert all(v == values[0] for v in values)

    def test_higher_demand_gets_more_green(self):
        densities = {"N": 50.0, "E": 5.0, "S": 5.0, "W": 5.0}
        allocation = allocate_cycle(densities)
        assert allocation["N"] > allocation["E"]
        assert allocation["N"] > allocation["S"]
        assert allocation["N"] > allocation["W"]

    def test_no_approach_starves_below_min_green(self):
        # Extreme skew: one approach has almost all the demand.
        densities = {"N": 1000.0, "E": 0.01, "S": 0.01, "W": 0.01}
        allocation = allocate_cycle(densities)
        assert all(v >= MIN_GREEN_S for v in allocation.values())

    def test_no_approach_exceeds_max_green(self):
        densities = {"N": 1000.0, "E": 0.01, "S": 0.01, "W": 0.01}
        allocation = allocate_cycle(densities)
        assert all(v <= MAX_GREEN_S for v in allocation.values())

    def test_sum_never_exceeds_total_after_clamping(self):
        # Extreme skew: N is clamped at MAX_GREEN_S while E/S/W are already
        # pinned at MIN_GREEN_S, so there's nothing left to redistribute
        # into — the sum legitimately lands under TOTAL_CYCLE_S (see
        # allocate_cycle's docstring) rather than breaking the max clamp.
        densities = {"N": 1000.0, "E": 0.01, "S": 0.01, "W": 0.01}
        allocation = allocate_cycle(densities)
        assert sum(allocation.values()) <= TOTAL_CYCLE_S

    def test_moderate_skew_still_sums_to_total(self):
        # Less extreme skew leaves room for redistribution to close the gap
        # exactly, which is the common case.
        densities = {"N": 40.0, "E": 20.0, "S": 20.0, "W": 20.0}
        allocation = allocate_cycle(densities)
        assert sum(allocation.values()) == TOTAL_CYCLE_S

    def test_custom_total_cycle_respected(self):
        densities = {"N": 10.0, "E": 10.0, "S": 10.0, "W": 10.0}
        allocation = allocate_cycle(densities, total_cycle_s=80)
        assert sum(allocation.values()) == 80

    def test_returns_int_seconds_for_every_approach(self):
        densities = {"N": 12.0, "E": 3.0, "S": 20.0, "W": 7.0}
        allocation = allocate_cycle(densities)
        assert all(isinstance(v, int) for v in allocation.values())
        assert set(allocation.keys()) == set(densities.keys())


class TestFuseDemand:
    def test_zero_weight_is_pure_current_demand(self):
        current = {"N": 10.0, "E": 5.0, "S": 8.0, "W": 2.0}
        predicted = {"N": 30.0, "E": 30.0, "S": 30.0, "W": 30.0}
        fused = fuse_demand(current, predicted, prediction_weight=0.0)
        assert fused == current

    def test_full_weight_is_pure_prediction(self):
        current = {"N": 10.0, "E": 5.0, "S": 8.0, "W": 2.0}
        predicted = {"N": 30.0, "E": 12.0, "S": 6.0, "W": 1.0}
        fused = fuse_demand(current, predicted, prediction_weight=1.0)
        assert fused == predicted

    def test_half_weight_averages(self):
        current = {"N": 10.0, "E": 10.0, "S": 10.0, "W": 10.0}
        predicted = {"N": 20.0, "E": 20.0, "S": 20.0, "W": 20.0}
        fused = fuse_demand(current, predicted, prediction_weight=0.5)
        assert all(v == pytest.approx(15.0) for v in fused.values())

    def test_out_of_range_weight_raises(self):
        current = {"N": 1.0, "E": 1.0, "S": 1.0, "W": 1.0}
        with pytest.raises(ValueError):
            fuse_demand(current, current, prediction_weight=1.5)
        with pytest.raises(ValueError):
            fuse_demand(current, current, prediction_weight=-0.1)

    def test_fused_output_still_allocates_validly(self):
        # End-to-end: fused demand should still feed allocate_cycle cleanly.
        current = {"N": 30.0, "E": 5.0, "S": 12.0, "W": 8.0}
        predicted = {"N": 10.0, "E": 25.0, "S": 5.0, "W": 15.0}
        fused = fuse_demand(current, predicted, prediction_weight=0.5)
        allocation = allocate_cycle(fused)
        assert sum(allocation.values()) == TOTAL_CYCLE_S
        assert all(MIN_GREEN_S <= v <= MAX_GREEN_S for v in allocation.values())


class TestComputeIsDemoMode:
    """Pure function — no pipeline/DeepSort construction needed."""

    def test_four_distinct_sources_is_not_demo_mode(self):
        sources = {"N": "n.mp4", "E": "e.mp4", "S": "s.mp4", "W": "w.mp4"}
        assert compute_is_demo_mode(sources) is False

    def test_one_shared_source_is_demo_mode(self):
        sources = {a: "shared.mp4" for a in ["N", "E", "S", "W"]}
        assert compute_is_demo_mode(sources) is True

    def test_partial_overlap_still_counts_as_distinct(self):
        # Three distinct paths (one pair shares a file) — still not the
        # single-video quadrant-split case, so not demo mode.
        sources = {"N": "a.mp4", "E": "a.mp4", "S": "b.mp4", "W": "c.mp4"}
        assert compute_is_demo_mode(sources) is False
