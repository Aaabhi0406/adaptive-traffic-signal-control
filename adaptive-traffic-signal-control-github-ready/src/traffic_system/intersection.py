"""
intersection.py — 4-way signal coordination (Phase 2).

Everything up to this file treats one video as one approach with its own
independent green_time recommendation. A real intersection has 4 competing
approaches sharing a single cycle: giving each one its own uncoordinated
green_time (as the single-signal pipeline does) doesn't mean anything until
something reconciles all 4 into one cycle that actually sums to something
drivable.

TWO WAYS TO GET 4 APPROACHES:
  1. Four separate camera feeds, one per approach — pass four video paths to
     FourWayIntersection and each gets its own full-frame TrafficPipeline.
  2. One intersection-overhead camera — QUADRANT_ROIS splits a single frame
     into four trapezoid ROIs (N/E/S/W) so one video can still drive a
     4-signal demo. This is the fallback used when you only have the two
     bundled single-approach clips: same frame processed 4 times, each with
     the corresponding quadrant as the "ROI".

ALLOCATION — extended Webster's formula, not just "biggest queue wins":
  Given 4 approach weighted-densities, split a fixed total cycle length
  proportionally to each approach's share of total demand, but never let an
  approach starve below MIN_GREEN_S or one hog above MAX_GREEN_S. This is the
  classical baseline (Webster 1958) that any RL/optimisation approach
  added later (Phase 6) should be benchmarked against — if a learned
  controller can't beat this, it isn't earning its complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config
from .detection import get_model
from .pipeline import TrafficPipeline


APPROACHES = ("N", "E", "S", "W")


# ── Cycle allocation constants ──────────────────────────────────────────────

TOTAL_CYCLE_S = 120
MIN_GREEN_S = 10
MAX_GREEN_S = 60
YELLOW_S = 3


def build_quadrant_roi(
    frame_h: int,
    frame_w: int,
    approach: str,
) -> np.ndarray:
    """ROI trapezoid for one quadrant of a single overhead/4-way frame.

    Splits the frame into four overlapping-at-centre wedges rather than a
    strict grid quadrant, since a real 4-way approach usually occupies a
    triangular wedge pointing at the intersection centre, not a rectangle.

    Falls back to the existing full-road trapezoid shape (see pipeline.
    build_roi) scaled into the approach's half of the frame.
    """

    if approach not in APPROACHES:
        raise ValueError(
            f"Unknown approach {approach!r}, expected one of {APPROACHES}"
        )

    cx, cy = frame_w / 2, frame_h / 2

    margin_x = frame_w * config.ROI_TOP_LEFT_X_FRAC
    margin_y = frame_h * config.ROI_TOP_Y_FRAC

    corners = {
        "N": [
            (margin_x, margin_y),
            (frame_w - margin_x, margin_y),
            (cx + 40, cy),
            (cx - 40, cy),
        ],
        "S": [
            (cx - 40, cy),
            (cx + 40, cy),
            (frame_w - margin_x, frame_h - margin_y),
            (margin_x, frame_h - margin_y),
        ],
        "E": [
            (cx, cy - 40),
            (frame_w - margin_x, margin_y),
            (frame_w - margin_x, frame_h - margin_y),
            (cx, cy + 40),
        ],
        "W": [
            (margin_x, margin_y),
            (cx, cy - 40),
            (cx, cy + 40),
            (margin_x, frame_h - margin_y),
        ],
    }

    return np.array(
        corners[approach],
        dtype=np.int32,
    )


def _apportion(
    shares: dict[str, float],
    total: int,
) -> dict[str, int]:
    """Largest-remainder (Hamilton) apportionment.

    Split an integer `total` across approaches proportionally to `shares`.
    The result sums to exactly `total`.
    """

    share_sum = sum(shares.values())

    normalised = (
        {
            a: v / share_sum
            for a, v in shares.items()
        }
        if share_sum > 0
        else {
            a: 1.0 / len(shares)
            for a in shares
        }
    )

    raw = {
        a: normalised[a] * total
        for a in shares
    }

    floor_vals = {
        a: int(raw[a])
        for a in shares
    }

    remainder = {
        a: raw[a] - floor_vals[a]
        for a in shares
    }

    leftover = total - sum(floor_vals.values())

    order = sorted(
        shares.keys(),
        key=lambda a: remainder[a],
        reverse=True,
    )

    allocation = dict(floor_vals)

    for a in order[:leftover]:
        allocation[a] += 1

    return allocation


def allocate_cycle(
    weighted_densities: dict[str, float],
    total_cycle_s: int = TOTAL_CYCLE_S,
    min_green_s: int = MIN_GREEN_S,
    max_green_s: int = MAX_GREEN_S,
) -> dict[str, int]:
    """Extended Webster-style proportional cycle allocation.

    Split total_cycle_s across approaches proportionally to each one's share
    of total weighted density, then enforce [min_green_s, max_green_s].

    Pure function — no I/O and no state.
    """

    approaches = list(weighted_densities.keys())

    if total_cycle_s < min_green_s * len(approaches):
        raise ValueError(
            f"total_cycle_s={total_cycle_s} can't give every approach "
            f"the {min_green_s}s floor across {len(approaches)} approaches"
        )

    free_shares = dict(weighted_densities)
    fixed: dict[str, int] = {}

    for _ in range(len(approaches)):

        remaining_total = (
            total_cycle_s
            - sum(fixed.values())
        )

        if not free_shares or remaining_total <= 0:
            break

        free_allocation = _apportion(
            free_shares,
            remaining_total,
        )

        newly_fixed = {
            a: (
                min_green_s
                if v < min_green_s
                else max_green_s
            )
            for a, v in free_allocation.items()
            if v < min_green_s or v > max_green_s
        }

        if not newly_fixed:
            return {
                **fixed,
                **free_allocation,
            }

        fixed.update(newly_fixed)

        for a in newly_fixed:
            del free_shares[a]

    # Extreme-skew fallback.
    for a in free_shares:
        fixed.setdefault(
            a,
            min_green_s,
        )

    return fixed


def fuse_demand(
    weighted_densities: dict[str, float],
    hedged_pred_densities: dict[str, float],
    prediction_weight: float = 0.5,
) -> dict[str, float]:
    """Combine current demand and uncertainty-hedged prediction."""

    if not 0.0 <= prediction_weight <= 1.0:
        raise ValueError(
            f"prediction_weight must be in [0, 1], "
            f"got {prediction_weight}"
        )

    return {
        a: (
            (1 - prediction_weight)
            * weighted_densities[a]
            + prediction_weight
            * hedged_pred_densities[a]
        )
        for a in weighted_densities
    }


def compute_is_demo_mode(
    sources: dict[str, str],
) -> bool:
    """Return True when the same video is reused for all approaches."""

    return len(set(sources.values())) <= 1


@dataclass
class FourWayIntersection:
    """Owns one TrafficPipeline per approach and coordinates their demand.

    `sources` maps approach -> video path.

    The primary research architecture is four independent camera feeds.

    If the same path is passed for all four approaches, this falls back to
    quadrant-splitting a single video for demonstration purposes.
    """

    sources: dict[str, str]

    model_path: str = config.DEFAULT_MODEL_PATH

    prediction_weight: float = 0.5

    pipelines: dict[str, TrafficPipeline] = field(
        init=False
    )

    caps: dict[str, "cv2.VideoCapture"] = field(
        init=False,
        default_factory=dict,
    )

    is_demo_mode: bool = field(
        init=False
    )

    signal_controller: "ConflictSafeSignalController" = field(
        init=False
    )

    def __post_init__(self) -> None:

        self.is_demo_mode = compute_is_demo_mode(
            self.sources
        )

        distinct_sources = not self.is_demo_mode

        self.pipelines = {}

        for approach in APPROACHES:

            if distinct_sources:

                # Independent camera per approach -> full-road ROI.
                roi_builder = None

            else:

                # Single-video demo -> quadrant ROI.
                roi_builder = (
                    lambda h, w, a=approach:
                    build_quadrant_roi(h, w, a)
                )

            self.pipelines[approach] = TrafficPipeline(
                roi_builder=roi_builder,
                label=approach,
            )

        # Local import deliberately avoids circular import problems.
        from .signal_controller import (
            ConflictSafeSignalController
        )

        self.signal_controller = (
            ConflictSafeSignalController()
        )

    def open(self) -> None:

        for approach, path in self.sources.items():

            cap = cv2.VideoCapture(path)

            if not cap.isOpened():

                raise RuntimeError(
                    f"Cannot open video for approach "
                    f"{approach}: {path}"
                )

            self.caps[approach] = cap

    def close(self) -> None:

        for cap in self.caps.values():
            cap.release()

        self.caps = {}

    def process_tick(
        self,
        model=None,
    ) -> dict:
        """Read one frame from each approach and return coordinated results."""

        model = model or get_model(
            self.model_path
        )

        results: dict[str, dict] = {}

        for approach, cap in self.caps.items():

            ret, frame = cap.read()

            if not ret:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0,
                )

                ret, frame = cap.read()

            results[approach] = (
                self.pipelines[approach].process_frame(
                    frame,
                    model,
                )
            )

        weighted_densities = {
            a: results[a]["total_weight"]
            for a in results
        }

        # Use the upper bound of the RF prediction interval.
        hedged_pred_densities = {
            a: results[a]["pred_upper"]
            for a in results
        }

        # Current-demand allocation.
        current_allocation = allocate_cycle(
            weighted_densities
        )

        # Predictive allocation.
        predictive_allocation = allocate_cycle(
            hedged_pred_densities
        )

        # Fuse current demand and prediction.
        fused_demand = fuse_demand(
            weighted_densities,
            hedged_pred_densities,
            self.prediction_weight,
        )

        # Final demand allocation.
        final_allocation = allocate_cycle(
            fused_demand
        )

        # --------------------------------------------------------------------
        # ACTUAL SIGNAL CONTROLLER
        # --------------------------------------------------------------------
        #
        # final_allocation is NOT a physical signal plan.
        #
        # The actual controller decides between compatible movement groups:
        #
        #     NS = North + South
        #     EW = East + West
        #
        # Only one group can have green at a time.
        # --------------------------------------------------------------------

        ns_demand = (
            fused_demand["N"]
            + fused_demand["S"]
        )

        ew_demand = (
            fused_demand["E"]
            + fused_demand["W"]
        )

        signal_snapshot = (
            self.signal_controller.update(
                ns_demand,
                ew_demand,
            )
        )

        return {
            "approach_results": results,

            "weighted_densities":
                weighted_densities,

            "current_allocation":
                current_allocation,

            "predictive_allocation":
                predictive_allocation,

            "final_allocation":
                final_allocation,

            "total_cycle_s":
                TOTAL_CYCLE_S,

            "is_demo_mode":
                self.is_demo_mode,

            **signal_snapshot,
        }