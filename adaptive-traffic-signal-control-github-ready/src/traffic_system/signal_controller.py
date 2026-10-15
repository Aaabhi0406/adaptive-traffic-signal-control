"""
signal_controller.py — real-time, demand-driven, conflict-safe signal
state machine.

WHAT THIS REPLACES CONCEPTUALLY: allocate_cycle() (intersection.py)
produces four independent green-second numbers (e.g. N=45, E=15, S=15,
W=45). Those are a DEMAND ALLOCATION, not a physically executable signal
plan — a real intersection cannot have all four approaches green at once.
This module is the piece that turns fused demand into an actual
conflict-safe signal state over time.

CONFLICT MODEL (current prototype — document this, don't overclaim):
  Only two compatible movement groups exist:
    NS  — North and South green together
    EW  — East and West green together
  N+E, N+W, S+E, S+W are never simultaneously green. This is a simple
  four-approach straight-through model, NOT a lane-level turning-movement
  controller (no protected left turns, no pedestrian phases, no conflicting
  through/turn combinations within a group).

STATE MACHINE (safety states, not a fixed repeating schedule):
    NS_GREEN -> NS_YELLOW -> ALL_RED -> EW_GREEN -> EW_YELLOW -> ALL_RED -> NS_GREEN -> ...
  Which transition happens next, and when, is decided every update() call
  from live (smoothed) demand — not a fixed timer. The only "fixed" timing
  is the safety intervals themselves (yellow clearance, all-red clearance,
  minimum green) — see the constants below.

DECISION FACTORS per update(), in the order they're actually checked:
  1. Minimum green not yet satisfied -> KEEP, full stop, no other factor
     can override a safety-minimum green.
  2. Maximum green reached -> FORCE SWITCH, regardless of demand
     (starvation-of-the-competing-side prevention via a hard ceiling).
  3. Competing side has been waiting >= starvation_threshold_s -> SWITCH
     (starvation-of-the-competing-side prevention via a soft trigger,
     kicks in before factor 2 would force it).
  4. Competing (EMA-smoothed) demand exceeds current demand by more than
     switch_margin -> SWITCH. The EMA smoothing + margin together are the
     anti-flicker/hysteresis mechanism — a single noisy frame can't flip
     the signal.
  5. Otherwise -> KEEP/EXTEND (current side still has sufficient demand).

RESEARCH / SAFETY HONESTY: this is a simulation/research-prototype
controller. ALL_RED_S below is an illustrative prototype value, not an
engineering-validated clearance interval — it (and every other timing
constant here) would need validation against local traffic-signal
engineering standards, real intersection geometry, and pedestrian-phase
requirements before anything resembling physical deployment. This module
does not implement, and does not claim to implement: pedestrian phases,
protected turning movements, emergency-vehicle preemption, fault
detection, controller redundancy, or hardware-in-the-loop safety
validation.

SUMO COMPATIBILITY: deliberately framework-agnostic — update() takes
plain floats and an optional explicit timestamp, and returns a plain
dict. Nothing here imports Streamlit or SUMO. The `signal_state` /
`signal_states` output uses the same N/E/S/W keys and RED/YELLOW/GREEN
vocabulary that sumo_integration.py's _phase_state() already expects, so
a future integration can consume this controller's decisions directly —
that wiring is NOT implemented here (see docs/STATUS.md); doing so
without fabricating results is separate follow-up work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .intersection import MAX_GREEN_S, MIN_GREEN_S, YELLOW_S

# ── Prototype safety-interval constants ─────────────────────────────────────
# ALL_RED_S is a conservative prototype value for simulation only — it is
# NOT an engineering-validated clearance interval. Validate against local
# traffic-signal standards and real intersection geometry before any use
# beyond simulation/research.
ALL_RED_S = 2.0

# Anti-flicker / hysteresis: only switch when the competing side's demand
# exceeds the current side's by more than this fraction. Prevents the
# signal reacting to a single noisy frame's demand estimate.
SWITCH_MARGIN = 0.15

# If a group has been waiting this long, it gets switched to regardless of
# the current side's demand — prevents indefinite starvation of the
# lower-demand direction.
STARVATION_THRESHOLD_S = 90.0

# Exponential-moving-average smoothing factor applied to demand before any
# switching decision is made (same purpose as SWITCH_MARGIN: don't react
# to one noisy tick).
DEMAND_EMA_ALPHA = 0.3

# ── States ───────────────────────────────────────────────────────────────
STATE_NS_GREEN = "NS_GREEN"
STATE_NS_YELLOW = "NS_YELLOW"
STATE_ALL_RED = "ALL_RED"
STATE_EW_GREEN = "EW_GREEN"
STATE_EW_YELLOW = "EW_YELLOW"

GREEN_STATES = {STATE_NS_GREEN: "NS", STATE_EW_GREEN: "EW"}
YELLOW_STATES = {STATE_NS_YELLOW: "NS", STATE_EW_YELLOW: "EW"}

SIGNAL_STATE_MAP = {
    STATE_NS_GREEN:  {"N": "GREEN",  "S": "GREEN",  "E": "RED",    "W": "RED"},
    STATE_NS_YELLOW: {"N": "YELLOW", "S": "YELLOW", "E": "RED",    "W": "RED"},
    STATE_ALL_RED:   {"N": "RED",    "S": "RED",    "E": "RED",    "W": "RED"},
    STATE_EW_GREEN:  {"N": "RED",    "S": "RED",    "E": "GREEN",  "W": "GREEN"},
    STATE_EW_YELLOW: {"N": "RED",    "S": "RED",    "E": "YELLOW", "W": "YELLOW"},
}


@dataclass
class ConflictSafeSignalController:
    """Continuously-evaluated demand-driven state machine. Call update()
    once per control tick with the current NS/EW fused demand; it returns
    a snapshot dict describing the live signal state and why.

    Deterministic for testing: pass an explicit `now` to update() (and/or
    `init_now` at construction) instead of relying on wall-clock time.
    """

    min_green_s: float = MIN_GREEN_S
    max_green_s: float = MAX_GREEN_S
    yellow_s: float = YELLOW_S
    all_red_s: float = ALL_RED_S
    switch_margin: float = SWITCH_MARGIN
    starvation_threshold_s: float = STARVATION_THRESHOLD_S
    ema_alpha: float = DEMAND_EMA_ALPHA
    initial_group: str = "NS"
    init_now: float | None = None  # for deterministic tests; None -> time.monotonic()

    state: str = field(init=False)
    state_start_time: float = field(init=False)
    green_start_time: float = field(init=False)
    next_group: str | None = field(init=False, default=None)
    smoothed_demand: dict[str, float] = field(init=False)
    last_green_end_time: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        if self.initial_group not in ("NS", "EW"):
            raise ValueError(f"initial_group must be 'NS' or 'EW', got {self.initial_group!r}")
        now = self.init_now if self.init_now is not None else time.monotonic()
        self.state = STATE_NS_GREEN if self.initial_group == "NS" else STATE_EW_GREEN
        self.state_start_time = now
        self.green_start_time = now
        self.smoothed_demand = {"NS": 0.0, "EW": 0.0}
        # Neither group has "waited" yet at t=0 — anchor both wait-clocks to
        # controller start so starvation timing is well-defined from tick one.
        self.last_green_end_time = {"NS": now, "EW": now}

    def update(self, ns_demand: float, ew_demand: float, now: float | None = None) -> dict:
        """Advance the state machine by one control tick and return a
        snapshot. `ns_demand`/`ew_demand` should be the fused (current +
        uncertainty-hedged-prediction) demand summed across each
        compatible pair — see intersection.py's process_tick for how
        these are computed from the existing fused_demand dict."""
        now = time.monotonic() if now is None else now
        a = self.ema_alpha
        self.smoothed_demand["NS"] = a * ns_demand + (1 - a) * self.smoothed_demand["NS"]
        self.smoothed_demand["EW"] = a * ew_demand + (1 - a) * self.smoothed_demand["EW"]

        if self.state in YELLOW_STATES:
            decision, reason = self._update_yellow(now)
        elif self.state == STATE_ALL_RED:
            decision, reason = self._update_all_red(now)
        else:
            decision, reason = self._update_green(now)

        return self._snapshot(now, decision, reason)

    def _update_yellow(self, now: float) -> tuple[str, str]:
        group = YELLOW_STATES[self.state]
        elapsed = now - self.state_start_time
        if elapsed >= self.yellow_s:
            self.state = STATE_ALL_RED
            self.state_start_time = now
            return "TRANSITION", f"ALL-RED safety interval — {group} yellow clearance complete"
        return "TRANSITION", f"{group} YELLOW — clearing ({elapsed:.1f}s/{self.yellow_s}s)"

    def _update_all_red(self, now: float) -> tuple[str, str]:
        elapsed = now - self.state_start_time
        if elapsed >= self.all_red_s:
            new_group = self.next_group
            self.state = STATE_NS_GREEN if new_group == "NS" else STATE_EW_GREEN
            self.state_start_time = now
            self.green_start_time = now
            self.next_group = None
            return "SWITCH", f"{new_group} GREEN — all-red safety interval complete, {new_group} now active"
        return "TRANSITION", f"ALL-RED safety interval — clearing ({elapsed:.1f}s/{self.all_red_s}s)"

    def _update_green(self, now: float) -> tuple[str, str]:
        current_group = GREEN_STATES[self.state]
        competing_group = "EW" if current_group == "NS" else "NS"
        green_elapsed = now - self.green_start_time
        current_demand = self.smoothed_demand[current_group]
        competing_demand = self.smoothed_demand[competing_group]
        waiting_time = now - self.last_green_end_time[competing_group]

        if green_elapsed < self.min_green_s:
            return "KEEP", (
                f"KEEP {current_group} GREEN — minimum green not yet satisfied "
                f"({green_elapsed:.1f}s/{self.min_green_s}s)"
            )

        if green_elapsed >= self.max_green_s:
            return self._begin_switch(
                current_group, competing_group, now,
                f"FORCE SWITCH — {current_group} reached maximum green ({self.max_green_s}s)",
            )

        if waiting_time >= self.starvation_threshold_s and competing_demand > 0:
            return self._begin_switch(
                current_group, competing_group, now,
                f"SWITCH {current_group} -> {competing_group} — starvation prevention "
                f"({competing_group} waited {waiting_time:.1f}s)",
            )

        if competing_demand > current_demand * (1 + self.switch_margin):
            return self._begin_switch(
                current_group, competing_group, now,
                f"SWITCH {current_group} -> {competing_group} — competing demand "
                f"{competing_demand:.1f} exceeds {current_group} demand {current_demand:.1f} "
                f"by more than {self.switch_margin * 100:.0f}%",
            )

        return "EXTEND", (
            f"KEEP {current_group} GREEN — {current_group} demand ({current_demand:.1f}) "
            f"remains sufficient vs {competing_group} ({competing_demand:.1f})"
        )

    def _begin_switch(self, current_group: str, competing_group: str, now: float, reason: str) -> tuple[str, str]:
        self.last_green_end_time[current_group] = now
        self.next_group = competing_group
        self.state = STATE_NS_YELLOW if current_group == "NS" else STATE_EW_YELLOW
        self.state_start_time = now
        return "SWITCH", reason

    def _priority(self, group: str, now: float) -> float:
        """Simple, documented priority score — NOT a claim of a
        sophisticated scheduling algorithm: demand scaled up by how long
        the group has been waiting, saturating at 2x demand once the wait
        reaches starvation_threshold_s. Exists so the dashboard can show a
        single "who's more urgent" number that reflects both demand and
        fairness, matching the decision logic above rather than
        contradicting it."""
        if self.state in GREEN_STATES and GREEN_STATES[self.state] == group:
            wait = 0.0
        else:
            wait = max(0.0, now - self.last_green_end_time[group])
        wait_factor = min(1.0, wait / self.starvation_threshold_s) if self.starvation_threshold_s > 0 else 0.0
        return self.smoothed_demand[group] * (1.0 + wait_factor)

    def _snapshot(self, now: float, decision: str, reason: str) -> dict:
        signal_states = dict(SIGNAL_STATE_MAP[self.state])
        is_green = self.state in GREEN_STATES
        green_elapsed = (now - self.green_start_time) if is_green else 0.0
        green_remaining = max(0.0, self.max_green_s - green_elapsed) if is_green else 0.0
        active_group = GREEN_STATES.get(self.state) or YELLOW_STATES.get(self.state) or self.next_group

        return {
            "signal_state": self.state,
            "active_group": active_group,
            "signal_states": signal_states,
            "green_elapsed_s": round(green_elapsed, 1),
            "green_remaining_s": round(green_remaining, 1),
            "ns_demand": round(self.smoothed_demand["NS"], 2),
            "ew_demand": round(self.smoothed_demand["EW"], 2),
            "ns_priority": round(self._priority("NS", now), 2),
            "ew_priority": round(self._priority("EW", now), 2),
            "decision": decision,
            "decision_reason": reason,
        }