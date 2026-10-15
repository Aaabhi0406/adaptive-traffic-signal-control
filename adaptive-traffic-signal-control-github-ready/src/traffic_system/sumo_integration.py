"""
sumo_integration.py — Real SUMO integration for traffic-signal experiments.

This module provides a microscopic SUMO/TraCI environment for comparing
traffic-signal control strategies on the project's synthetic 4-arm junction.

Research benchmark phase structure:
    NS green -> NS yellow -> EW green -> EW yellow

Where:
    NS = North + South simultaneously
    EW = East + West simultaneously

The generated benchmark demand contains one straight-through flow per
approach. Therefore, left-turn movements are kept red in the benchmark
signal states.

The old four-approach run_cycle() method is retained for backwards
compatibility with earlier project code.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .intersection import (
    APPROACHES,
    MIN_GREEN_S,
    TOTAL_CYCLE_S,
    YELLOW_S,
    allocate_cycle,
)


SUMO_ENV_DIR = Path(__file__).resolve().parent.parent.parent / "sumo_env"
NET_FILE = SUMO_ENV_DIR / "intersection.net.xml"


# ---------------------------------------------------------------------------
# SUMO network mapping
# ---------------------------------------------------------------------------

APPROACH_EDGES = {
    "N": "B2A1",
    "E": "B1A1",
    "S": "B4A1",
    "W": "B3A1",
}

APPROACH_LINK_INDICES = {
    "N": (0, 1, 2),
    "E": (3, 4, 5),
    "S": (6, 7, 8),
    "W": (9, 10, 11),
}

N_LINKS = 12


# ---------------------------------------------------------------------------
# Signal-state construction
# ---------------------------------------------------------------------------

def _phase_state(
    green_approach: str | None,
    yellow_approach: str | None = None,
) -> str:
    """
    Build the SUMO traffic-light state.

    Supported groups:
        NS -> North + South
        EW -> East + West

    Single approaches N/E/S/W remain supported for backwards compatibility.

    The benchmark demand is straight-through only, so the left-turn
    connection remains red.
    """
    if green_approach is not None and yellow_approach is not None:
        raise ValueError(
            "green_approach and yellow_approach cannot both be specified"
        )

    target = green_approach if green_approach is not None else yellow_approach

    if target is None:
        return "r" * N_LINKS

    if target == "NS":
        targets = ("N", "S")
    elif target == "EW":
        targets = ("E", "W")
    elif target in APPROACHES:
        targets = (target,)
    else:
        raise ValueError(f"Unknown approach/group: {target}")

    state = ["r"] * N_LINKS
    is_yellow = yellow_approach is not None

    for approach in targets:
        right_i, straight_i, left_i = APPROACH_LINK_INDICES[approach]

        movement_char = "y" if is_yellow else "G"

        state[right_i] = movement_char
        state[straight_i] = movement_char

        # Left turns remain red because the benchmark uses straight-through
        # demand only and we do not model protected/permitted turn phases.
        state[left_i] = "r"

    return "".join(state)


# ---------------------------------------------------------------------------
# Route-file generation
# ---------------------------------------------------------------------------

def build_route_file(
    arrival_rates: dict[str, float],
    path: Path,
    sim_duration_s: int,
    seed: int = 0,
) -> None:
    """
    Write a SUMO route file with one straight-through flow per approach.

    arrival_rates are specified in vehicles/minute.
    SUMO receives the equivalent vehicles/hour value.
    """
    del seed  # SUMO receives the seed when the environment starts.

    for approach in APPROACHES:
        if approach not in arrival_rates:
            raise ValueError(
                f"Missing arrival rate for approach {approach}"
            )

        if arrival_rates[approach] < 0:
            raise ValueError(
                f"Arrival rate for {approach} cannot be negative"
            )

    destination_edge = {
        "N": "A1B4",
        "E": "A1B3",
        "S": "A1B2",
        "W": "A1B1",
    }

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        '  <vType id="car" accel="2.6" decel="4.5" length="4.5" maxSpeed="13.9"/>',
    ]

    for approach in APPROACHES:
        lines.append(
            f'  <route id="route_{approach}" '
            f'edges="{APPROACH_EDGES[approach]} {destination_edge[approach]}"/>'
        )

    for approach in APPROACHES:
        vehicles_per_hour = arrival_rates[approach] * 60.0

        lines.append(
            f'  <flow id="flow_{approach}" '
            f'type="car" '
            f'route="route_{approach}" '
            f'begin="0" '
            f'end="{sim_duration_s}" '
            f'vehsPerHour="{vehicles_per_hour:.1f}" '
            f'departLane="best" '
            f'departSpeed="max"/>'
        )

    lines.append("</routes>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Reading benchmark demand
# ---------------------------------------------------------------------------

def read_arrival_rates_from_route_file(
    route_file: Path,
) -> dict[str, float]:
    """
    Read SUMO flow rates from a generated route file.

    Returns rates in vehicles/hour, which are the units required by
    classical Webster calculations.
    """
    root = ET.parse(route_file).getroot()
    rates: dict[str, float] = {}

    for approach in APPROACHES:
        flow_id = f"flow_{approach}"
        element = root.find(f".//flow[@id='{flow_id}']")

        if element is None:
            raise ValueError(
                f"Could not find {flow_id} in route file: {route_file}"
            )

        value = element.get("vehsPerHour")

        if value is None:
            raise ValueError(
                f"{flow_id} has no vehsPerHour attribute"
            )

        rates[approach] = float(value)

    return rates


# ---------------------------------------------------------------------------
# Classical Webster signal timing
# ---------------------------------------------------------------------------

def classical_webster_split(
    flows: dict[str, float],
    cycle_s: float = 60.0,
    lost_time_s: float = 6.0,
    saturation_flow_vph: float = 1800.0,
    min_green_s: float = 10.0,
    max_green_s: float | None = None,
) -> dict[str, float]:
    """
    Compute a classical Webster-style NS/EW green split.

    NS = North + South
    EW = East + West

    flows are vehicles/hour.

    Webster optimum cycle:
        C = (1.5L + 5) / (1 - Y)

    where:
        Y = y_NS + y_EW
        y_i = q_i / s

    The benchmark uses Webster only for feasible demand (Y < 0.95).
    Near-saturated and oversaturated demand is rejected because the classical
    Webster optimum is not mathematically well-defined there.
    """

    ns_flow = max(
        0.0,
        flows.get("N", 0.0) + flows.get("S", 0.0),
    )

    ew_flow = max(
        0.0,
        flows.get("E", 0.0) + flows.get("W", 0.0),
    )

    y_ns = ns_flow / saturation_flow_vph
    y_ew = ew_flow / saturation_flow_vph
    Y = y_ns + y_ew

    # No demand: use the requested fallback cycle and split equally.
    if Y <= 1e-9:
        effective_green = max(0.0, cycle_s - lost_time_s)
        green = max(min_green_s, min(max_green_s, effective_green / 2.0))

        return {
            "NS": green,
            "EW": green,
        }

    if Y >= 0.95:
        raise ValueError(
            f"Webster baseline requires feasible demand with Y < 0.95; "
            f"received Y={Y:.3f}. Use a lower arrival rate or treat the "
            f"scenario as an oversaturated stress test without Webster."
        )

    if Y < 1.0:
        webster_cycle = (
            1.5 * lost_time_s + 5.0
        ) / (1.0 - Y)

        cycle = max(40.0, webster_cycle)
    effective_green = max(0.0, cycle - lost_time_s)

    # First compute the unconstrained Webster split.
    ns_green = effective_green * (y_ns / Y)
    ew_green = effective_green * (y_ew / Y)

    # Apply only a practical minimum green, then preserve the total effective
    # green exactly.  A hard maximum is intentionally not imposed here: an
    # arbitrary max-green cap would no longer be the classical Webster split.
    # If a maximum is explicitly requested, redistribute the excess to the
    # other phase while preserving the effective-green total.
    if ns_green < min_green_s:
        ns_green = min_green_s
        ew_green = effective_green - ns_green
    elif ew_green < min_green_s:
        ew_green = min_green_s
        ns_green = effective_green - ew_green

    if max_green_s is not None:
        if ns_green > max_green_s:
            ns_green = max_green_s
            ew_green = effective_green - ns_green
        elif ew_green > max_green_s:
            ew_green = max_green_s
            ns_green = effective_green - ew_green

    if ns_green < min_green_s or ew_green < min_green_s:
        raise ValueError(
            "Webster green bounds are infeasible for the computed effective "
            f"green time ({effective_green:.2f}s): "
            f"NS={ns_green:.2f}s, EW={ew_green:.2f}s."
        )

    return {
        "NS": ns_green,
        "EW": ew_green,
    }


# ---------------------------------------------------------------------------
# SUMO / TraCI environment
# ---------------------------------------------------------------------------

class SumoTraciEnv:
    """
    Owns one TraCI connection to a running SUMO process.

    Provides:
        - approach queue measurements
        - legacy four-approach phases
        - compatible NS/EW phases
        - SUMO waiting-time measurements
    """

    def __init__(
        self,
        route_file: Path,
        gui: bool = False,
        seed: int = 0,
    ):
        self.route_file = route_file
        self.gui = gui
        self.seed = seed
        self._started = False
        self._waiting_time_total = 0.0

    def start(self) -> None:
        """Start SUMO through TraCI."""
        import traci

        binary = "sumo-gui" if self.gui else "sumo"

        self._waiting_time_total = 0.0

        traci.start(
            [
                binary,
                "-n",
                str(NET_FILE),
                "-r",
                str(self.route_file),
                "--no-step-log",
                "true",
                "--waiting-time-memory",
                "10000",
                "--step-length",
                "1.0",
                "--seed",
                str(self.seed),
                "--quit-on-end",
                "true",
            ]
        )

        self._started = True

    def close(self) -> None:
        """Close the active TraCI connection."""
        import traci

        if self._started:
            traci.close()
            self._started = False

    def get_queue_lengths(self) -> dict[str, float]:
        """Return SUMO's current halting-vehicle count per approach."""
        import traci

        return {
            approach: float(
                traci.edge.getLastStepHaltingNumber(
                    APPROACH_EDGES[approach]
                )
            )
            for approach in APPROACHES
        }

    def _accumulate_waiting_time(self) -> None:
        """Accumulate instantaneous edge waiting time over one simulation step."""
        import traci

        current_waiting = sum(
            float(traci.edge.getWaitingTime(APPROACH_EDGES[approach]))
            for approach in APPROACHES
        )
        # SUMO is configured with a 1-second simulation step.
        self._waiting_time_total += current_waiting

    def _run_phase(
        self,
        approach: str | None,
        duration_s: float,
        yellow: bool,
    ) -> None:
        """Apply a signal state and advance SUMO for duration_s steps."""
        import traci

        duration = max(1, int(round(duration_s)))

        if yellow:
            state = _phase_state(
                green_approach=None,
                yellow_approach=approach,
            )
        else:
            state = _phase_state(
                green_approach=approach,
            )

        traci.trafficlight.setRedYellowGreenState(
            "A1",
            state,
        )

        for _ in range(duration):
            traci.simulationStep()
            self._accumulate_waiting_time()

    def run_cycle(
        self,
        allocation: dict[str, int],
    ) -> None:
        """
        Legacy four-approach cycle:

            N -> E -> S -> W

        Retained for backwards compatibility.
        """
        for approach in APPROACHES:
            green_s = max(
                1,
                int(allocation[approach]),
            )

            self._run_phase(
                approach,
                green_s,
                yellow=False,
            )

            self._run_phase(
                approach,
                YELLOW_S,
                yellow=True,
            )

    def run_compatible_cycle(
        self,
        allocation: dict[str, float],
    ) -> None:
        """
        Run one conflict-safe two-group cycle:

            NS green -> NS yellow -> EW green -> EW yellow
        """

        ns_green_s = max(
            1.0,
            float(allocation["NS"]),
        )

        ew_green_s = max(
            1.0,
            float(allocation["EW"]),
        )

        # North + South green simultaneously.
        self._run_phase(
            "NS",
            ns_green_s,
            yellow=False,
        )

        self._run_phase(
            "NS",
            YELLOW_S,
            yellow=True,
        )

        # East + West green simultaneously.
        self._run_phase(
            "EW",
            ew_green_s,
            yellow=False,
        )

        self._run_phase(
            "EW",
            YELLOW_S,
            yellow=True,
        )

    def run_compatible_duration(
        self,
        allocation: dict[str, float],
        duration_s: float,
    ) -> None:
        """Run the conflict-safe NS/EW controller for an exact integer horizon."""
        # SUMO advances in discrete simulation steps.  Keep the benchmark
        # horizon integer so the loop can never get stuck on a fractional
        # remainder.
        remaining = max(0, int(round(duration_s)))

        phases = (
            ("NS", False, max(1, int(round(float(allocation["NS"]))))),
            ("NS", True, max(1, int(round(float(YELLOW_S))))),
            ("EW", False, max(1, int(round(float(allocation["EW"]))))),
            ("EW", True, max(1, int(round(float(YELLOW_S))))),
        )

        while remaining > 0:
            for approach, yellow, phase_duration in phases:
                if remaining <= 0:
                    break

                run_for = min(remaining, phase_duration)
                self._run_phase(approach, run_for, yellow=yellow)
                remaining -= run_for

                # The final phase can be truncated to end exactly at the
                # requested common simulation horizon.
                if run_for < phase_duration:
                    return

    def run_reactive_duration(
        self,
        duration_s: int,
        min_green_s: float = 10.0,
        max_green_s: float = 60.0,
        switch_margin: float = 0.15,
        starvation_threshold_s: float = 90.0,
        ema_alpha: float = 0.3,
        initial_group: str = "NS",
    ) -> dict:
        """Run the conflict-safe reactive controller for an exact horizon.

        Demand is measured directly from SUMO's current halting-vehicle counts
        on the four approach edges. This is a controller baseline: it tests the
        adaptive signal logic using current observed demand, without using the
        project's YOLO/RF vision pipeline.

        Returns summary statistics useful for benchmarking.
        """
        from .signal_controller import ConflictSafeSignalController, SIGNAL_STATE_MAP

        duration = max(0, int(round(duration_s)))
        controller = ConflictSafeSignalController(
            min_green_s=min_green_s,
            max_green_s=max_green_s,
            switch_margin=switch_margin,
            starvation_threshold_s=starvation_threshold_s,
            ema_alpha=ema_alpha,
            initial_group=initial_group,
            init_now=0.0,
        )

        import traci

        last_state = None
        switches = 0
        state_seconds = {state: 0 for state in SIGNAL_STATE_MAP}

        for sim_t in range(duration):
            queues = self.get_queue_lengths()
            ns_demand = queues["N"] + queues["S"]
            ew_demand = queues["E"] + queues["W"]
            snapshot = controller.update(float(ns_demand), float(ew_demand), now=float(sim_t))
            state = snapshot["signal_state"]

            if state != last_state:
                if last_state is not None and snapshot["decision"] == "SWITCH":
                    switches += 1
                last_state = state

            # Use the same compact signal encoding as the benchmark.
            traci.trafficlight.setRedYellowGreenState(
                "A1", _phase_state_for_controller_state(state)
            )
            traci.simulationStep()
            self._accumulate_waiting_time()
            state_seconds[state] += 1

        return {
            "waiting_time": self.total_waiting_time(),
            "switches": switches,
            "state_seconds": state_seconds,
        }

    def total_waiting_time(self) -> float:
        """Return accumulated vehicle waiting time in vehicle-seconds."""
        return float(self._waiting_time_total)


def _phase_state_for_controller_state(state: str) -> str:
    """Translate controller state names into this SUMO network's signal state."""
    if state == "NS_GREEN":
        return _phase_state("NS")
    if state == "NS_YELLOW":
        return _phase_state(None, "NS")
    if state == "EW_GREEN":
        return _phase_state("EW")
    if state == "EW_YELLOW":
        return _phase_state(None, "EW")
    if state == "ALL_RED":
        return _phase_state(None, None)
    raise ValueError(f"Unknown controller state: {state}")



# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_fixed_timer(
    route_file: Path,
    n_cycles: int,
    fixed_green_s: int = 25,
    seed: int = 0,
) -> float:
    """
    Run a fixed-time NS/EW controller.

    Both compatible groups receive the same fixed green duration.
    """

    env = SumoTraciEnv(
        route_file,
        seed=seed,
    )

    env.start()

    try:
        allocation = {
            "NS": float(fixed_green_s),
            "EW": float(fixed_green_s),
        }

        for _ in range(n_cycles):
            env.run_compatible_cycle(allocation)

        return env.total_waiting_time()

    finally:
        env.close()


def run_webster(
    route_file: Path,
    n_cycles: int,
    seed: int = 0,
) -> float:
    """
    Run the classical Webster baseline.

    Webster uses the known arrival flow rates encoded in the SUMO route file.
    It does not substitute queue length for flow rate.

    The same NS/EW phase architecture is used as the fixed-time baseline.
    """

    flows_vph = read_arrival_rates_from_route_file(
        route_file
    )

    allocation = classical_webster_split(
        flows_vph
    )

    env = SumoTraciEnv(
        route_file,
        seed=seed,
    )

    env.start()

    try:
        for _ in range(n_cycles):
            env.run_compatible_cycle(allocation)

        return env.total_waiting_time()

    finally:
        env.close()
