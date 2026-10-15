"""
scripts/sumo_benchmark.py — fair fixed-time vs. classical Webster benchmark.

Both controllers are evaluated on the SAME SUMO demand and for the SAME
simulation duration. This avoids the previous confound where the controllers
ran for the same number of cycles but potentially different total durations.

Usage:
    python scripts/sumo_benchmark.py --duration 900 \
        --rate N=2 --rate E=10 --rate S=2 --rate W=10

The benchmark is intended for feasible demand (Webster Y < 0.95).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_system.sumo_integration import (  # noqa: E402
    APPROACHES,
    SumoTraciEnv,
    build_route_file,
    classical_webster_split,
)

SATURATION_FLOW_VPH = 1800.0


def parse_rates(overrides: list[str]) -> dict[str, float]:
    """Parse approach rates in vehicles/minute."""
    rates = {
        "N": 5.0,
        "E": 5.0,
        "S": 5.0,
        "W": 5.0,
    }

    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid --rate '{override}'. Use e.g. --rate N=5"
            )

        approach, value = override.split("=", 1)
        approach = approach.strip().upper()

        if approach not in APPROACHES:
            raise ValueError(
                f"Unknown approach '{approach}'. Use N, E, S, or W."
            )

        rate = float(value)
        if rate < 0:
            raise ValueError("Arrival rates cannot be negative.")

        rates[approach] = rate

    return rates


def run_controller(
    route_file: Path,
    allocation: dict[str, float],
    duration_s: int,
    seed: int,
) -> float:
    """Run one controller for exactly duration_s simulated seconds."""
    env = SumoTraciEnv(route_file, seed=seed)
    env.start()

    try:
        env.run_compatible_duration(
            allocation=allocation,
            duration_s=duration_s,
        )
        return env.total_waiting_time()
    finally:
        env.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fair SUMO benchmark: fixed-time vs. classical Webster "
            "with equal simulation duration."
        )
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=900,
        help="Simulation duration in seconds for BOTH controllers.",
    )
    parser.add_argument(
        "--fixed-green",
        type=int,
        default=25,
        help="Fixed green duration for NS and EW (seconds).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="SUMO random seed.",
    )
    parser.add_argument(
        "--rate",
        action="append",
        default=[],
        help="Approach arrival rate override, e.g. --rate N=2.",
    )
    args = parser.parse_args(argv)

    if args.duration <= 0:
        parser.error("--duration must be positive.")
    if args.fixed_green <= 0:
        parser.error("--fixed-green must be positive.")

    try:
        rates = parse_rates(args.rate)
    except ValueError as exc:
        parser.error(str(exc))

    # Webster uses combined opposing approaches: NS and EW.
    ns_flow_vph = (rates["N"] + rates["S"]) * 60.0
    ew_flow_vph = (rates["E"] + rates["W"]) * 60.0
    Y = (ns_flow_vph + ew_flow_vph) / SATURATION_FLOW_VPH

    if Y >= 0.95:
        parser.error(
            f"Demand is too close to saturation for the classical Webster "
            f"comparison: Y={Y:.3f}. Use lower rates (Y < 0.95)."
        )

    route_path = (
        Path("sumo_env") / f"benchmark_seed{args.seed}_{args.duration}s.rou.xml"
    )
    build_route_file(
        rates,
        route_path,
        sim_duration_s=args.duration,
        seed=args.seed,
    )

    fixed_allocation = {
        "NS": float(args.fixed_green),
        "EW": float(args.fixed_green),
    }

    # Read/derive flows directly from the same demand specification.
    flows_vph = {
        "N": rates["N"] * 60.0,
        "E": rates["E"] * 60.0,
        "S": rates["S"] * 60.0,
        "W": rates["W"] * 60.0,
    }
    webster_allocation = classical_webster_split(flows_vph)

    print("=" * 72)
    print("FAIR SUMO BENCHMARK")
    print("=" * 72)
    print(f"Arrival rates (veh/min): {rates}")
    print(f"NS flow: {ns_flow_vph:.1f} veh/h")
    print(f"EW flow: {ew_flow_vph:.1f} veh/h")
    print(f"Demand ratio Y: {Y:.3f}")
    print(f"Simulation duration: {args.duration}s")
    print(f"Random seed: {args.seed}")
    print()
    print(f"Fixed allocation:   NS={fixed_allocation['NS']:.1f}s  "
          f"EW={fixed_allocation['EW']:.1f}s")
    print(f"Webster allocation: NS={webster_allocation['NS']:.1f}s  "
          f"EW={webster_allocation['EW']:.1f}s")

    print("\n[1/2] Running fixed-time controller...")
    fixed_wait = run_controller(
        route_path,
        fixed_allocation,
        args.duration,
        args.seed,
    )
    print(f"      accumulated waiting time: {fixed_wait:.1f} vehicle-seconds")

    print("\n[2/2] Running classical Webster controller...")
    webster_wait = run_controller(
        route_path,
        webster_allocation,
        args.duration,
        args.seed,
    )
    print(
        f"      accumulated waiting time: "
        f"{webster_wait:.1f} vehicle-seconds"
    )

    improvement = (
        100.0 * (fixed_wait - webster_wait) / fixed_wait
        if fixed_wait > 0
        else 0.0
    )

    print("\n" + "=" * 72)
    print(f"Fixed-time waiting:       {fixed_wait:.1f} vehicle-seconds")
    print(f"Classical Webster waiting: {webster_wait:.1f} vehicle-seconds")
    print(f"Webster improvement:       {improvement:.2f}%")
    # Verification: recompute from the exact values used above.
    check = 100.0 * (fixed_wait - webster_wait) / fixed_wait if fixed_wait > 0 else 0.0
    if abs(check - improvement) > 1e-9:
        raise RuntimeError(f"Improvement calculation mismatch: {improvement} vs {check}")
    print("=" * 72)


if __name__ == "__main__":
    main()
