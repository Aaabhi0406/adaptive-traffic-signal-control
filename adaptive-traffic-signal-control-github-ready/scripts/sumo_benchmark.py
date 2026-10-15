"""
scripts/sumo_benchmark.py — Phase 6, real SUMO comparison.

Runs fixed-timer vs. Webster (intersection.allocate_cycle, driven by real
SUMO-measured queue lengths) on the generated 4-way intersection network,
using SUMO's own accumulated waiting-time metric — not the simplified
queueing approximation in baseline.py/rl_baseline.py.

Usage:
    export SUMO_HOME=/usr/share/sumo   # or wherever SUMO was installed
    python scripts/sumo_benchmark.py --cycles 15

Requires: apt package `sumo`, pip package `traci` — see
sumo_env/build_network.sh for how the network was generated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_system.sumo_integration import (  # noqa: E402
    APPROACHES,
    build_route_file,
    run_fixed_timer,
    run_webster,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 6 — real SUMO fixed-timer vs. Webster comparison")
    parser.add_argument("--cycles", type=int, default=15, help="Number of signal cycles to simulate")
    parser.add_argument("--fixed-green", type=int, default=25, help="Fixed-timer green time per approach (s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rate", action="append", default=[],
        help="Approach arrival rate override, e.g. --rate N=15 --rate E=6 (vehicles/minute)",
    )
    args = parser.parse_args(argv)

    rates = {"N": 15.0, "E": 6.0, "S": 12.0, "W": 4.0}
    for override in args.rate:
        approach, value = override.split("=")
        rates[approach.upper()] = float(value)

    sim_duration_s = args.cycles * 132  # ~ (4 greens + 4 yellows) worst case; flows just need to cover the run
    route_path = Path("sumo_env") / f"benchmark_seed{args.seed}.rou.xml"
    build_route_file(rates, route_path, sim_duration_s=sim_duration_s, seed=args.seed)

    print(f"Arrival rates (veh/min): {rates}")
    print(f"Cycles: {args.cycles}  |  Fixed green: {args.fixed_green}s per approach\n")

    print("[1/2] Running fixed-timer controller in SUMO...")
    fixed_delay = run_fixed_timer(route_path, args.cycles, fixed_green_s=args.fixed_green, seed=args.seed)
    print(f"      total waiting time: {fixed_delay:.1f} vehicle-seconds")

    print("\n[2/2] Running Webster controller in SUMO (allocation recomputed from real queues each cycle)...")
    webster_delay = run_webster(route_path, args.cycles, seed=args.seed)
    print(f"      total waiting time: {webster_delay:.1f} vehicle-seconds")

    improvement = 100.0 * (fixed_delay - webster_delay) / fixed_delay if fixed_delay > 0 else 0.0
    print(f"\nWebster vs. fixed-timer improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
