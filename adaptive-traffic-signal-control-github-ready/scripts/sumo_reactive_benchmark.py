"""
Fair SUMO benchmark: Fixed-time vs Classical Webster vs Reactive demand-driven.

Reactive uses SUMO's current halting-vehicle counts as the observed demand.
It deliberately does NOT use YOLO/RF yet; this isolates the contribution of
adaptive signal switching before we integrate the vision/prediction pipeline.
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
    rates = {a: 5.0 for a in APPROACHES}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid --rate '{override}'. Use e.g. --rate N=5")
        approach, value = override.split("=", 1)
        approach = approach.strip().upper()
        if approach not in APPROACHES:
            raise ValueError(f"Unknown approach '{approach}'. Use N, E, S, or W.")
        rate = float(value)
        if rate < 0:
            raise ValueError("Arrival rates cannot be negative.")
        rates[approach] = rate
    return rates


def run_fixed(route_file: Path, duration: int, fixed_green: int, seed: int) -> float:
    env = SumoTraciEnv(route_file, seed=seed)
    env.start()
    try:
        return _run_allocation(env, {"NS": float(fixed_green), "EW": float(fixed_green)}, duration)
    finally:
        env.close()


def run_webster(route_file: Path, duration: int, seed: int) -> tuple[float, dict[str, float]]:
    flows = _read_rates_from_route(route_file)
    allocation = classical_webster_split(flows)
    env = SumoTraciEnv(route_file, seed=seed)
    env.start()
    try:
        wait = _run_allocation(env, allocation, duration)
        return wait, allocation
    finally:
        env.close()


def run_reactive(route_file: Path, duration: int, seed: int) -> dict:
    env = SumoTraciEnv(route_file, seed=seed)
    env.start()
    try:
        return env.run_reactive_duration(duration)
    finally:
        env.close()


def _run_allocation(env: SumoTraciEnv, allocation: dict[str, float], duration: int) -> float:
    env.run_compatible_duration(allocation, duration)
    return env.total_waiting_time()


def _read_rates_from_route(route_file: Path) -> dict[str, float]:
    return __import__("traffic_system.sumo_integration", fromlist=["read_arrival_rates_from_route_file"]).read_arrival_rates_from_route_file(route_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=900)
    parser.add_argument("--fixed-green", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rate", action="append", default=[])
    args = parser.parse_args()

    if args.duration <= 0 or args.fixed_green <= 0:
        parser.error("duration and fixed-green must be positive")

    try:
        rates = parse_rates(args.rate)
    except ValueError as exc:
        parser.error(str(exc))

    flows = {a: rates[a] * 60.0 for a in APPROACHES}
    ns_flow = flows["N"] + flows["S"]
    ew_flow = flows["E"] + flows["W"]
    Y = ns_flow / SATURATION_FLOW_VPH + ew_flow / SATURATION_FLOW_VPH
    if Y >= 0.95:
        parser.error(f"Webster baseline requires Y < 0.95; received Y={Y:.3f}")

    route_path = Path("sumo_env") / f"reactive_benchmark_seed{args.seed}_{args.duration}s.rou.xml"
    build_route_file(rates, route_path, sim_duration_s=args.duration, seed=args.seed)

    fixed_alloc = {"NS": float(args.fixed_green), "EW": float(args.fixed_green)}
    webster_wait_preview, webster_alloc = run_webster(route_path, args.duration, args.seed)

    print("=" * 72)
    print("FAIR SUMO CONTROLLER BENCHMARK")
    print("=" * 72)
    print(f"Arrival rates (veh/min): {rates}")
    print(f"NS flow: {ns_flow:.1f} veh/h")
    print(f"EW flow: {ew_flow:.1f} veh/h")
    print(f"Demand ratio Y: {Y:.3f}")
    print(f"Simulation duration: {args.duration}s")
    print(f"Random seed: {args.seed}")
    print(f"Fixed allocation:   NS={fixed_alloc['NS']:.1f}s EW={fixed_alloc['EW']:.1f}s")
    print(f"Webster allocation: NS={webster_alloc['NS']:.1f}s EW={webster_alloc['EW']:.1f}s")

    print("\n[1/3] Running fixed-time controller...")
    fixed_wait = run_fixed(route_path, args.duration, args.fixed_green, args.seed)
    print(f"      waiting: {fixed_wait:.1f} vehicle-seconds")

    print("[2/3] Running classical Webster controller...")
    webster_wait = webster_wait_preview
    print(f"      waiting: {webster_wait:.1f} vehicle-seconds")

    print("[3/3] Running reactive demand-driven controller...")
    reactive = run_reactive(route_path, args.duration, args.seed)
    reactive_wait = reactive["waiting_time"]
    print(f"      waiting: {reactive_wait:.1f} vehicle-seconds")
    print(f"      signal switches: {reactive['switches']}")

    def improvement(base: float, method: float) -> float:
        return 100.0 * (base - method) / base if base > 0 else 0.0

    print("\n" + "=" * 72)
    print(f"Fixed-time waiting:         {fixed_wait:.1f}")
    print(f"Classical Webster waiting:  {webster_wait:.1f}")
    print(f"Reactive waiting:           {reactive_wait:.1f}")
    print(f"Webster vs fixed:           {improvement(fixed_wait, webster_wait):.2f}%")
    print(f"Reactive vs fixed:          {improvement(fixed_wait, reactive_wait):.2f}%")
    print(f"Reactive vs Webster:        {improvement(webster_wait, reactive_wait):.2f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
