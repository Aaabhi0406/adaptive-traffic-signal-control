"""
sumo_predictive_benchmark.py — predictive control ablation for the research paper.

IMPORTANT RESEARCH SCOPE
------------------------
This script ADDS two SUMO controller variants without modifying the existing
fixed/Webster/reactive benchmarks:

    1) Reactive + RF
       fused demand = 0.5 * current SUMO queue + 0.5 * RF point prediction

    2) Reactive + RF + Uncertainty
       fused demand = 0.5 * current SUMO queue + 0.5 * RF upper bound

The RF model is the project's existing TrafficDensityPredictor. In this SUMO
ablation, one predictor is maintained per approach and is fed by SUMO's
per-approach halting-vehicle count. This is a CONTROL ablation / surrogate
sensing experiment; it is NOT an end-to-end YOLO-video-to-SUMO experiment.

One SUMO simulation second is treated as one predictor/control tick. Therefore,
the project's 15-step RF horizon is 15 SUMO seconds in this ablation. This is
different from the frame-based horizon used by the video evaluation and must be
reported explicitly in the paper.

The existing baseline script remains untouched.

Usage examples:
    # 120 s smoke test, point-prediction variant
    python scripts/sumo_predictive_benchmark.py \
        --scenario balanced --seed 0 --duration 120 --controller reactive-rf

    # 120 s smoke test, uncertainty variant
    python scripts/sumo_predictive_benchmark.py \
        --scenario balanced --seed 0 --duration 120 --controller reactive-rf-uncertainty

    # Final 4 scenarios × 5 seeds × 2 predictive variants
    python scripts/sumo_predictive_benchmark.py \
        --run-all --duration 900 --seeds 0 1 2 3 4

Output for --run-all:
    results/sumo_predictive_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
SUMO_ENV_DIR = PROJECT_ROOT / "sumo_env"

sys.path.insert(0, str(SRC_DIR))

from traffic_system.intersection import fuse_demand  # noqa: E402
from traffic_system.signal_controller import (  # noqa: E402
    ConflictSafeSignalController,
    SIGNAL_STATE_MAP,
)
from traffic_system.signal_logic import TrafficDensityPredictor  # noqa: E402
from traffic_system.sumo_integration import (  # noqa: E402
    APPROACHES,
    SumoTraciEnv,
    _phase_state_for_controller_state,
    build_route_file,
)


SCENARIOS: dict[str, dict[str, float]] = {
    "balanced": {"N": 5.0, "E": 5.0, "S": 5.0, "W": 5.0},
    "ns-heavy": {"N": 8.0, "E": 2.0, "S": 8.0, "W": 2.0},
    "ew-heavy": {"N": 2.0, "E": 8.0, "S": 2.0, "W": 8.0},
    "asymmetric": {"N": 6.0, "E": 3.0, "S": 5.0, "W": 2.0},
}

CONTROLLERS = ("reactive-rf", "reactive-rf-uncertainty")


@dataclass
class PredictiveRunResult:
    scenario: str
    controller: str
    seed: int
    duration_s: int
    waiting_time: float
    switches: int
    rf_trained_ticks_mean: float
    final_ns_demand: float
    final_ew_demand: float


def _validate_controller(name: str) -> None:
    if name not in CONTROLLERS:
        raise ValueError(
            f"Unknown controller '{name}'. Use one of: {', '.join(CONTROLLERS)}"
        )


def _route_path(scenario: str, seed: int, duration: int) -> Path:
    return SUMO_ENV_DIR / f"predictive_{scenario}_seed{seed}_{duration}s.rou.xml"


def run_predictive(
    route_file: Path,
    duration: int,
    seed: int,
    controller_name: str,
) -> dict:
    """Run one predictive controller variant for an exact SUMO horizon."""
    _validate_controller(controller_name)

    use_uncertainty = controller_name == "reactive-rf-uncertainty"

    env = SumoTraciEnv(route_file, seed=seed)
    env.start()

    try:
        controller = ConflictSafeSignalController(
            min_green_s=10.0,
            max_green_s=60.0,
            switch_margin=0.15,
            starvation_threshold_s=90.0,
            ema_alpha=0.3,
            initial_group="NS",
            init_now=0.0,
        )

        predictors = {
            approach: TrafficDensityPredictor()
            for approach in APPROACHES
        }

        last_state = None
        switches = 0
        state_seconds = {state: 0 for state in SIGNAL_STATE_MAP}
        trained_counts = {approach: 0 for approach in APPROACHES}
        last_fused = {approach: 0.0 for approach in APPROACHES}

        for sim_t in range(max(0, int(duration))):
            queues = env.get_queue_lengths()

            # Feed each approach into the SAME RF implementation used by the
            # vision pipeline. Here the SUMO queue is the surrogate observed
            # demand signal for the control ablation.
            point_predictions: dict[str, float] = {}
            upper_predictions: dict[str, float] = {}

            for approach in APPROACHES:
                observed = float(queues[approach])
                predictor = predictors[approach]
                predictor.update(observed)
                point, _lower, upper, _level, _confidence = predictor.predict_interval()
                point_predictions[approach] = point
                upper_predictions[approach] = upper
                if predictor.is_trained:
                    trained_counts[approach] += 1

            selected_prediction = (
                upper_predictions if use_uncertainty else point_predictions
            )

            # Match the project's deployed architecture: 50% current demand +
            # 50% predictive demand, with uncertainty using the RF upper bound.
            fused = fuse_demand(
                {a: float(queues[a]) for a in APPROACHES},
                selected_prediction,
                prediction_weight=0.5,
            )
            last_fused = fused

            ns_demand = float(fused["N"] + fused["S"])
            ew_demand = float(fused["E"] + fused["W"])

            snapshot = controller.update(
                ns_demand,
                ew_demand,
                now=float(sim_t),
            )
            state = snapshot["signal_state"]

            if state != last_state:
                if last_state is not None and snapshot["decision"] == "SWITCH":
                    switches += 1
                last_state = state

            import traci  # local import keeps script importable without SUMO
            traci.trafficlight.setRedYellowGreenState(
                "A1",
                _phase_state_for_controller_state(state),
            )
            traci.simulationStep()
            env._accumulate_waiting_time()
            state_seconds[state] += 1

        total_wait = env.total_waiting_time()
        mean_trained_ticks = sum(trained_counts.values()) / (4.0 * max(duration, 1))

        return {
            "waiting_time": total_wait,
            "switches": switches,
            "state_seconds": state_seconds,
            "rf_trained_ticks_mean": mean_trained_ticks,
            "final_ns_demand": last_fused["N"] + last_fused["S"],
            "final_ew_demand": last_fused["E"] + last_fused["W"],
        }
    finally:
        env.close()


def run_one(
    scenario: str,
    seed: int,
    duration: int,
    controller_name: str,
) -> PredictiveRunResult:
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Use one of: {', '.join(SCENARIOS)}"
        )

    rates = SCENARIOS[scenario]
    route_path = _route_path(scenario, seed, duration)
    build_route_file(
        rates,
        route_path,
        sim_duration_s=duration,
        seed=seed,
    )

    result = run_predictive(
        route_path,
        duration=duration,
        seed=seed,
        controller_name=controller_name,
    )

    return PredictiveRunResult(
        scenario=scenario,
        controller=controller_name,
        seed=seed,
        duration_s=duration,
        waiting_time=float(result["waiting_time"]),
        switches=int(result["switches"]),
        rf_trained_ticks_mean=float(result["rf_trained_ticks_mean"]),
        final_ns_demand=float(result["final_ns_demand"]),
        final_ew_demand=float(result["final_ew_demand"]),
    )


def _print_result(result: PredictiveRunResult) -> None:
    print("\n" + "=" * 72)
    print("PREDICTIVE SUMO ABLATION")
    print("=" * 72)
    print(f"Scenario:              {result.scenario}")
    print(f"Controller:            {result.controller}")
    print(f"Seed:                  {result.seed}")
    print(f"Duration:              {result.duration_s}s")
    print(f"Waiting time:          {result.waiting_time:.1f} vehicle-seconds")
    print(f"Signal switches:       {result.switches}")
    print(f"Mean RF trained ticks: {result.rf_trained_ticks_mean * 100:.1f}%")
    print(f"Final NS fused demand: {result.final_ns_demand:.2f}")
    print(f"Final EW fused demand: {result.final_ew_demand:.2f}")
    print("=" * 72)


def _write_csv(results: list[PredictiveRunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "controller",
                "seed",
                "duration_s",
                "waiting_time",
                "switches",
                "rf_trained_ticks_fraction",
                "final_ns_fused_demand",
                "final_ew_fused_demand",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.scenario,
                    r.controller,
                    r.seed,
                    r.duration_s,
                    f"{r.waiting_time:.6f}",
                    r.switches,
                    f"{r.rf_trained_ticks_mean:.6f}",
                    f"{r.final_ns_demand:.6f}",
                    f"{r.final_ew_demand:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SUMO predictive controller ablation using the project's RF predictor."
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="balanced")
    parser.add_argument("--controller", choices=CONTROLLERS, default="reactive-rf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=int, default=900)
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run 4 scenarios × all requested seeds × both predictive controllers.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seeds used with --run-all (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "sumo_predictive_ablation.csv",
    )
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be positive")

    if args.run_all:
        results: list[PredictiveRunResult] = []
        total = len(SCENARIOS) * len(args.seeds) * len(CONTROLLERS)
        idx = 0

        for scenario in SCENARIOS:
            for seed in args.seeds:
                for controller_name in CONTROLLERS:
                    idx += 1
                    print(
                        f"\n[{idx}/{total}] {scenario} | seed={seed} | {controller_name}"
                    )
                    result = run_one(
                        scenario=scenario,
                        seed=seed,
                        duration=args.duration,
                        controller_name=controller_name,
                    )
                    results.append(result)
                    _print_result(result)

        _write_csv(results, args.output)
        print(f"\nSaved results to: {args.output}")
        return

    result = run_one(
        scenario=args.scenario,
        seed=args.seed,
        duration=args.duration,
        controller_name=args.controller,
    )
    _print_result(result)


if __name__ == "__main__":
    main()
