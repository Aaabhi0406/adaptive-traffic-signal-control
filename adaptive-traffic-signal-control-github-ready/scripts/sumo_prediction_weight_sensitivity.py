"""
sumo_prediction_weight_sensitivity.py

Targeted sensitivity experiment for the research paper.

Question:
    How sensitive is adaptive signal control to the amount of trust placed in
    the RF point prediction?

Fusion:
    fused = (1 - prediction_weight) * current_SUMO_queue
            + prediction_weight * RF_point_prediction

Weights:
    0.00 -> reactive baseline behaviour (prediction ignored)
    0.25 -> low prediction trust
    0.50 -> current deployed setting
    0.75 -> high prediction trust
    1.00 -> prediction-only

IMPORTANT:
- This is a SUMO control ablation / surrogate-sensing experiment.
- It is NOT an end-to-end YOLO-video-to-SUMO evaluation.
- Existing benchmark scripts and the existing predictive benchmark are left
  untouched.

Recommended first run:
    python scripts/sumo_prediction_weight_sensitivity.py \
        --seed 0 --duration 900

This produces:
    results/sumo_prediction_weight_sensitivity.csv
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

DEFAULT_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)


@dataclass
class WeightRunResult:
    scenario: str
    seed: int
    duration_s: int
    prediction_weight: float
    waiting_time: float
    switches: int
    rf_trained_ticks_mean: float
    final_ns_demand: float
    final_ew_demand: float


def _validate_weight(weight: float) -> float:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("prediction weight must be between 0 and 1")
    return float(weight)


def _route_path(scenario: str, seed: int, duration: int) -> Path:
    return SUMO_ENV_DIR / f"weight_sensitivity_{scenario}_seed{seed}_{duration}s.rou.xml"


def run_weighted_predictive(
    route_file: Path,
    duration: int,
    seed: int,
    prediction_weight: float,
) -> dict:
    """Run one RF point-prediction fusion weight for an exact SUMO horizon."""
    del seed  # TraCI/SUMO owns the random seed through SumoTraciEnv.
    prediction_weight = _validate_weight(prediction_weight)

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
            approach: TrafficDensityPredictor() for approach in APPROACHES
        }

        last_state = None
        switches = 0
        state_seconds = {state: 0 for state in SIGNAL_STATE_MAP}
        trained_counts = {approach: 0 for approach in APPROACHES}
        last_fused = {approach: 0.0 for approach in APPROACHES}

        for sim_t in range(max(0, int(duration))):
            queues = env.get_queue_lengths()

            point_predictions: dict[str, float] = {}
            for approach in APPROACHES:
                observed = float(queues[approach])
                predictor = predictors[approach]
                predictor.update(observed)
                point, _lower, _upper, _level, _confidence = predictor.predict_interval()
                point_predictions[approach] = float(point)
                if predictor.is_trained:
                    trained_counts[approach] += 1

            # IMPORTANT:
            # Keep exactly the same fusion mechanism used by the existing
            # predictive benchmark; only the prediction_weight changes.
            fused = fuse_demand(
                {a: float(queues[a]) for a in APPROACHES},
                point_predictions,
                prediction_weight=prediction_weight,
            )
            last_fused = fused

            ns_demand = float(fused["N"] + fused["S"])
            ew_demand = float(fused["E"] + fused["W"])

            snapshot = controller.update(ns_demand, ew_demand, now=float(sim_t))
            state = snapshot["signal_state"]

            if state != last_state:
                if last_state is not None and snapshot["decision"] == "SWITCH":
                    switches += 1
                last_state = state

            import traci  # local import keeps script importable without SUMO

            traci.trafficlight.setRedYellowGreenState(
                "A1", _phase_state_for_controller_state(state)
            )
            traci.simulationStep()
            env._accumulate_waiting_time()
            state_seconds[state] += 1

        mean_trained_ticks = sum(trained_counts.values()) / (4.0 * max(duration, 1))

        return {
            "waiting_time": env.total_waiting_time(),
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
    prediction_weight: float,
) -> WeightRunResult:
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Use one of: {', '.join(SCENARIOS)}"
        )

    prediction_weight = _validate_weight(prediction_weight)
    rates = SCENARIOS[scenario]
    route_path = _route_path(scenario, seed, duration)
    build_route_file(rates, route_path, sim_duration_s=duration, seed=seed)

    result = run_weighted_predictive(
        route_path,
        duration=duration,
        seed=seed,
        prediction_weight=prediction_weight,
    )

    return WeightRunResult(
        scenario=scenario,
        seed=seed,
        duration_s=duration,
        prediction_weight=prediction_weight,
        waiting_time=float(result["waiting_time"]),
        switches=int(result["switches"]),
        rf_trained_ticks_mean=float(result["rf_trained_ticks_mean"]),
        final_ns_demand=float(result["final_ns_demand"]),
        final_ew_demand=float(result["final_ew_demand"]),
    )


def _print_result(result: WeightRunResult) -> None:
    print("\n" + "=" * 72)
    print("RF PREDICTION-WEIGHT SENSITIVITY")
    print("=" * 72)
    print(f"Scenario:              {result.scenario}")
    print(f"Seed:                  {result.seed}")
    print(f"Duration:              {result.duration_s}s")
    print(f"Prediction weight:     {result.prediction_weight:.2f}")
    print(f"Waiting time:          {result.waiting_time:.1f} vehicle-seconds")
    print(f"Signal switches:       {result.switches}")
    print(f"Mean RF trained ticks: {result.rf_trained_ticks_mean * 100:.1f}%")
    print(f"Final NS fused demand: {result.final_ns_demand:.2f}")
    print(f"Final EW fused demand: {result.final_ew_demand:.2f}")
    print("=" * 72)


def _write_csv(results: list[WeightRunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "seed",
                "duration_s",
                "prediction_weight",
                "current_weight",
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
                    r.seed,
                    r.duration_s,
                    f"{r.prediction_weight:.6f}",
                    f"{1.0 - r.prediction_weight:.6f}",
                    f"{r.waiting_time:.6f}",
                    r.switches,
                    f"{r.rf_trained_ticks_mean:.6f}",
                    f"{r.final_ns_demand:.6f}",
                    f"{r.final_ew_demand:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SUMO sensitivity analysis for RF prediction trust in adaptive control."
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="balanced")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=int, default=900)
    parser.add_argument(
        "--prediction-weight",
        type=float,
        default=0.5,
        help="Weight given to RF prediction; current-demand weight is 1-w (default: 0.50).",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_WEIGHTS),
        help="Weights used with --run-all. Default: 0 0.25 0.5 0.75 1.0",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run every scenario at every requested prediction weight for one seed.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Seeds used with --run-all-seeds.",
    )
    parser.add_argument(
        "--run-all-seeds",
        action="store_true",
        help="Run every scenario, every requested weight, and every requested seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "sumo_prediction_weight_sensitivity.csv",
    )
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be positive")

    try:
        weights = [_validate_weight(w) for w in args.weights]
        for w in [args.prediction_weight]:
            _validate_weight(w)
    except ValueError as exc:
        parser.error(str(exc))

    if args.run_all or args.run_all_seeds:
        seeds = args.seeds if args.run_all_seeds else [args.seed]
        total = len(SCENARIOS) * len(seeds) * len(weights)
        results: list[WeightRunResult] = []
        idx = 0

        for scenario in SCENARIOS:
            for seed in seeds:
                for weight in weights:
                    idx += 1
                    print(
                        f"\n[{idx}/{total}] {scenario} | seed={seed} | "
                        f"prediction_weight={weight:.2f}"
                    )
                    result = run_one(
                        scenario=scenario,
                        seed=seed,
                        duration=args.duration,
                        prediction_weight=weight,
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
        prediction_weight=args.prediction_weight,
    )
    _print_result(result)


if __name__ == "__main__":
    main()
