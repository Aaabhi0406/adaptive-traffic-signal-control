"""
sumo_25pct_uncertainty_ablation.py

Final targeted ablation:
    A) Reactive + 25% RF point prediction
    B) Reactive + 25% RF upper-bound / uncertainty hedge

Current-demand weight = 0.75 in both variants.

This is a SUMO control-ablation / surrogate-sensing experiment, NOT an
end-to-end YOLO-video-to-SUMO evaluation.

Run all 4 scenarios x 5 seeds x 2 variants:
    python scripts/sumo_25pct_uncertainty_ablation.py --run-all --duration 900 --seeds 0 1 2 3 4

Output:
    results/sumo_25pct_uncertainty_ablation.csv
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

CONTROLLERS = (
    "reactive-rf-25",
    "reactive-rf-25-uncertainty",
)
PREDICTION_WEIGHT = 0.25


@dataclass
class Result:
    scenario: str
    controller: str
    seed: int
    duration_s: int
    prediction_weight: float
    uncertainty: bool
    waiting_time: float
    switches: int
    rf_trained_ticks_fraction: float
    final_ns_fused_demand: float
    final_ew_fused_demand: float


def _route_path(scenario: str, seed: int, duration: int) -> Path:
    return SUMO_ENV_DIR / f"final_25pct_{scenario}_seed{seed}_{duration}s.rou.xml"


def _run_one(route_file: Path, duration: int, seed: int, uncertainty: bool) -> dict:
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
        predictors = {a: TrafficDensityPredictor() for a in APPROACHES}
        trained_counts = {a: 0 for a in APPROACHES}
        last_state = None
        switches = 0
        last_fused = {a: 0.0 for a in APPROACHES}
        state_seconds = {state: 0 for state in SIGNAL_STATE_MAP}

        import traci

        for sim_t in range(max(0, int(duration))):
            queues = env.get_queue_lengths()
            current = {a: float(queues[a]) for a in APPROACHES}
            selected_prediction: dict[str, float] = {}

            for a in APPROACHES:
                predictor = predictors[a]
                predictor.update(current[a])
                point, _lower, upper, _level, _confidence = predictor.predict_interval()
                selected_prediction[a] = float(upper if uncertainty else point)
                if predictor.is_trained:
                    trained_counts[a] += 1

            fused = fuse_demand(
                current,
                selected_prediction,
                prediction_weight=PREDICTION_WEIGHT,
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

            traci.trafficlight.setRedYellowGreenState(
                "A1", _phase_state_for_controller_state(state)
            )
            traci.simulationStep()
            env._accumulate_waiting_time()
            state_seconds[state] += 1

        trained_fraction = sum(trained_counts.values()) / (4.0 * max(duration, 1))
        return {
            "waiting_time": env.total_waiting_time(),
            "switches": switches,
            "rf_trained_ticks_fraction": trained_fraction,
            "final_ns_fused_demand": last_fused["N"] + last_fused["S"],
            "final_ew_fused_demand": last_fused["E"] + last_fused["W"],
        }
    finally:
        env.close()


def run_one(scenario: str, seed: int, duration: int, controller: str) -> Result:
    if controller not in CONTROLLERS:
        raise ValueError(f"Unknown controller: {controller}")
    uncertainty = controller.endswith("uncertainty")
    route = _route_path(scenario, seed, duration)
    build_route_file(SCENARIOS[scenario], route, sim_duration_s=duration, seed=seed)
    r = _run_one(route, duration, seed, uncertainty)
    return Result(
        scenario=scenario,
        controller=controller,
        seed=seed,
        duration_s=duration,
        prediction_weight=PREDICTION_WEIGHT,
        uncertainty=uncertainty,
        waiting_time=float(r["waiting_time"]),
        switches=int(r["switches"]),
        rf_trained_ticks_fraction=float(r["rf_trained_ticks_fraction"]),
        final_ns_fused_demand=float(r["final_ns_fused_demand"]),
        final_ew_fused_demand=float(r["final_ew_fused_demand"]),
    )


def _write(results: list[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario", "controller", "seed", "duration_s", "prediction_weight",
            "uncertainty", "waiting_time", "switches", "rf_trained_ticks_fraction",
            "final_ns_fused_demand", "final_ew_fused_demand",
        ])
        for r in results:
            w.writerow([
                r.scenario, r.controller, r.seed, r.duration_s, r.prediction_weight,
                r.uncertainty, r.waiting_time, r.switches,
                r.rf_trained_ticks_fraction, r.final_ns_fused_demand,
                r.final_ew_fused_demand,
            ])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=900)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--controller", choices=CONTROLLERS)
    p.add_argument("--scenario", choices=sorted(SCENARIOS))
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "sumo_25pct_uncertainty_ablation.csv")
    args = p.parse_args()
    if args.duration <= 0:
        p.error("--duration must be positive")

    results: list[Result] = []
    if args.run_all:
        jobs = [(s, seed, c) for s in SCENARIOS for seed in args.seeds for c in CONTROLLERS]
        total = len(jobs)
        for i, (scenario, seed, controller) in enumerate(jobs, 1):
            print(f"\n[{i}/{total}] {scenario} | seed={seed} | {controller}")
            r = run_one(scenario, seed, args.duration, controller)
            results.append(r)
            print(f"  waiting: {r.waiting_time:.1f} vehicle-seconds")
            print(f"  switches: {r.switches}")
            print(f"  RF trained ticks: {100*r.rf_trained_ticks_fraction:.1f}%")
    else:
        scenario = args.scenario or "balanced"
        controller = args.controller or "reactive-rf-25"
        r = run_one(scenario, args.seeds[0], args.duration, controller)
        results.append(r)
        print("\n" + "=" * 72)
        print("25% RF + UNCERTAINTY ABLATION")
        print("=" * 72)
        print(f"Scenario: {scenario}")
        print(f"Controller: {controller}")
        print(f"Seed: {r.seed}")
        print(f"Duration: {r.duration_s}s")
        print(f"Waiting time: {r.waiting_time:.1f} vehicle-seconds")
        print(f"Signal switches: {r.switches}")
        print(f"RF trained ticks: {100*r.rf_trained_ticks_fraction:.1f}%")
        print("=" * 72)

    if args.run_all:
        _write(results, args.output)
        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
