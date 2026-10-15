# SUMO environment and synthetic intersection

This directory contains the generated four-arm signalized junction used for microscopic simulation experiments. It is a **synthetic network**, not a surveyed map or digital twin of a real intersection. Scenario arrival rates are illustrative.

## Requirements

- Install SUMO using the [official installation guide](https://sumo.dlr.de/docs/Installing/index.html).
- Ensure the `sumo` executable is available on `PATH`.
- Ensure the SUMO tools directory is available to Python if required by your installation (`SUMO_HOME` is commonly used).
- Install the repository's Python dependencies with `pip install -r requirements.txt`.

## Reproducible experiment scripts

From the repository root, after installation:

```bash
# Fixed-time vs Webster vs reactive; one seed and one demand scenario
python scripts/sumo_reactive_benchmark.py --duration 900 --seed 0 \
  --rate N=5 --rate E=5 --rate S=5 --rate W=5

# Prediction-weight sweep over five seeds and all scripted scenarios
python scripts/sumo_prediction_weight_sensitivity.py --run-all-seeds \
  --duration 900 --seeds 0 1 2 3 4

# RF point estimate vs. the simple upper-bound uncertainty variant
python scripts/sumo_25pct_uncertainty_ablation.py --run-all \
  --duration 900 --seeds 0 1 2 3 4

# Original 50%-prediction point and uncertainty ablation
python scripts/sumo_predictive_benchmark.py --run-all \
  --duration 900 --seeds 0 1 2 3 4
```

The multi-seed ablations can take a while. The scripts write route files and CSV outputs to project-relative paths. Generated route files should remain local and are excluded by `.gitignore`; curated result CSVs are tracked under `results/`.

## Scenario design

| Scenario | N | E | S | W |
|---|---:|---:|---:|---:|
| Balanced | 5 | 5 | 5 | 5 |
| N/S-heavy | 8 | 2 | 8 | 2 |
| E/W-heavy | 2 | 8 | 2 | 8 |
| Asymmetric | 6 | 3 | 5 | 2 |

Rates are vehicles per minute. Simulated runs are 900 seconds and use seeds 0–4 for the main summaries.

## Interpretation requirements

- The predictor in the SUMO ablations is fed by SUMO's own per-approach halting-vehicle count. This is a surrogate-sensing control experiment, not an end-to-end YOLO-video-to-SUMO evaluation.
- A 15-step forecast means 15 simulated seconds in SUMO. The video evaluation uses 15 frames, which is approximately 0.25–0.30 seconds for the two clips examined.
- The reported metric is accumulated approach-edge waiting time as implemented in this project. Do not describe it as conventional trip delay without further metric validation.
- In the recorded baseline comparison, fixed-time and classical Webster did not include the adaptive controller's 2-second all-red clearance. Webster also uses the scenario arrival rates to compute its split. These are fairness caveats.
- Detailed numerical results and raw CSV references are in [`../docs/EXPERIMENTAL_RESULTS.md`](../docs/EXPERIMENTAL_RESULTS.md).
