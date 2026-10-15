# Experimental results and interpretation

This page records the currently available experimental outputs. It is intentionally conservative: the video and SUMO evaluations are separate, and neither establishes live-road performance.

## 1. SUMO baseline comparison

Four demand regimes were simulated for 900 seconds with five seeds per controller/scenario. Reported values are means of the metric recorded by the project. Lower is better.

| Scenario | Fixed-time mean | Webster mean | Reactive mean | Reactive vs fixed-time | Best mean |
|---|---:|---:|---:|---:|---|
| Balanced | 22,260.8 | 12,396.6 | 8,856.6 | 60.2% lower | Reactive |
| N/S-heavy | 22,287.6 | 9,566.0 | 9,819.0 | 55.9% lower | Webster |
| E/W-heavy | 21,098.2 | 7,616.2 | 10,164.2 | 51.8% lower | Webster |
| Asymmetric | 17,490.0 | 5,891.2 | 7,002.2 | 60.0% lower | Webster |

Demand rates (vehicles/minute) used in the scenario definitions:

| Scenario | N | E | S | W |
|---|---:|---:|---:|---:|
| Balanced | 5 | 5 | 5 | 5 |
| N/S-heavy | 8 | 2 | 8 | 2 |
| E/W-heavy | 2 | 8 | 2 | 8 |
| Asymmetric | 6 | 3 | 5 | 2 |

### Metric and fairness caveats

- The reported quantity is **accumulated approach-edge waiting time as implemented in the SUMO wrapper**, accumulated across simulation steps. It should not be relabelled as conventional trip delay or travel time without separate metric verification.
- The network is a synthetic four-arm junction, not a map of a real intersection. Demand rates are scenario inputs.
- The fixed-time and classical Webster baselines in these runs did not include the adaptive controller's 2-second all-red clearance. This is a fairness caveat.
- Webster timing is calculated from known scenario arrival rates, while the reactive controller responds to simulated queue observations; the comparison is not a perfectly matched information setting.
- Five seeds provide a useful repeatability check, not proof of statistical significance. No inferential significance tests are claimed.

Raw run values are in [`../results/sumo_baseline_results_5seeds.csv`](../results/sumo_baseline_results_5seeds.csv).

## 2. RF video forecasting: RF versus EMA

The evaluation script processed two short clips. Forecast horizon is 15 frames. For the 50 FPS clip, this is approximately 0.30 seconds; for the 60 FPS clip, approximately 0.25 seconds. Each prediction was matched to the weighted density produced by the detector pipeline 15 frames later.

| Clip / evaluation subset | n | RF MAE | EMA MAE | RF RMSE | EMA RMSE | RF R² | EMA R² | RF interval coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Video 1, trained predictions | 151 | 1.9345 | 2.3113 | 2.4819 | 2.7982 | -0.3307 | -0.6915 | 70.20% |
| Video 1, all matched rows | 285 | 1.8481 | 2.0839 | 2.3320 | 2.5447 | -0.1732 | -0.3970 | Not summarized |
| Video 2, trained predictions | 151 | 1.1284 | 1.1775 | 1.4106 | 1.4335 | -0.3331 | -0.3767 | 56.95% |
| Video 2, all matched rows | 285 | 1.3430 | 1.3429 | 1.6525 | 1.6480 | 0.1763 | 0.1808 | Not summarized |

Mean prediction-interval width on trained rows was 5.444 for Video 1 and 2.401 for Video 2. Coverage is the observed fraction of rows where the target fell between the stored lower and upper bounds. These bounds are derived from tree disagreement and are **not calibrated confidence intervals**.

Interpretation:

- On the trained subset of each clip, RF had lower MAE and RMSE than EMA; both trained-subset R² values were negative, meaning neither performed better than the subset-mean predictor under R²'s definition.
- Full-sample evaluation includes 134 startup predictions for which the RF had not yet trained, so it is not an RF-only comparison. On Video 2, full-sample RF and EMA errors were effectively tied.
- The targets are detector-derived weighted ROI density values, not manual annotations or independent ground truth. These results do not establish detection accuracy, tracking accuracy, or generalization to other videos.
- An exploratory persistence baseline was calculated post hoc on the trained subsets; it is supplementary rather than a pre-registered/main planned baseline.

Raw frame-level predictions are in [`../results/rf_video1_eval.csv`](../results/rf_video1_eval.csv) and [`../results/rf_video2_eval.csv`](../results/rf_video2_eval.csv). The recomputed summary is [`../results/rf_video_evaluation_summary.csv`](../results/rf_video_evaluation_summary.csv).

## 3. Prediction-weight sensitivity in SUMO

The fusion policy combines current queue observations and RF forecasts. A prediction weight of 0 means reactive-only; 0.25 means 75% current queue and 25% RF point prediction; 0.5 means equal weighting.

### Five-seed 0.25-weight comparison

| Scenario | Reactive mean ± SD | RF at 0.25 mean ± SD | RF at 0.25 relative change | Runs where 0.25 was better |
|---|---:|---:|---:|---:|
| Balanced | 8,856.6 ± 223.2 | 8,447.4 ± 184.0 | 4.62% lower | 5/5 |
| N/S-heavy | 9,819.0 ± 424.6 | 10,231.0 ± 678.4 | 4.20% higher | 1/5 |
| E/W-heavy | 10,164.2 ± 1,591.2 | 9,859.4 ± 555.2 | 3.00% lower | 2/5 |
| Asymmetric | 7,002.2 ± 279.8 | 7,935.8 ± 153.8 | 13.33% higher | 0/5 |

The equal-scenario mean is 8,960.5 for reactive control and 9,118.4 for the 0.25 prediction-weight controller: a 1.76% increase in the recorded waiting metric. Benefits are demand-regime dependent. The E/W-heavy mean improvement is sensitive to one high reactive seed and should not be presented as a robust win.

**Selection caveat:** the 0.25 weight was chosen after an initial seed-0 sweep; the later five-seed table includes seed 0, so it is not fully independent validation.

### 0.50-weight controller

At the original equal-weight setting (0.50 RF / 0.50 current queue), mean waiting was higher than reactive control in all four scenarios: balanced 9,940.0; N/S-heavy 13,837.0; E/W-heavy 12,490.8; asymmetric 8,384.8. It did not outperform reactive in the five-seed scenario summaries.

## 4. Uncertainty adjustment

The tested upper-bound variant substitutes a one-sided RF upper estimate based on point prediction plus a multiple of the tree prediction standard deviation. This is a heuristic uncertainty adjustment, not a calibrated probabilistic model.

| Scenario | 0.25 RF point estimate | 0.25 RF + uncertainty | Change in mean metric | Runs where uncertainty was better |
|---|---:|---:|---:|---:|
| Balanced | 8,447.4 ± 184.0 | 8,602.0 ± 313.6 | 1.83% higher | 1/5 |
| N/S-heavy | 10,231.0 ± 678.4 | 12,753.2 ± 4,184.8 | 24.65% higher | 3/5 |
| E/W-heavy | 9,859.4 ± 555.2 | 9,801.8 ± 1,485.4 | 0.58% lower | 4/5 |
| Asymmetric | 7,935.8 ± 153.8 | 8,019.4 ± 650.9 | 1.05% higher | 2/5 |

Across equal scenario weights, the point estimate mean was 9,118.4 and the uncertainty variant was 9,794.1, a 7.41% increase in the recorded metric. The N/S-heavy mean is strongly influenced by two high observations; its median comparison differs from its mean comparison. The overall evidence does not support claiming that the tested uncertainty adjustment improves control.

Raw CSV outputs are under [`../results/`](../results/): `sumo_predictive_ablation.csv`, `sumo_prediction_weight_sweep_seed0.csv`, `sumo_prediction_weight_sensitivity_5seeds.csv`, and `sumo_25pct_uncertainty_ablation.csv`.

## 5. Defensible conclusion

The available experiments support a bounded conclusion: reactive control was better than fixed-time control in the tested SUMO scenarios, while Webster was best in three of four baseline regimes. Adding short-horizon RF predictions changed controller behaviour, but predictive performance depended on the demand regime and on prediction weight. The tested uncertainty adjustment did not consistently help. Longer and more diverse videos, independent annotations, calibrated uncertainty, and more balanced controller baselines are needed before stronger claims can be made.
