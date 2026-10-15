# Experimental CSVs

This folder contains the recorded output files used for the summaries in [`../docs/EXPERIMENTAL_RESULTS.md`](../docs/EXPERIMENTAL_RESULTS.md).

- `rf_video1_eval.csv`, `rf_video2_eval.csv`: frame-level RF/EMA forecast evaluation for two short clips. The targets are detector-derived weighted ROI density, not manual ground truth.
- `rf_video_evaluation_summary.csv`: recomputed aggregate metrics for trained and full matched samples.
- `sumo_baseline_results_5seeds.csv`: recorded per-seed waiting metrics for fixed-time, Webster, and reactive controllers.
- `sumo_predictive_ablation.csv`: 50%-prediction point-estimate and uncertainty variants.
- `sumo_prediction_weight_sweep_seed0.csv`: initial five-weight sweep using seed 0.
- `sumo_prediction_weight_sensitivity_5seeds.csv`: reactive (weight 0) versus 0.25 RF point predictions across five seeds.
- `sumo_25pct_uncertainty_ablation.csv`: five-seed comparison of 0.25 RF point estimate versus its uncertainty-upper-bound variant.

SUMO metric values are the accumulated approach-edge waiting-time measure produced by the current simulator wrapper. Do not relabel these as trip delay without verifying the metric definition. Re-running scripts may overwrite result files; preserve a copy of the existing CSVs before conducting new experiments.
