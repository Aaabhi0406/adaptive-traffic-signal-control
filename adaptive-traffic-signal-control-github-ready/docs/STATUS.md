# Project status: implementation versus validation

An implemented feature is not automatically a validated research result. This table separates the code architecture from the available experiments.

| Component | Implementation status | Available evaluation | Main caveat |
|---|---|---|---|
| YOLOv8 detection and ROI demand estimation | Implemented | Run on two short traffic clips | Targets remain detector-derived; no manually annotated detection ground truth |
| DeepSORT tracking and traffic analytics | Implemented | Qualitative application output | No formal tracking benchmark such as MOTA/IDF1 is reported |
| RF short-horizon density prediction | Implemented | Two clips, 15-frame horizon, RF vs EMA | Small sample; online same-clip evaluation; startup forecasts included in full-sample totals |
| Intersection-level demand allocation | Implemented and unit-tested in the project | Dashboard/demo mode | Not validated using four synchronized cameras at one real intersection |
| Conflict-safe NS/EW phase controller | Implemented | Software tests and SUMO experiments | Research prototype, not safety-certified hardware control |
| Fixed-time / Webster / reactive SUMO baselines | Implemented | Four synthetic demand regimes, five seeds, 900 seconds | Synthetic network; metric semantics and unequal all-red timing must be disclosed |
| RF predictive-control ablation | Implemented | Four SUMO regimes and sensitivity variants | Uses SUMO queue counts as surrogate sensing; not end-to-end YOLO-to-SUMO |
| Simple RF uncertainty upper-bound | Implemented | Compared with RF point prediction at 0.25 weight | Heuristic, not calibrated; no consistent benefit observed |
| Public traffic dataset validation | Not completed | None | Needs an appropriately licensed benchmark and labels |
| Multi-intersection green-wave coordination | Not completed | None | Future work |
| Live traffic-signal hardware integration | Not completed | None | Out of scope; safety certification and external approval required |

## Reproducibility notes

- Main SUMO baseline runs: 4 scenarios × 5 seeds × 900 seconds.
- Prediction weight 0.25 was selected after a seed-0 weight sweep, and seed 0 is included in the subsequent multi-seed table. Treat the multi-seed comparison as exploratory rather than independent holdout validation.
- Video forecast horizon is 15 frames (approximately 0.30 s at 50 FPS and 0.25 s at 60 FPS). In the SUMO predictor experiment, 15 steps mean 15 simulated seconds.
- See [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md) for source CSVs, summarized numbers, and interpretation caveats.
