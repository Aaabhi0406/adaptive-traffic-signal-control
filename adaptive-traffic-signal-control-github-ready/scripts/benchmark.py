"""
scripts/benchmark.py — Phase 1 baseline metrics, runnable end-to-end.

Produces the numbers the README's "Results / metrics" section says don't
exist yet:
  1. Detection FPS at the configured imgsz
  2. RF predictor R^2 once trained, over a full run
  3. Adaptive vs. fixed-timer delay comparison (see baseline.py) — the
     number a research writeup actually needs: "how much better is this
     than the classical control", not just raw FPS/R^2 in isolation.

Usage:
    python scripts/benchmark.py videos/14552311-hd_1920_1080_50fps.mp4
    python scripts/benchmark.py videos/14552311-hd_1920_1080_50fps.mp4 --report benchmark_report.json

Counting accuracy vs. a manual count is NOT automated here — that step
inherently needs a human to watch the clip and count crossings by eye (see
README "Results / metrics", item 2). Everything else in this file is fully
automated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402

from traffic_system import config  # noqa: E402
from traffic_system.baseline import compare_controllers  # noqa: E402
from traffic_system.detection import detect_vehicles, get_model  # noqa: E402
from traffic_system.pipeline import TrafficPipeline  # noqa: E402


def benchmark_fps(video_path: str, model, n_frames: int = 200) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    times = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        detect_vehicles(frame, model, frame_w=w, frame_h=h)
        times.append(time.perf_counter() - t0)
    cap.release()

    if not times:
        return {"mean_fps": 0.0, "frames_measured": 0}

    mean_time = sum(times) / len(times)
    return {"mean_fps": round(1.0 / mean_time, 1), "frames_measured": len(times)}


def run_full_trace(video_path: str, model, max_frames: int | None = None) -> dict:
    """Run the full pipeline over a video, collecting per-frame counts and
    green times, plus the RF predictor's final R^2 once trained."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    pipeline = TrafficPipeline()
    counts: list[int] = []
    adaptive_green_times: list[int] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pipeline.process_frame(frame, model)
        counts.append(result["vehicle_count"])
        adaptive_green_times.append(result["green_time"])
        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break
    cap.release()

    return {
        "counts": counts,
        "adaptive_green_times": adaptive_green_times,
        "rf_r2": pipeline.density_predictor.confidence,
        "rf_trained": pipeline.density_predictor.is_trained,
        "frames_processed": frame_idx,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 1 baseline benchmark")
    parser.add_argument("video", help="Path to a video file")
    parser.add_argument("--model", default=config.DEFAULT_MODEL_PATH)
    parser.add_argument("--fps-frames", type=int, default=200, help="Frames to measure FPS over")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap full-trace run (omit for whole video)")
    parser.add_argument("--fixed-green", type=int, default=25, help="Fixed-timer baseline green time (s)")
    parser.add_argument("--report", default=None, help="Optional path to dump JSON report")
    args = parser.parse_args(argv)

    print(f"Loading model: {args.model}")
    model = get_model(args.model)

    print(f"\n[1/2] Detection FPS over {args.fps_frames} frames...")
    fps_result = benchmark_fps(args.video, model, n_frames=args.fps_frames)
    print(f"      mean FPS: {fps_result['mean_fps']} ({fps_result['frames_measured']} frames measured)")

    print(f"\n[2/2] Full trace: counts, green-time sequence, RF R^2...")
    trace = run_full_trace(args.video, model, max_frames=args.max_frames)
    print(f"      frames processed : {trace['frames_processed']}")
    print(f"      RF trained       : {trace['rf_trained']}")
    print(f"      RF R^2           : {trace['rf_r2']:.3f}")

    comparison = compare_controllers(
        trace["counts"], trace["adaptive_green_times"], fixed_green_time=args.fixed_green
    )
    print(f"\nAdaptive vs. fixed-timer ({args.fixed_green}s) delay comparison:")
    print(f"      adaptive delay   : {comparison['adaptive_delay_vehicle_seconds']:.1f} vehicle-seconds")
    print(f"      fixed delay      : {comparison['fixed_delay_vehicle_seconds']:.1f} vehicle-seconds")
    print(f"      improvement      : {comparison['improvement_pct']:.1f}%")

    report = {
        "video": args.video,
        "fps": fps_result,
        "rf_r2": trace["rf_r2"],
        "rf_trained": trace["rf_trained"],
        "frames_processed": trace["frames_processed"],
        "delay_comparison": comparison,
    }

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
