import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2

from traffic_system import config
from traffic_system.detection import get_model
from traffic_system.pipeline import TrafficPipeline


def calculate_ema_baseline(history, alpha=0.3):
    """
    Calculate an EMA baseline using observations available at the
    current forecast frame. No future observations are used.
    """
    values = list(history)

    if not values:
        return 0.0

    ema = float(values[0])

    for value in values[1:]:
        ema = alpha * float(value) + (1.0 - alpha) * ema

    return float(ema)


def main():
    parser = argparse.ArgumentParser(
        description="Capture RF and EMA predictions for evaluation."
    )

    parser.add_argument("video", help="Path to input video")

    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL_PATH,
        help="YOLO model path",
    )

    parser.add_argument(
        "--output",
        default="results/rf_eval.csv",
        help="Output CSV path",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process; 0 means the full video",
    )

    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO model...", flush=True)
    model = get_model(args.model)

    print("Initializing traffic pipeline...", flush=True)
    pipeline = TrafficPipeline()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    horizon = config.RF_PREDICT_HORIZON

    print(f"Video: {video_path}", flush=True)
    print(f"Video FPS: {video_fps:.2f}", flush=True)
    print(f"Video frames: {total_video_frames}", flush=True)
    print(f"Prediction horizon: {horizon} frames", flush=True)
    print("EMA alpha: 0.3", flush=True)
    print("Starting frame processing...", flush=True)

    pending = deque()

    frame_index = 0
    matched_forecasts = 0
    trained_forecasts = 0

    start_time = time.time()

    fields = [
        "forecast_frame",
        "target_frame",
        "prediction",
        "ema_prediction",
        "lower",
        "upper",
        "trained",
        "actual",
        "absolute_error",
        "ema_absolute_error",
        "inside_interval",
    ]

    try:
        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fields,
            )

            writer.writeheader()

            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                # Process current frame through the traffic pipeline.
                result = pipeline.process_frame(frame, model)

                actual_now = float(result["total_weight"])

                # Get observation history after current frame processing.
                predictor_history = pipeline.density_predictor.history

                # Calculate EMA baseline using only available history.
                ema_prediction = calculate_ema_baseline(
                    predictor_history,
                    alpha=0.3,
                )

                # Store RF and EMA forecasts for future matching.
                pending.append({
                    "forecast_frame": frame_index,
                    "target_frame": frame_index + horizon,
                    "prediction": float(result["pred_density"]),
                    "ema_prediction": ema_prediction,
                    "lower": float(result["pred_lower"]),
                    "upper": float(result["pred_upper"]),
                    "trained": bool(
                        pipeline.density_predictor.is_trained
                    ),
                })

                # Match forecast from horizon frames earlier
                # with the actual weighted density at this frame.
                while (
                    pending
                    and pending[0]["target_frame"] == frame_index
                ):
                    forecast = pending.popleft()

                    writer.writerow({
                        **forecast,

                        "actual": actual_now,

                        "absolute_error": abs(
                            forecast["prediction"] - actual_now
                        ),

                        "ema_absolute_error": abs(
                            forecast["ema_prediction"] - actual_now
                        ),

                        "inside_interval": (
                            forecast["lower"]
                            <= actual_now
                            <= forecast["upper"]
                        ),
                    })

                    matched_forecasts += 1

                    if forecast["trained"]:
                        trained_forecasts += 1

                frame_index += 1

                # Print progress every 100 frames.
                if frame_index % 100 == 0:
                    elapsed = time.time() - start_time

                    fps = (
                        frame_index / elapsed
                        if elapsed > 0
                        else 0
                    )

                    print(
                        f"Processed {frame_index}/{total_video_frames} "
                        f"frames | {fps:.2f} frames/sec | "
                        f"Matched forecasts: {matched_forecasts}",
                        flush=True,
                    )

                # Stop if maximum frame limit is reached.
                if (
                    args.max_frames > 0
                    and frame_index >= args.max_frames
                ):
                    print(
                        f"Reached --max-frames limit "
                        f"({args.max_frames}).",
                        flush=True,
                    )
                    break

    finally:
        cap.release()

    elapsed = time.time() - start_time

    print("\nEvaluation capture complete.", flush=True)
    print(f"Frames processed: {frame_index}", flush=True)
    print(f"Matched forecasts: {matched_forecasts}", flush=True)

    print(
        f"Forecasts made after RF training: {trained_forecasts}",
        flush=True,
    )

    print(
        f"Elapsed time: {elapsed / 60:.2f} minutes",
        flush=True,
    )

    print(
        f"Saved CSV: {output_path.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()