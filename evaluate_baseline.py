"""
evaluate_baseline.py
====================
Evaluate YOLOv8n baseline on HRSID test set.

Reports comprehensive metrics:
  - Precision, Recall, F1-Score
  - mAP@0.5, mAP@0.5:0.95
  - False Positive Rate
  - Confusion matrix
  - Per-image prediction visualizations

Usage:
  python evaluate_baseline.py
  python evaluate_baseline.py --model runs/baseline/weights/best.pt --conf 0.25
"""

import argparse
import json
import csv
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# --- Configuration -----------------------------------------------------------

DEFAULT_MODEL = Path(r"d:\CAPSTONE_HYPOTHESIS\runs\baseline\weights\best.pt")
DATASET_YAML = Path(r"d:\CAPSTONE_HYPOTHESIS\hrsid_yolo\hrsid.yaml")
OUTPUT_DIR = Path(r"d:\CAPSTONE_HYPOTHESIS\evaluation_results")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8n on HRSID test set")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="Path to trained model weights")
    parser.add_argument("--data", type=str, default=str(DATASET_YAML),
                        help="Path to dataset YAML")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for NMS (default: 0.5)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size for inference (default: 16)")
    parser.add_argument("--save-predictions", action="store_true", default=True,
                        help="Save prediction visualizations")
    parser.add_argument("--name", type=str, default="baseline",
                        help="Experiment name for output organization")
    return parser.parse_args()


def compute_detailed_metrics(model, args):
    """
    Run validation on the test set and extract comprehensive metrics.
    """
    print("\n[1/3] Running model validation on test set...")

    # Run YOLO's built-in validation on test split
    results = model.val(
        data=args.data,
        split="test",
        batch=args.batch,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
        verbose=True,
    )

    # Extract metrics from results
    metrics = {}

    # Box metrics (detection)
    metrics["precision"] = float(results.box.mp)      # Mean precision
    metrics["recall"] = float(results.box.mr)          # Mean recall
    metrics["mAP50"] = float(results.box.map50)        # mAP@0.5
    metrics["mAP50_95"] = float(results.box.map)       # mAP@0.5:0.95

    # F1 score
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / \
                        (metrics["precision"] + metrics["recall"])
    else:
        metrics["f1"] = 0.0

    # False positive rate estimate
    # FP rate = 1 - Precision (approximation at the given conf threshold)
    metrics["false_positive_rate"] = 1.0 - metrics["precision"]
    metrics["confidence_threshold"] = args.conf
    metrics["iou_threshold"] = args.iou

    return metrics, results


def run_test_predictions(model, args, output_dir):
    """
    Run inference on the test set and save prediction visualizations.
    """
    print("\n[2/3] Running predictions on test set...")

    test_images_dir = Path(r"d:\CAPSTONE_HYPOTHESIS\hrsid_yolo\test\images")
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Run prediction with visualization
    results = model.predict(
        source=str(test_images_dir),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=args.save_predictions,
        save_txt=True,
        save_conf=True,
        project=str(output_dir),
        name="predictions",
        exist_ok=True,
        verbose=False,
    )

    # Collect per-image statistics
    per_image_stats = []
    total_detections = 0
    images_with_detections = 0

    for r in results:
        img_name = Path(r.path).name
        num_dets = len(r.boxes)
        total_detections += num_dets
        if num_dets > 0:
            images_with_detections += 1

        confs = r.boxes.conf.cpu().numpy().tolist() if num_dets > 0 else []
        per_image_stats.append({
            "image": img_name,
            "num_detections": num_dets,
            "avg_confidence": float(np.mean(confs)) if confs else 0.0,
            "max_confidence": float(np.max(confs)) if confs else 0.0,
            "min_confidence": float(np.min(confs)) if confs else 0.0,
        })

    summary = {
        "total_test_images": len(results),
        "images_with_detections": images_with_detections,
        "total_detections": total_detections,
        "avg_detections_per_image": total_detections / len(results) if results else 0,
    }

    return per_image_stats, summary


def save_results(metrics, per_image_stats, summary, output_dir, experiment_name):
    """Save all results to JSON and CSV files."""
    print("\n[3/3] Saving results...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Save comprehensive metrics JSON ---
    results_json = {
        "experiment": experiment_name,
        "metrics": metrics,
        "test_summary": summary,
    }
    json_path = output_dir / f"{experiment_name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Metrics JSON: {json_path}")

    # --- Save per-image stats CSV ---
    csv_path = output_dir / f"{experiment_name}_per_image.csv"
    if per_image_stats:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_image_stats[0].keys())
            writer.writeheader()
            writer.writerows(per_image_stats)
        print(f"  Per-image CSV: {csv_path}")

    # --- Print summary table ---
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS: {experiment_name}")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'Precision':<25} {metrics['precision']:>10.4f}")
    print(f"  {'Recall':<25} {metrics['recall']:>10.4f}")
    print(f"  {'F1-Score':<25} {metrics['f1']:>10.4f}")
    print(f"  {'mAP@0.5':<25} {metrics['mAP50']:>10.4f}")
    print(f"  {'mAP@0.5:0.95':<25} {metrics['mAP50_95']:>10.4f}")
    print(f"  {'False Positive Rate':<25} {metrics['false_positive_rate']:>10.4f}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'Conf Threshold':<25} {metrics['confidence_threshold']:>10.2f}")
    print(f"  {'IoU Threshold':<25} {metrics['iou_threshold']:>10.2f}")
    print("=" * 60)

    print(f"\n  Test images:              {summary['total_test_images']}")
    print(f"  Images with detections:   {summary['images_with_detections']}")
    print(f"  Total detections:         {summary['total_detections']}")
    print(f"  Avg detections/image:     {summary['avg_detections_per_image']:.2f}")

    return json_path


def main():
    args = parse_args()

    print("=" * 60)
    print("YOLOv8n Ship Detection - Baseline Evaluation")
    print("=" * 60)
    print(f"  Model:  {args.model}")
    print(f"  Data:   {args.data}")
    print(f"  Conf:   {args.conf}")
    print(f"  IoU:    {args.iou}")

    # Verify model exists
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run train_baseline.py first to train the model."
        )

    # Load model
    model = YOLO(str(model_path))
    print(f"\nLoaded model from: {model_path}")

    # Create output directory
    output_dir = OUTPUT_DIR / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute metrics
    metrics, val_results = compute_detailed_metrics(model, args)

    # Step 2: Run predictions and save visualizations
    per_image_stats, summary = run_test_predictions(model, args, output_dir)

    # Step 3: Save everything
    json_path = save_results(metrics, per_image_stats, summary, output_dir, args.name)

    print(f"\nAll results saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
