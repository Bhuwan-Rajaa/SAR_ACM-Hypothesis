"""
run_acm_pipeline.py
===================
End-to-end pipeline: YOLOv8n inference + ACM post-processing on HRSID test set.

Runs the trained baseline model, applies Morphological Chan-Vese filtering
to each detection, and evaluates the filtered results against ground truth.

Usage:
    python run_acm_pipeline.py
    python run_acm_pipeline.py --conf 0.25 --num_iter 150
"""

import argparse
import json
import csv
import time
import os
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

from acm_postprocess import ACMFilter, apply_acm_filter


# --- Configuration ----------------------------------------------------------

MODEL_PATH = Path(r"d:\CAPSTONE_HYPOTHESIS\runs\baseline\weights\best.pt")
DATASET_YAML = Path(r"d:\CAPSTONE_HYPOTHESIS\hrsid_yolo\hrsid.yaml")
TEST_IMAGES = Path(r"d:\CAPSTONE_HYPOTHESIS\hrsid_yolo\test\images")
TEST_LABELS = Path(r"d:\CAPSTONE_HYPOTHESIS\hrsid_yolo\test\labels")
OUTPUT_DIR = Path(r"d:\CAPSTONE_HYPOTHESIS\evaluation_results\acm")
BASELINE_METRICS = Path(r"d:\CAPSTONE_HYPOTHESIS\evaluation_results\baseline\baseline_metrics.json")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8n + ACM Pipeline")

    # Model
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLOv8 confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640)

    # ACM parameters
    parser.add_argument("--num_iter", type=int, default=100,
                        help="Chan-Vese iterations")
    parser.add_argument("--smoothing", type=int, default=1)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.5)
    parser.add_argument("--bbox_padding", type=float, default=0.5)
    parser.add_argument("--median_filter_size", type=int, default=3)

    # Filtering thresholds
    parser.add_argument("--min_area_ratio", type=float, default=0.05)
    parser.add_argument("--max_area_ratio", type=float, default=0.85)
    parser.add_argument("--max_compactness", type=float, default=0.80)
    parser.add_argument("--min_elongation", type=float, default=1.5)
    parser.add_argument("--min_contrast_ratio", type=float, default=3.0)
    # The confidence gate is no longer used since we apply soft scaling to all detections
    # parser.add_argument("--conf_gate", type=float, default=0.75)
    # parser.add_argument("--quality_weight", type=float, default=0.3)

    # Visualization
    parser.add_argument("--save_viz", type=int, default=30,
                        help="Number of images to save visualizations for")

    return parser.parse_args()


def load_ground_truth(labels_dir: Path):
    """Load YOLO-format ground truth labels."""
    gt = {}
    for label_file in labels_dir.glob("*.txt"):
        stem = label_file.stem
        boxes = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, w, h = map(float, parts)
                boxes.append([cx, cy, w, h])
        gt[stem] = np.array(boxes) if boxes else np.zeros((0, 4))
    return gt


def yolo_to_xyxy(yolo_box, img_w, img_h):
    """Convert YOLO [cx, cy, w, h] normalized to [x1, y1, x2, y2] pixels."""
    cx, cy, w, h = yolo_box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def match_detections_to_gt(det_boxes, gt_boxes, iou_threshold=0.5):
    """
    Match detections to ground truth using greedy IoU matching.

    Returns:
        tp_indices: indices of detections that are true positives
        fp_indices: indices of detections that are false positives
        matched_gt: indices of GT boxes that were matched
    """
    if len(det_boxes) == 0:
        return [], [], []
    if len(gt_boxes) == 0:
        return [], list(range(len(det_boxes))), []

    # Compute IoU matrix
    iou_matrix = np.zeros((len(det_boxes), len(gt_boxes)))
    for i, det in enumerate(det_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(det, gt)

    tp_indices = []
    fp_indices = []
    matched_gt = set()

    # Greedy matching: highest IoU first
    while True:
        if iou_matrix.size == 0:
            break
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        det_idx, gt_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        tp_indices.append(det_idx)
        matched_gt.add(gt_idx)
        iou_matrix[det_idx, :] = 0
        iou_matrix[:, gt_idx] = 0

    for i in range(len(det_boxes)):
        if i not in tp_indices:
            fp_indices.append(i)

    return tp_indices, fp_indices, list(matched_gt)


def compute_ap(precisions, recalls):
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= t]
        if p_at_r:
            ap += max(p_at_r)
    return ap / 11.0


def evaluate_detections(all_det_boxes, all_det_confs, all_gt_boxes, iou_threshold=0.5):
    """
    Compute detection metrics across all images.

    Returns dict with precision, recall, f1, mAP50, false_positive_rate.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0

    # For mAP computation: collect all detections with their TP/FP labels
    all_scores = []
    all_is_tp = []

    total_tp_at_conf = 0
    total_fp_at_conf = 0

    for img_key in all_gt_boxes:
        gt = all_gt_boxes[img_key]
        det = all_det_boxes.get(img_key, np.zeros((0, 4)))
        confs = all_det_confs.get(img_key, np.zeros(0))

        total_gt += len(gt)

        if len(det) == 0:
            total_fn += len(gt)
            continue

        tp_idx, fp_idx, matched_gt = match_detections_to_gt(det, gt, iou_threshold)
        total_tp += len(tp_idx)
        total_fp += len(fp_idx)
        total_fn += len(gt) - len(matched_gt)

        for i in tp_idx:
            all_scores.append(float(confs[i]))
            all_is_tp.append(True)
            if float(confs[i]) >= 0.25:
                total_tp_at_conf += 1
                
        for i in fp_idx:
            all_scores.append(float(confs[i]))
            all_is_tp.append(False)
            if float(confs[i]) >= 0.25:
                total_fp_at_conf += 1

    # Precision, Recall, F1 at conf=0.25
    precision = total_tp_at_conf / (total_tp_at_conf + total_fp_at_conf) if (total_tp_at_conf + total_fp_at_conf) > 0 else 0
    recall = total_tp_at_conf / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # mAP@0.5 (simple AP computation)
    # Sort by confidence descending
    sorted_indices = np.argsort(-np.array(all_scores))
    sorted_tp = [all_is_tp[i] for i in sorted_indices]

    cum_tp = np.cumsum(sorted_tp).astype(float)
    cum_fp = np.cumsum([not t for t in sorted_tp]).astype(float)

    precisions = cum_tp / (cum_tp + cum_fp)
    recalls = cum_tp / total_gt if total_gt > 0 else cum_tp

    # All-points interpolation AP
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[1.0], precisions, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = np.where(mrec[1:] != mrec[:-1])[0] + 1
    mAP50 = np.sum((mrec[i_list] - mrec[i_list - 1]) * mpre[i_list])

    fpr = total_fp_at_conf / (total_tp_at_conf + total_fp_at_conf) if (total_tp_at_conf + total_fp_at_conf) > 0 else 0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mAP50": float(mAP50),
        "false_positive_rate": float(fpr),
        "total_tp": int(total_tp_at_conf),
        "total_fp": int(total_fp_at_conf),
        "total_fn": int(total_gt - total_tp_at_conf),
        "total_gt": int(total_gt),
        "total_detections": int(total_tp_at_conf + total_fp_at_conf),
    }


def save_visualization(
    image: np.ndarray,
    gt_boxes_xyxy: list,
    baseline_boxes: np.ndarray,
    baseline_confs: np.ndarray,
    filtered_boxes: np.ndarray,
    filtered_confs: np.ndarray,
    acm_results: list,
    output_path: Path,
):
    """Save a side-by-side visualization of baseline vs ACM-filtered detections."""

    h, w = image.shape[:2]
    if len(image.shape) == 2:
        vis_base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        vis_acm = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_base = image.copy()
        vis_acm = image.copy()

    # Draw GT boxes (blue, thin)
    for gt in gt_boxes_xyxy:
        x1, y1, x2, y2 = [int(v) for v in gt]
        cv2.rectangle(vis_base, (x1, y1), (x2, y2), (255, 150, 0), 1)
        cv2.rectangle(vis_acm, (x1, y1), (x2, y2), (255, 150, 0), 1)

    # Baseline: all detections in green
    for i in range(len(baseline_boxes)):
        x1, y1, x2, y2 = [int(v) for v in baseline_boxes[i]]
        conf = baseline_confs[i]
        cv2.rectangle(vis_base, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_base, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # ACM-filtered: kept in green, discarded in red
    for r in acm_results:
        x1, y1, x2, y2 = [int(v) for v in r.bbox]
        if r.kept:
            color = (0, 255, 0)
            label = f"{r.adjusted_conf:.2f}"
        else:
            color = (0, 0, 255)
            label = f"X:{r.discard_reason[:15]}"
        cv2.rectangle(vis_acm, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_acm, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Labels
    cv2.putText(vis_base, "BASELINE (YOLOv8n)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis_acm, "ACM FILTERED", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Concatenate side by side
    combined = np.hstack([vis_base, vis_acm])
    cv2.imwrite(str(output_path), combined)


# --- Main -------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("YOLOv8n + ACM Post-Processing Pipeline")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Load model
    print("\n[1/6] Loading YOLOv8n model...")
    model = YOLO(args.model)

    # Create ACM filter
    print("[2/6] Initializing ACM filter (Morphological Chan-Vese)...")
    acm = ACMFilter(
        num_iter=args.num_iter,
        smoothing=args.smoothing,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        bbox_padding=args.bbox_padding,
        median_filter_size=args.median_filter_size,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        max_compactness=args.max_compactness,
        min_elongation=args.min_elongation,
        min_contrast_ratio=args.min_contrast_ratio,
    )
    print(f"  Parameters: num_iter={args.num_iter}, smoothing={args.smoothing}, "
          f"lambda1={args.lambda1}, lambda2={args.lambda2}")
    print(f"  Thresholds: max_area={args.max_area_ratio}, "
          f"max_compact={args.max_compactness}, min_elong={args.min_elongation}")

    # Load ground truth
    print("[3/6] Loading ground truth labels...")
    gt_data = load_ground_truth(TEST_LABELS)
    print(f"  Loaded GT for {len(gt_data)} images")

    # Get test image list
    test_images = sorted(TEST_IMAGES.glob("*.png"))
    print(f"  Found {len(test_images)} test images")

    # Run pipeline
    print("\n[4/6] Running YOLOv8n + ACM pipeline on test set...")

    baseline_det_boxes = {}    # image_key -> np.array of [x1,y1,x2,y2]
    baseline_det_confs = {}
    acm_det_boxes = {}
    acm_det_confs = {}
    gt_boxes_xyxy_all = {}

    # Statistics
    total_baseline_dets = 0
    total_acm_dets = 0
    total_discarded = 0
    discard_reasons = defaultdict(int)
    acm_times = []

    # Per-image stats for CSV
    per_image_stats = []

    for idx, img_path in enumerate(test_images):
        stem = img_path.stem

        # Read image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        img_h, img_w = image.shape

        results = model.predict(
            source=str(img_path),
            conf=0.05,  # low conf to allow ACM to score and potentially boost borderline detections
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )

        r = results[0]
        if len(r.boxes) == 0:
            baseline_det_boxes[stem] = np.zeros((0, 4))
            baseline_det_confs[stem] = np.zeros(0)
            acm_det_boxes[stem] = np.zeros((0, 4))
            acm_det_confs[stem] = np.zeros(0)

            # GT boxes
            gt_yolo = gt_data.get(stem, np.zeros((0, 4)))
            gt_xyxy = [yolo_to_xyxy(b, img_w, img_h) for b in gt_yolo]
            gt_boxes_xyxy_all[stem] = np.array(gt_xyxy) if gt_xyxy else np.zeros((0, 4))

            per_image_stats.append({
                "image": stem,
                "baseline_dets": 0, "acm_dets": 0, "discarded": 0,
                "gt_count": len(gt_yolo),
            })
            continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        baseline_det_boxes[stem] = boxes_xyxy
        baseline_det_confs[stem] = confs
        total_baseline_dets += len(boxes_xyxy)

        # ACM filtering
        t0 = time.time()
        filt_boxes, filt_confs, _, acm_results = apply_acm_filter(
            image, boxes_xyxy, confs, acm_filter=acm,
            return_masks=(idx < args.save_viz)
        )
        acm_time = time.time() - t0
        acm_times.append(acm_time)

        acm_det_boxes[stem] = filt_boxes
        acm_det_confs[stem] = filt_confs
        total_acm_dets += len(filt_boxes)

        img_discarded = sum(1 for r_item in acm_results if r_item.adjusted_conf < 0.25 and r_item.original_conf >= 0.25)
        total_discarded += img_discarded
        for r_item in acm_results:
            if not r_item.kept:
                reason_key = r_item.discard_reason.split("(")[0].strip()
                discard_reasons[reason_key] += 1

        # GT boxes (convert to xyxy)
        gt_yolo = gt_data.get(stem, np.zeros((0, 4)))
        gt_xyxy = [yolo_to_xyxy(b, img_w, img_h) for b in gt_yolo]
        gt_boxes_xyxy_all[stem] = np.array(gt_xyxy) if gt_xyxy else np.zeros((0, 4))

        per_image_stats.append({
            "image": stem,
            "baseline_dets": len(boxes_xyxy),
            "acm_dets": len(filt_boxes),
            "discarded": img_discarded,
            "gt_count": len(gt_yolo),
            "acm_time_ms": f"{acm_time * 1000:.1f}",
        })

        # Save visualization
        if idx < args.save_viz:
            save_visualization(
                image, gt_xyxy,
                boxes_xyxy, confs,
                filt_boxes, filt_confs,
                acm_results,
                viz_dir / f"viz_{stem}.png",
            )

        # Progress
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(test_images)} images "
                  f"(avg ACM: {np.mean(acm_times[-200:])*1000:.1f}ms)")

    print(f"\n  Total: {len(test_images)} images processed")
    print(f"  Avg ACM time: {np.mean(acm_times)*1000:.1f}ms per image")

    # --- Evaluate ---
    print("\n[5/6] Computing metrics...")

    baseline_metrics = evaluate_detections(
        baseline_det_boxes, baseline_det_confs, gt_boxes_xyxy_all, iou_threshold=0.5
    )
    acm_metrics = evaluate_detections(
        acm_det_boxes, acm_det_confs, gt_boxes_xyxy_all, iou_threshold=0.5
    )

    # Also run YOLO's built-in val for mAP50-95
    print("\n  Running YOLO built-in validation for mAP@0.5:0.95...")
    yolo_val = model.val(
        data=str(DATASET_YAML), split="test",
        batch=16, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        verbose=False,
    )
    baseline_metrics["mAP50_yolo"] = float(yolo_val.box.map50)
    baseline_metrics["mAP50_95_yolo"] = float(yolo_val.box.map)

    # --- Save results ---
    print("\n[6/6] Saving results...")

    # Filter analysis
    filter_analysis = {
        "total_baseline_detections": total_baseline_dets,
        "total_acm_detections": total_acm_dets,
        "total_discarded": total_discarded,
        "discard_rate": total_discarded / total_baseline_dets if total_baseline_dets > 0 else 0,
        "discard_reasons": dict(discard_reasons),
        "avg_acm_time_ms": float(np.mean(acm_times) * 1000),
    }

    # Analyze TP vs FP discards
    tp_discarded = 0
    fp_discarded = 0
    for stem in baseline_det_boxes:
        gt = gt_boxes_xyxy_all.get(stem, np.zeros((0, 4)))
        base = baseline_det_boxes[stem]
        base_confs = baseline_det_confs[stem]
        acm_kept = acm_det_boxes[stem]

        if len(base) == 0:
            continue

        # Find which baseline detections were TP/FP
        tp_idx, fp_idx, _ = match_detections_to_gt(base, gt)

        # Check which were discarded by ACM
        kept_set = set()
        for i in range(len(acm_kept)):
            for j in range(len(base)):
                if compute_iou(acm_kept[i], base[j]) > 0.9:
                    kept_set.add(j)
                    break

        for idx_val in tp_idx:
            if idx_val not in kept_set:
                tp_discarded += 1
        for idx_val in fp_idx:
            if idx_val not in kept_set:
                fp_discarded += 1

    filter_analysis["tp_discarded"] = tp_discarded
    filter_analysis["fp_discarded"] = fp_discarded
    filter_analysis["fp_discard_precision"] = (
        fp_discarded / total_discarded if total_discarded > 0 else 0
    )

    # Save comprehensive JSON
    results_json = {
        "acm_parameters": {
            "num_iter": args.num_iter,
            "smoothing": args.smoothing,
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
            "bbox_padding": args.bbox_padding,
            "median_filter_size": args.median_filter_size,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
            "max_compactness": args.max_compactness,
            "min_elongation": args.min_elongation,
            "min_contrast_ratio": args.min_contrast_ratio,
        },
        "baseline_metrics": baseline_metrics,
        "acm_metrics": acm_metrics,
        "filter_analysis": filter_analysis,
        "improvement": {
            "precision_delta": acm_metrics["precision"] - baseline_metrics["precision"],
            "recall_delta": acm_metrics["recall"] - baseline_metrics["recall"],
            "f1_delta": acm_metrics["f1"] - baseline_metrics["f1"],
            "mAP50_delta": acm_metrics["mAP50"] - baseline_metrics["mAP50"],
            "fpr_delta": acm_metrics["false_positive_rate"] - baseline_metrics["false_positive_rate"],
            "fpr_reduction_pct": (
                (baseline_metrics["false_positive_rate"] - acm_metrics["false_positive_rate"])
                / baseline_metrics["false_positive_rate"] * 100
                if baseline_metrics["false_positive_rate"] > 0 else 0
            ),
        },
    }

    json_path = OUTPUT_DIR / "acm_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Results JSON: {json_path}")

    # Per-image CSV
    csv_path = OUTPUT_DIR / "acm_per_image.csv"
    if per_image_stats:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_image_stats[0].keys())
            writer.writeheader()
            writer.writerows(per_image_stats)
        print(f"  Per-image CSV: {csv_path}")

    # --- Print comparison table ---
    print("\n" + "=" * 70)
    print("COMPARISON: BASELINE vs ACM-FILTERED")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Baseline':>12} {'ACM':>12} {'Delta':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    for key, label in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-Score"),
        ("mAP50", "mAP@0.5"),
        ("false_positive_rate", "False Positive Rate"),
    ]:
        b = baseline_metrics[key]
        a = acm_metrics[key]
        d = a - b
        sign = "+" if d > 0 else ""
        # For FPR, negative delta is good
        print(f"  {label:<25} {b:>12.4f} {a:>12.4f} {sign}{d:>11.4f}")

    print(f"\n  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Total Detections':<25} {baseline_metrics['total_detections']:>12d} "
          f"{acm_metrics['total_detections']:>12d} "
          f"{acm_metrics['total_detections'] - baseline_metrics['total_detections']:>12d}")
    print(f"  {'True Positives':<25} {baseline_metrics['total_tp']:>12d} "
          f"{acm_metrics['total_tp']:>12d}")
    print(f"  {'False Positives':<25} {baseline_metrics['total_fp']:>12d} "
          f"{acm_metrics['total_fp']:>12d}")

    print(f"\n  Filter Analysis:")
    print(f"    Detections discarded:    {total_discarded} / {total_baseline_dets} "
          f"({total_discarded/total_baseline_dets*100:.1f}%)")
    print(f"    FPs discarded:           {fp_discarded}")
    print(f"    TPs discarded (lost):    {tp_discarded}")
    print(f"    Filter precision:        {filter_analysis['fp_discard_precision']:.3f} "
          f"(fraction of discards that were FPs)")
    print(f"    Avg ACM time:            {np.mean(acm_times)*1000:.1f}ms per image")

    print(f"\n  FPR Reduction: {results_json['improvement']['fpr_reduction_pct']:.1f}%")
    print("=" * 70)

    print(f"\n  Visualizations saved to: {viz_dir}")
    print(f"  Full results saved to: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
