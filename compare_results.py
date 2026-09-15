"""
compare_results.py
==================
Generate comparative analysis between baseline and ACM-filtered results.

Creates:
- Comparison summary table
- Bar charts of key metrics
- Analysis of what types of detections ACM filters

Usage:
    python compare_results.py
"""

import json
import numpy as np
import cv2
from pathlib import Path


# --- Configuration ----------------------------------------------------------

BASELINE_METRICS = Path(r"d:\CAPSTONE_HYPOTHESIS\evaluation_results\baseline\baseline_metrics.json")
ACM_RESULTS = Path(r"d:\CAPSTONE_HYPOTHESIS\evaluation_results\acm\acm_results.json")
OUTPUT_DIR = Path(r"d:\CAPSTONE_HYPOTHESIS\evaluation_results\comparison")


def create_comparison_chart(baseline, acm, output_path):
    """Create a bar chart comparing baseline vs ACM metrics."""

    metrics = ["Precision", "Recall", "F1", "mAP@0.5"]
    baseline_vals = [
        baseline["precision"], baseline["recall"],
        baseline["f1"], baseline["mAP50"],
    ]
    acm_vals = [
        acm["precision"], acm["recall"],
        acm["f1"], acm["mAP50"],
    ]

    # Create chart with OpenCV (no matplotlib dependency)
    chart_w, chart_h = 800, 500
    chart = np.ones((chart_h, chart_w, 3), dtype=np.uint8) * 255

    # Colors
    base_color = (180, 120, 60)   # Blue-ish
    acm_color = (60, 180, 60)     # Green
    text_color = (40, 40, 40)

    # Title
    cv2.putText(chart, "Baseline vs ACM-Filtered Performance",
                (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    # Bar dimensions
    bar_w = 60
    gap = 30
    group_w = 2 * bar_w + gap
    total_w = len(metrics) * group_w + (len(metrics) - 1) * 40
    start_x = (chart_w - total_w) // 2
    bar_bottom = 420
    bar_max_h = 300

    for i, (name, bv, av) in enumerate(zip(metrics, baseline_vals, acm_vals)):
        x_offset = start_x + i * (group_w + 40)

        # Baseline bar
        bh = int(bv * bar_max_h)
        cv2.rectangle(chart, (x_offset, bar_bottom - bh),
                       (x_offset + bar_w, bar_bottom), base_color, -1)
        cv2.putText(chart, f"{bv:.3f}", (x_offset, bar_bottom - bh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, base_color, 1)

        # ACM bar
        ah = int(av * bar_max_h)
        cv2.rectangle(chart, (x_offset + bar_w + gap, bar_bottom - ah),
                       (x_offset + bar_w + gap + bar_w, bar_bottom), acm_color, -1)
        cv2.putText(chart, f"{av:.3f}",
                    (x_offset + bar_w + gap, bar_bottom - ah - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, acm_color, 1)

        # Label
        cv2.putText(chart, name, (x_offset, bar_bottom + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Legend
    cv2.rectangle(chart, (50, 460), (80, 480), base_color, -1)
    cv2.putText(chart, "Baseline", (90, 476), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.rectangle(chart, (200, 460), (230, 480), acm_color, -1)
    cv2.putText(chart, "ACM-Filtered", (240, 476), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    cv2.imwrite(str(output_path), chart)
    print(f"  Chart saved: {output_path}")


def create_fpr_chart(baseline_fpr, acm_fpr, output_path):
    """Create a focused FPR reduction chart."""

    chart_w, chart_h = 500, 400
    chart = np.ones((chart_h, chart_w, 3), dtype=np.uint8) * 255

    text_color = (40, 40, 40)
    base_color = (0, 0, 200)   # Red for FPR
    acm_color = (0, 160, 0)    # Green for improved FPR

    cv2.putText(chart, "False Positive Rate Reduction",
                (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    bar_w = 100
    bar_bottom = 320
    bar_max_h = 250
    center_x = chart_w // 2

    # Baseline FPR bar
    bh = int(baseline_fpr * bar_max_h * 10)  # Scale up for visibility
    x1 = center_x - bar_w - 30
    cv2.rectangle(chart, (x1, bar_bottom - bh),
                   (x1 + bar_w, bar_bottom), base_color, -1)
    cv2.putText(chart, f"{baseline_fpr*100:.1f}%",
                (x1 + 15, bar_bottom - bh - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, base_color, 2)
    cv2.putText(chart, "Baseline", (x1 + 10, bar_bottom + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # ACM FPR bar
    ah = int(acm_fpr * bar_max_h * 10)
    x2 = center_x + 30
    cv2.rectangle(chart, (x2, bar_bottom - ah),
                   (x2 + bar_w, bar_bottom), acm_color, -1)
    cv2.putText(chart, f"{acm_fpr*100:.1f}%",
                (x2 + 15, bar_bottom - ah - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, acm_color, 2)
    cv2.putText(chart, "ACM", (x2 + 25, bar_bottom + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Reduction annotation
    reduction = (baseline_fpr - acm_fpr) / baseline_fpr * 100 if baseline_fpr > 0 else 0
    cv2.putText(chart, f"Reduction: {reduction:.1f}%",
                (center_x - 80, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 0), 2)

    cv2.imwrite(str(output_path), chart)
    print(f"  FPR chart saved: {output_path}")


def main():
    print("=" * 60)
    print("Baseline vs ACM Comparison Analysis")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    if not ACM_RESULTS.exists():
        print(f"ERROR: ACM results not found at {ACM_RESULTS}")
        print("Run run_acm_pipeline.py first.")
        return

    with open(ACM_RESULTS, "r") as f:
        acm_data = json.load(f)

    baseline = acm_data["baseline_metrics"]
    acm = acm_data["acm_metrics"]
    improvement = acm_data["improvement"]
    analysis = acm_data["filter_analysis"]

    # Print detailed comparison
    print("\n--- Detection Metrics ---")
    print(f"  {'Metric':<25} {'Baseline':>10} {'ACM':>10} {'Delta':>10} {'Change':>10}")
    print(f"  {'-'*65}")

    for key, label in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-Score"),
        ("mAP50", "mAP@0.5"),
        ("false_positive_rate", "False Positive Rate"),
    ]:
        b = baseline[key]
        a = acm[key]
        d = a - b
        pct = (d / b * 100) if b != 0 else 0
        sign = "+" if d > 0 else ""
        print(f"  {label:<25} {b:>10.4f} {a:>10.4f} {sign}{d:>9.4f} {sign}{pct:>8.1f}%")

    print(f"\n--- Filter Analysis ---")
    print(f"  Total baseline detections: {analysis['total_baseline_detections']}")
    print(f"  Total ACM detections:      {analysis['total_acm_detections']}")
    print(f"  Discarded:                 {analysis['total_discarded']} "
          f"({analysis['discard_rate']*100:.1f}%)")
    print(f"  FPs discarded:             {analysis.get('fp_discarded', 'N/A')}")
    print(f"  TPs discarded (lost):      {analysis.get('tp_discarded', 'N/A')}")
    print(f"  Filter precision:          "
          f"{analysis.get('fp_discard_precision', 0):.1%} "
          f"(fraction of discards that were FPs)")

    if "discard_reasons" in analysis:
        print(f"\n--- Discard Reasons ---")
        for reason, count in sorted(analysis["discard_reasons"].items(),
                                     key=lambda x: -x[1]):
            pct = count / analysis["total_discarded"] * 100 if analysis["total_discarded"] > 0 else 0
            print(f"  {reason:<40} {count:>6} ({pct:>5.1f}%)")

    # Generate charts
    print(f"\n--- Generating charts ---")
    create_comparison_chart(baseline, acm, OUTPUT_DIR / "comparison_chart.png")
    create_fpr_chart(
        baseline["false_positive_rate"],
        acm["false_positive_rate"],
        OUTPUT_DIR / "fpr_reduction_chart.png",
    )

    # Save summary markdown
    summary_md = f"""# Baseline vs ACM Comparison

## Key Results

| Metric | Baseline | ACM | Delta |
|--------|----------|-----|-------|
| Precision | {baseline['precision']:.4f} | {acm['precision']:.4f} | {improvement['precision_delta']:+.4f} |
| Recall | {baseline['recall']:.4f} | {acm['recall']:.4f} | {improvement['recall_delta']:+.4f} |
| F1-Score | {baseline['f1']:.4f} | {acm['f1']:.4f} | {improvement['f1_delta']:+.4f} |
| mAP@0.5 | {baseline['mAP50']:.4f} | {acm['mAP50']:.4f} | {improvement['mAP50_delta']:+.4f} |
| FPR | {baseline['false_positive_rate']:.4f} | {acm['false_positive_rate']:.4f} | {improvement['fpr_delta']:+.4f} |

## FPR Reduction: {improvement['fpr_reduction_pct']:.1f}%

## Filter Analysis
- Discarded {analysis['total_discarded']} / {analysis['total_baseline_detections']} detections ({analysis['discard_rate']*100:.1f}%)
- FPs discarded: {analysis.get('fp_discarded', 'N/A')}
- TPs lost: {analysis.get('tp_discarded', 'N/A')}
- Filter precision: {analysis.get('fp_discard_precision', 0):.1%}
"""
    summary_path = OUTPUT_DIR / "comparison_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary_md)
    print(f"  Summary saved: {summary_path}")

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
