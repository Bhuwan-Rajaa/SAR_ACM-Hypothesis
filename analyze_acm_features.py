"""
analyze_acm_features.py
=======================
Extracts ACM features for YOLO detections, matches them against Ground Truth,
and saves a CSV to analyze feature distributions for TPs vs FPs.
This helps find optimal thresholds for the ACM filter.
"""
import argparse
import numpy as np
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO
from acm_postprocess import ACMFilter
from run_acm_pipeline import load_ground_truth, yolo_to_xyxy, match_detections_to_gt

MODEL_PATH = Path(r"d:\CAPSTONE_HYPOTHESIS\runs\baseline\weights\best.pt")
TEST_IMAGES = Path(r"d:\CAPSTONE_HYPOTHESIS\hrsid_yolo\test\images")
TEST_LABELS = Path(r"d:\CAPSTONE_HYPOTHESIS\hrsid_yolo\test\labels")
OUTPUT_CSV = Path(r"d:\CAPSTONE_HYPOTHESIS\evaluation_results\acm\feature_analysis.csv")

def main():
    print("Loading model and data...")
    model = YOLO(MODEL_PATH)
    gt_data = load_ground_truth(TEST_LABELS)
    test_images = sorted(TEST_IMAGES.glob("*.png"))[:500] # Use 500 images for speed

    # Initialize ACM with large padding to ensure good background estimation
    acm = ACMFilter(
        bbox_padding=0.5,
        conf_gate=1.0, # Run ACM on ALL detections
        min_contrast_ratio=0.0, # Don't filter anything yet
        min_area_ratio=0.0,
        max_area_ratio=1.0,
    )

    records = []

    print(f"Processing {len(test_images)} images...")
    for idx, img_path in enumerate(test_images):
        stem = img_path.stem
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None: continue
        img_h, img_w = image.shape

        results = model.predict(source=str(img_path), conf=0.1, iou=0.5, imgsz=640, verbose=False) # Lower conf to get more FPs
        r = results[0]
        if len(r.boxes) == 0: continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        
        gt_yolo = gt_data.get(stem, np.zeros((0, 4)))
        gt_xyxy = [yolo_to_xyxy(b, img_w, img_h) for b in gt_yolo]
        gt_boxes = np.array(gt_xyxy) if gt_xyxy else np.zeros((0, 4))

        tp_idx, fp_idx, _ = match_detections_to_gt(boxes_xyxy, gt_boxes, iou_threshold=0.5)
        
        # Run ACM feature extraction manually to get all features
        gray = image.astype(np.float64) / image.max()
        for i, bbox in enumerate(boxes_xyxy):
            is_tp = i in tp_idx
            
            # Process with ACM
            res = acm._process_single_detection(gray, bbox, confs[i], img_h, img_w, return_mask=False)
            feats = res.shape_features
            
            records.append({
                'is_tp': int(is_tp),
                'conf': confs[i],
                'is_valid': int(feats.is_valid),
                'area_ratio': feats.area_ratio,
                'compactness': feats.compactness,
                'solidity': feats.solidity,
                'elongation': feats.elongation,
                'contrast': feats.contrast_ratio,
                'segment_area': feats.segment_area_px
            })
            
        if (idx+1) % 50 == 0:
            print(f"  Done {idx+1}/{len(test_images)}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if records:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    print(f"\nSaved features for {len(records)} detections to {OUTPUT_CSV}")
    
    # Quick analysis
    print("\n--- Feature Analysis ---")
    valid_records = [r for r in records if r['is_valid'] == 1]
    tp_records = [r for r in valid_records if r['is_tp'] == 1]
    fp_records = [r for r in valid_records if r['is_tp'] == 0]
    
    print(f"Total TPs analyzed: {len(tp_records)}")
    print(f"Total FPs analyzed: {len(fp_records)}")
    
    features_to_check = ['area_ratio', 'compactness', 'solidity', 'elongation', 'contrast']
    for f in features_to_check:
        tp_vals = [r[f] for r in tp_records if not np.isinf(r[f]) and not np.isnan(r[f])]
        fp_vals = [r[f] for r in fp_records if not np.isinf(r[f]) and not np.isnan(r[f])]
        
        print(f"\n{f.upper()}:")
        if tp_vals:
            print(f"  TP: mean={np.mean(tp_vals):.3f}, median={np.median(tp_vals):.3f}, 5th%={np.percentile(tp_vals, 5):.3f}, 95th%={np.percentile(tp_vals, 95):.3f}")
        if fp_vals:
            print(f"  FP: mean={np.mean(fp_vals):.3f}, median={np.median(fp_vals):.3f}, 5th%={np.percentile(fp_vals, 5):.3f}, 95th%={np.percentile(fp_vals, 95):.3f}")

if __name__ == '__main__':
    main()
