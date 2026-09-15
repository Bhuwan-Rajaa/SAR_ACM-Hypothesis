"""
convert_hrsid.py
================
Convert HRSID dataset from COCO JSON format to YOLO detection format.

Creates:
  hrsid_yolo/
  ├── train/images/  train/labels/
  ├── val/images/    val/labels/
  ├── test/images/   test/labels/
  └── hrsid.yaml

Usage:
  python convert_hrsid.py
"""

import json
import os
import shutil
import random
import yaml
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────

HRSID_ROOT = Path(r"d:\CAPSTONE_HYPOTHESIS\hrsid")
OUTPUT_ROOT = Path(r"d:\CAPSTONE_HYPOTHESIS\hrsid_yolo")
IMAGES_DIR = HRSID_ROOT / "images"
TRAIN_JSON = HRSID_ROOT / "annotations" / "train2017.json"
TEST_JSON = HRSID_ROOT / "annotations" / "test2017.json"

VAL_SPLIT_RATIO = 0.15  # 15% of training set for validation
RANDOM_SEED = 42
NUM_SPOT_CHECKS = 5  # Number of images to visualize for verification


# ─── Helper Functions ────────────────────────────────────────────────────────

def load_coco_json(json_path: Path) -> dict:
    """Load a COCO-format JSON annotation file."""
    print(f"  Loading {json_path.name}...")
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"  -> {len(data['images'])} images, {len(data['annotations'])} annotations")
    return data


def coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox [x_min, y_min, width, height] (absolute pixels)
    to YOLO format [cx, cy, w, h] (normalized 0-1).
    """
    x_min, y_min, w, h = bbox
    cx = (x_min + w / 2.0) / img_width
    cy = (y_min + h / 2.0) / img_height
    nw = w / img_width
    nh = h / img_height
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def build_image_annotation_map(coco_data: dict) -> dict:
    """
    Build a mapping from image_id -> list of YOLO-format annotation strings.
    Also returns image_id -> image_info mapping.
    """
    # Build image_id -> image_info
    id_to_info = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        img_info = id_to_info[ann["image_id"]]
        img_w = img_info["width"]
        img_h = img_info["height"]
        # Single class (ship) -> class_id = 0
        cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
        # Skip degenerate boxes
        if nw <= 0 or nh <= 0:
            continue
        annotations_by_image[ann["image_id"]].append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return id_to_info, annotations_by_image


def write_labels_and_copy_images(
    image_ids: list,
    id_to_info: dict,
    annotations_by_image: dict,
    images_src: Path,
    images_dst: Path,
    labels_dst: Path,
):
    """Write YOLO label files and copy images to the target split directory."""
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for img_id in image_ids:
        info = id_to_info[img_id]
        filename = info["file_name"]
        stem = Path(filename).stem

        # Copy image
        src_path = images_src / filename
        if not src_path.exists():
            skipped += 1
            continue

        dst_img_path = images_dst / filename
        if not dst_img_path.exists():
            shutil.copy2(src_path, dst_img_path)

        # Write label file (even if empty — means no ships in this image)
        label_path = labels_dst / f"{stem}.txt"
        lines = annotations_by_image.get(img_id, [])
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")  # trailing newline

        copied += 1

    return copied, skipped


def generate_yaml(output_root: Path):
    """Generate the YOLO dataset YAML config file."""
    config = {
        "path": str(output_root).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "ship"},
        "nc": 1,
    }
    yaml_path = output_root / "hrsid.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Dataset YAML written to: {yaml_path}")
    return yaml_path


def visualize_spot_checks(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    num_samples: int = 5,
):
    """
    Pick random images and draw their YOLO bounding boxes
    to visually verify the conversion is correct.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = list(labels_dir.glob("*.txt"))
    # Filter to non-empty label files for more interesting visualizations
    non_empty = [f for f in label_files if f.stat().st_size > 0]
    if len(non_empty) < num_samples:
        non_empty = label_files
    samples = random.sample(non_empty, min(num_samples, len(non_empty)))

    print(f"\n  Generating {len(samples)} spot-check visualizations...")
    for label_file in samples:
        stem = label_file.stem
        # Find matching image (try common extensions)
        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tif"]:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # If grayscale, convert to BGR for colored boxes
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Read and draw YOLO labels
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, "ship", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = output_dir / f"spot_check_{stem}.png"
        cv2.imwrite(str(out_path), img)
        print(f"    Saved: {out_path.name}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    print("=" * 60)
    print("HRSID COCO -> YOLO Format Converter")
    print("=" * 60)

    # ── Step 1: Load COCO annotations ──
    print("\n[1/5] Loading COCO annotations...")
    train_coco = load_coco_json(TRAIN_JSON)
    test_coco = load_coco_json(TEST_JSON)

    # ── Step 2: Build annotation maps ──
    print("\n[2/5] Building annotation maps...")
    train_id_to_info, train_anns = build_image_annotation_map(train_coco)
    test_id_to_info, test_anns = build_image_annotation_map(test_coco)

    # ── Step 3: Split training into train + val ──
    print("\n[3/5] Splitting training set (85/15)...")
    all_train_ids = list(train_id_to_info.keys())
    random.shuffle(all_train_ids)
    val_count = int(len(all_train_ids) * VAL_SPLIT_RATIO)
    val_ids = all_train_ids[:val_count]
    train_ids = all_train_ids[val_count:]

    # Count annotations
    train_ann_count = sum(len(train_anns.get(i, [])) for i in train_ids)
    val_ann_count = sum(len(train_anns.get(i, [])) for i in val_ids)
    test_ids = list(test_id_to_info.keys())
    test_ann_count = sum(len(test_anns.get(i, [])) for i in test_ids)

    print(f"  Train: {len(train_ids)} images, {train_ann_count} annotations")
    print(f"  Val:   {len(val_ids)} images, {val_ann_count} annotations")
    print(f"  Test:  {len(test_ids)} images, {test_ann_count} annotations")

    # ── Step 4: Write output ──
    print("\n[4/5] Writing YOLO-format labels and copying images...")

    # Clean output directory
    if OUTPUT_ROOT.exists():
        print(f"  Removing existing output: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    # Train split
    print("  Processing train split...")
    copied, skipped = write_labels_and_copy_images(
        train_ids, train_id_to_info, train_anns,
        IMAGES_DIR, OUTPUT_ROOT / "train" / "images", OUTPUT_ROOT / "train" / "labels"
    )
    print(f"    Copied: {copied}, Skipped: {skipped}")

    # Val split
    print("  Processing val split...")
    copied, skipped = write_labels_and_copy_images(
        val_ids, train_id_to_info, train_anns,
        IMAGES_DIR, OUTPUT_ROOT / "val" / "images", OUTPUT_ROOT / "val" / "labels"
    )
    print(f"    Copied: {copied}, Skipped: {skipped}")

    # Test split
    print("  Processing test split...")
    copied, skipped = write_labels_and_copy_images(
        test_ids, test_id_to_info, test_anns,
        IMAGES_DIR, OUTPUT_ROOT / "test" / "images", OUTPUT_ROOT / "test" / "labels"
    )
    print(f"    Copied: {copied}, Skipped: {skipped}")

    # Generate YAML
    yaml_path = generate_yaml(OUTPUT_ROOT)

    # ── Step 5: Spot-check visualizations ──
    print("\n[5/5] Generating spot-check visualizations...")
    visualize_spot_checks(
        OUTPUT_ROOT / "train" / "images",
        OUTPUT_ROOT / "train" / "labels",
        OUTPUT_ROOT / "spot_checks",
        num_samples=NUM_SPOT_CHECKS,
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_ROOT}")
    print(f"  Dataset config:   {yaml_path}")
    print(f"  Spot checks:      {OUTPUT_ROOT / 'spot_checks'}")
    print(f"\n  Train: {len(train_ids)} images")
    print(f"  Val:   {len(val_ids)} images")
    print(f"  Test:  {len(test_ids)} images")
    print(f"  Total: {len(train_ids) + len(val_ids) + len(test_ids)} images")

    # Verification assertions
    total_label_files = (
        len(list((OUTPUT_ROOT / "train" / "labels").glob("*.txt"))) +
        len(list((OUTPUT_ROOT / "val" / "labels").glob("*.txt"))) +
        len(list((OUTPUT_ROOT / "test" / "labels").glob("*.txt")))
    )
    total_expected = len(train_ids) + len(val_ids) + len(test_ids)
    assert total_label_files == total_expected, (
        f"Label count mismatch: {total_label_files} files vs {total_expected} expected"
    )
    print(f"\n  [OK] Verification passed: {total_label_files} label files generated")


if __name__ == "__main__":
    main()
