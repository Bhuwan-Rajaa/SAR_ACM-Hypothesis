"""
train_baseline.py
=================
Train YOLOv8n baseline for ship detection on HRSID dataset.

Optimized for RTX 4050 Laptop GPU (6GB VRAM).

Usage:
  python train_baseline.py
  python train_baseline.py --epochs 50 --batch 8  # override defaults
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


# --- Configuration -----------------------------------------------------------

DATASET_YAML = Path(r"d:\CAPSTONE_HYPOTHESIS\hrsid_yolo\hrsid.yaml")
PROJECT_DIR = Path(r"d:\CAPSTONE_HYPOTHESIS\runs")
EXPERIMENT_NAME = "baseline"


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8n on HRSID")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Pretrained model to start from (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16, safe for 6GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--name", type=str, default=EXPERIMENT_NAME,
                        help="Experiment name (default: baseline)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("YOLOv8n Ship Detection - Baseline Training")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {DATASET_YAML}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Patience:   {args.patience}")
    print(f"  Workers:    {args.workers}")
    print(f"  Project:    {PROJECT_DIR}")
    print(f"  Name:       {args.name}")
    print("=" * 60)

    # Verify dataset config exists
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATASET_YAML}\n"
            "Run convert_hrsid.py first to create the YOLO-format dataset."
        )

    # Load model
    if args.resume:
        # Resume from last checkpoint
        last_ckpt = PROJECT_DIR / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found at: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        print(f"\nResuming from: {last_ckpt}")
    else:
        model = YOLO(args.model)
        print(f"\nLoaded pretrained model: {args.model}")

    # Train
    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        workers=args.workers,
        project=str(PROJECT_DIR),
        name=args.name,
        exist_ok=True,

        # --- Optimizer ---
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,       # Final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,

        # --- Augmentation (adapted for grayscale SAR) ---
        hsv_h=0.0,      # No hue augmentation (grayscale)
        hsv_s=0.0,      # No saturation augmentation (grayscale)
        hsv_v=0.1,      # Slight brightness jitter (useful for SAR intensity variation)
        degrees=15.0,    # Slight rotation (ships can be at any angle)
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,     # Vertical flip is valid for SAR overhead imagery
        mosaic=1.0,
        mixup=0.1,

        # --- Other ---
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,      # Generate training plots
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best model: {PROJECT_DIR / args.name / 'weights' / 'best.pt'}")
    print(f"  Results:    {PROJECT_DIR / args.name}")


if __name__ == "__main__":
    main()
