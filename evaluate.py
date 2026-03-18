"""
evaluate.py
-----------
Computes mIoU and Dice score on the test set for each prompt and overall.
Saves results to results.json and prints a formatted table.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PREDICTIONS_DIR = Path("predictions")
RESULTS_FILE = Path("results.json")

DATASETS = [
    {
        "name": "taping",
        "prompt": "segment taping area",
        "test_images": Path("data") / "taping" / "test" / "images",
        "test_masks": Path("data") / "taping" / "test" / "masks",
    },
    {
        "name": "cracks",
        "prompt": "segment crack",
        "test_images": Path("data") / "cracks" / "test" / "images",
        "test_masks": Path("data") / "cracks" / "test" / "masks",
    },
]


def slugify(prompt: str) -> str:
    return prompt.strip().lower().replace(" ", "_")


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Intersection over Union for binary masks."""
    pred_bin = pred > 127
    gt_bin = gt > 127
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient for binary masks."""
    pred_bin = pred > 127
    gt_bin = gt > 127
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / total)


def evaluate_dataset(ds_config: dict) -> dict:
    """Evaluate one dataset. Returns dict with iou_list and dice_list."""
    prompt = ds_config["prompt"]
    slug = slugify(prompt)
    img_dir = ds_config["test_images"]
    mask_dir = ds_config["test_masks"]

    if not img_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {img_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Test masks directory not found: {mask_dir}")

    image_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

    iou_list = []
    dice_list = []
    missing = 0

    for img_path in tqdm(image_files, desc=f"  Evaluating {ds_config['name']}", leave=False):
        image_id = img_path.stem
        pred_path = PREDICTIONS_DIR / f"{image_id}__{slug}.png"
        gt_path = mask_dir / (image_id + ".png")

        if not pred_path.exists():
            missing += 1
            continue
        if not gt_path.exists():
            missing += 1
            continue

        pred_mask = np.array(Image.open(pred_path).convert("L"))
        gt_mask = np.array(Image.open(gt_path).convert("L"))

        # Resize prediction to match ground truth if sizes differ
        if pred_mask.shape != gt_mask.shape:
            pred_img = Image.fromarray(pred_mask)
            pred_img = pred_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
            pred_mask = np.array(pred_img)

        iou_list.append(compute_iou(pred_mask, gt_mask))
        dice_list.append(compute_dice(pred_mask, gt_mask))

    if missing > 0:
        print(f"    WARNING: {missing} images skipped (missing prediction or GT)")

    return {"iou_list": iou_list, "dice_list": dice_list}


def main() -> None:
    print("=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    if not PREDICTIONS_DIR.exists():
        print("  ERROR: predictions/ directory not found. Run inference.py first.")
        sys.exit(1)

    results = {}
    all_ious = []
    all_dices = []

    for ds in DATASETS:
        try:
            metrics = evaluate_dataset(ds)
        except Exception as e:
            print(f"  ERROR evaluating {ds['name']}: {e}")
            raise

        mean_iou = np.mean(metrics["iou_list"]) if metrics["iou_list"] else 0.0
        mean_dice = np.mean(metrics["dice_list"]) if metrics["dice_list"] else 0.0

        results[ds["prompt"]] = {
            "mIoU": round(float(mean_iou) * 100, 2),
            "Dice": round(float(mean_dice) * 100, 2),
            "num_samples": len(metrics["iou_list"]),
        }

        all_ious.extend(metrics["iou_list"])
        all_dices.extend(metrics["dice_list"])

    overall_iou = np.mean(all_ious) if all_ious else 0.0
    overall_dice = np.mean(all_dices) if all_dices else 0.0

    results["Overall"] = {
        "mIoU": round(float(overall_iou) * 100, 2),
        "Dice": round(float(overall_dice) * 100, 2),
        "num_samples": len(all_ious),
    }

    # ---- Save JSON ----
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")

    # ---- Print table ----
    print("\n  " + "-" * 56)
    print(f"  {'Prompt':<26s} {'mIoU':>8s} {'Dice':>8s} {'N':>6s}")
    print("  " + "-" * 56)
    for key in ["segment crack", "segment taping area", "Overall"]:
        if key in results:
            r = results[key]
            print(f"  {key:<26s} {r['mIoU']:>7.2f}% {r['Dice']:>7.2f}% {r['num_samples']:>6d}")
    print("  " + "-" * 56)
    print("\n  Evaluation complete.\n")


if __name__ == "__main__":
    main()
