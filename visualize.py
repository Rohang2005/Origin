"""
visualize.py
------------
Generates side-by-side visualizations (Original | Ground Truth | Prediction)
for 4 test images (2 per dataset — one good, one failure case each).
Saves PNGs to visuals/ for direct use in the assignment report.
"""

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PREDICTIONS_DIR = Path("predictions")
VISUALS_DIR = Path("visuals")
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
    pred_bin = pred > 127
    gt_bin = gt > 127
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def pick_best_and_worst(ds_config: dict):
    """
    Rank test images by IoU and return (best_path, worst_path) tuples
    of (image_path, gt_path, pred_path, iou).
    """
    prompt = ds_config["prompt"]
    slug = slugify(prompt)
    img_dir = ds_config["test_images"]
    mask_dir = ds_config["test_masks"]

    image_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

    scored = []
    for img_path in image_files:
        image_id = img_path.stem
        pred_path = PREDICTIONS_DIR / f"{image_id}__{slug}.png"
        gt_path = mask_dir / (image_id + ".png")
        if not pred_path.exists() or not gt_path.exists():
            continue

        pred_mask = np.array(Image.open(pred_path).convert("L"))
        gt_mask = np.array(Image.open(gt_path).convert("L"))

        if pred_mask.shape != gt_mask.shape:
            pred_img = Image.fromarray(pred_mask)
            pred_img = pred_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
            pred_mask = np.array(pred_img)

        iou = compute_iou(pred_mask, gt_mask)
        scored.append((img_path, gt_path, pred_path, iou))

    scored.sort(key=lambda x: x[3])

    if len(scored) < 2:
        return scored[:1] + scored[:1]  # duplicate if only one available

    worst = scored[0]   # lowest IoU → failure case
    best = scored[-1]   # highest IoU → good result
    return best, worst


def make_visualization(
    img_path: Path,
    gt_path: Path,
    pred_path: Path,
    iou: float,
    prompt: str,
    label: str,
    out_path: Path,
) -> None:
    """Create a 3-panel figure: Original | Ground Truth | Prediction."""
    image = np.array(Image.open(img_path).convert("RGB"))
    gt = np.array(Image.open(gt_path).convert("L"))
    pred = np.array(Image.open(pred_path).convert("L"))

    if pred.shape != gt.shape:
        pred_img = Image.fromarray(pred)
        pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
        pred = np.array(pred_img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Ground Truth Mask", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Predicted Mask", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(
        f'{label}  |  Prompt: "{prompt}"  |  IoU: {iou:.2%}',
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


def main() -> None:
    print("=" * 60)
    print("  VISUALIZATION")
    print("=" * 60)

    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        name = ds["name"]
        prompt = ds["prompt"]
        print(f"\n  Dataset: {name} (\"{prompt}\")")

        try:
            best, worst = pick_best_and_worst(ds)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        # Good result
        make_visualization(
            *best,
            prompt=prompt,
            label=f"{name.title()} — Best",
            out_path=VISUALS_DIR / f"{name}_best.png",
        )

        # Failure case
        make_visualization(
            *worst,
            prompt=prompt,
            label=f"{name.title()} — Worst",
            out_path=VISUALS_DIR / f"{name}_worst.png",
        )

    print(f"\n  All visualizations saved to {VISUALS_DIR}/")
    print("  Visualization complete.\n")


if __name__ == "__main__":
    main()
