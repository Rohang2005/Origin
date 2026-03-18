"""
inference.py
------------
Loads the best fine-tuned CLIPSeg checkpoint and runs inference on all
test images for both prompts, saving binary PNG masks to predictions/.
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "CIDAS/clipseg-rd64-refined"
CHECKPOINT_PATH = Path("checkpoints") / "best_model.pt"
PREDICTIONS_DIR = Path("predictions")

DATASETS = [
    {
        "name": "taping",
        "prompt": "segment taping area",
        "test_images": Path("data") / "taping" / "test" / "images",
    },
    {
        "name": "cracks",
        "prompt": "segment crack",
        "test_images": Path("data") / "cracks" / "test" / "images",
    },
]


def load_model(device: torch.device):
    """Load CLIPSeg model with fine-tuned weights."""
    print(f"  Loading model from {CHECKPOINT_PATH} …")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME)
    state_dict = torch.load(str(CHECKPOINT_PATH), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def slugify(prompt: str) -> str:
    """Convert prompt to filename-safe slug."""
    return prompt.strip().lower().replace(" ", "_")


@torch.no_grad()
def run_inference(model, processor, device) -> int:
    """
    Run inference on all test images for both prompts.
    Returns total number of predictions generated.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    for ds in DATASETS:
        prompt = ds["prompt"]
        slug = slugify(prompt)
        img_dir = ds["test_images"]

        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found, skipping.")
            continue

        image_files = sorted([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])
        print(f"\n  Running inference on {len(image_files)} images for prompt: \"{prompt}\"")

        for img_path in tqdm(image_files, desc=f"    {ds['name']}", leave=False):
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size

            inputs = processor(
                text=[prompt],
                images=[image],
                padding="max_length",
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(device)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (1, H, W)

            # Upsample to original image size
            logits = F.interpolate(
                logits.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            # Sigmoid → threshold → binary mask (0 or 255)
            prob = torch.sigmoid(logits).cpu().numpy()
            binary_mask = ((prob > 0.5) * 255).astype(np.uint8)

            # Save with format: {image_id}__{prompt_slug}.png
            image_id = img_path.stem
            out_name = f"{image_id}__{slug}.png"
            Image.fromarray(binary_mask, mode="L").save(str(PREDICTIONS_DIR / out_name))
            total += 1

    return total


def main() -> None:
    print("=" * 60)
    print("  INFERENCE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    try:
        model = load_model(device)
        processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
        count = run_inference(model, processor, device)
        print(f"\n  Total predictions saved: {count}")
        print(f"  Output directory: {PREDICTIONS_DIR}/")
        print("  Inference complete.\n")
    except Exception as e:
        print(f"  ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
