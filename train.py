"""
train.py
--------
Fine-tunes CLIPSeg (CIDAS/clipseg-rd64-refined) on both drywall taping and
crack datasets simultaneously using Binary Cross-Entropy loss, Adam optimizer,
and saves the best checkpoint based on validation loss.
"""

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from transformers import CLIPSegForImageSegmentation

from dataset import build_dataloader

# ────────────────────────────────────────────
# Hyperparameters & configuration
# ────────────────────────────────────────────
SEED = 42
LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 4
IMAGE_SIZE = 352
MODEL_NAME = "CIDAS/clipseg-rd64-refined"
CHECKPOINT_DIR = Path("checkpoints")


def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    count = 0

    for batch in tqdm(loader, desc="    train", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (B, H, W)

        # Resize logits to match mask size if needed
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits.unsqueeze(1),
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * pixel_values.size(0)
        count += pixel_values.size(0)

    return running_loss / max(count, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation. Returns average loss."""
    model.eval()
    running_loss = 0.0
    count = 0

    for batch in tqdm(loader, desc="    val  ", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits.unsqueeze(1),
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        loss = criterion(logits, masks)
        running_loss += loss.item() * pixel_values.size(0)
        count += pixel_values.size(0)

    return running_loss / max(count, 1)


def main() -> None:
    set_seed(SEED)

    print("=" * 60)
    print("  CLIPSeg FINE-TUNING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # ---- Data ----
    print("\nLoading datasets …")
    train_loader = build_dataloader("train", batch_size=BATCH_SIZE, shuffle=True, image_size=IMAGE_SIZE)
    val_loader = build_dataloader("val", batch_size=BATCH_SIZE, shuffle=False, image_size=IMAGE_SIZE)

    # ---- Model ----
    print(f"\nLoading pretrained model: {MODEL_NAME}")
    try:
        model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = -1

    print(f"\nTraining for {EPOCHS} epochs  (lr={LR}, batch_size={BATCH_SIZE})")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Save best checkpoint
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            improved = "  ** saved best **"

        print(
            f"  Epoch {epoch:02d}/{EPOCHS}  |  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  |  "
            f"time={epoch_time:.1f}s{improved}"
        )

        # After first epoch, estimate total training time
        if epoch == 1:
            est_total = epoch_time * EPOCHS
            est_min = est_total / 60.0
            print(f"  >>> Estimated total training time: ~{est_min:.1f} minutes")

    print("-" * 60)
    print(f"  Best val loss = {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"  Checkpoint saved to: {CHECKPOINT_DIR / 'best_model.pt'}")
    print("  Training complete.\n")


if __name__ == "__main__":
    main()
