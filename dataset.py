"""
dataset.py
----------
PyTorch Dataset class for text-conditioned segmentation.
Loads image/mask pairs from the canonical folder structure, preprocesses
images through the CLIPSeg processor, and returns tensors ready for training.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPSegProcessor

PROCESSOR_NAME = "CIDAS/clipseg-rd64-refined"


class CLIPSegDataset(Dataset):
    """
    Unified dataset for text-conditioned image segmentation.

    Parameters
    ----------
    image_dirs : list of Path
        Directories containing source images (one per dataset/split).
    mask_dirs : list of Path
        Corresponding directories containing binary masks.
    prompts : list of str
        Text prompt for each directory (same order as image_dirs).
    processor : CLIPSegProcessor
        HuggingFace processor for CLIPSeg.
    image_size : int
        Target spatial size for masks (must match CLIPSeg input).
    """

    def __init__(
        self,
        image_dirs: List[Path],
        mask_dirs: List[Path],
        prompts: List[str],
        processor: CLIPSegProcessor,
        image_size: int = 352,
    ):
        super().__init__()
        self.processor = processor
        self.image_size = image_size
        self.samples: List[Tuple[Path, Path, str]] = []

        for img_dir, msk_dir, prompt in zip(image_dirs, mask_dirs, prompts):
            img_dir = Path(img_dir)
            msk_dir = Path(msk_dir)
            for img_file in sorted(img_dir.iterdir()):
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                mask_file = msk_dir / (img_file.stem + ".png")
                if not mask_file.exists():
                    continue
                self.samples.append((img_file, mask_file, prompt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, prompt = self.samples[idx]

        # Load image as RGB
        image = Image.open(img_path).convert("RGB")

        # Load mask as single-channel grayscale, resize to model input size
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0)

        # Process image + text through CLIPSeg processor
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding="max_length",
            return_tensors="pt",
        )
        # Squeeze the batch dimension added by the processor
        pixel_values = inputs["pixel_values"].squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask": mask_tensor,
            "prompt": prompt,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
        }


def build_dataloader(
    split: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    image_size: int = 352,
) -> torch.utils.data.DataLoader:
    """
    Convenience function: builds a unified DataLoader for a given split
    combining both taping and cracks datasets.
    """
    processor = CLIPSegProcessor.from_pretrained(PROCESSOR_NAME)

    data_root = Path("data")
    image_dirs = [
        data_root / "taping" / split / "images",
        data_root / "cracks" / split / "images",
    ]
    mask_dirs = [
        data_root / "taping" / split / "masks",
        data_root / "cracks" / split / "masks",
    ]
    prompts = ["segment taping area", "segment crack"]

    # Validate paths exist
    for d in image_dirs + mask_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Expected directory not found: {d}")

    dataset = CLIPSegDataset(image_dirs, mask_dirs, prompts, processor, image_size)
    print(f"  [{split}] loaded {len(dataset)} samples")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


if __name__ == "__main__":
    # Quick smoke test
    print("Building train dataloader …")
    try:
        loader = build_dataloader("train", batch_size=2, shuffle=False)
        batch = next(iter(loader))
        print(f"  pixel_values: {batch['pixel_values'].shape}")
        print(f"  mask:          {batch['mask'].shape}")
        print(f"  prompts:       {batch['prompt']}")
        print("Dataset smoke test passed.")
    except Exception as e:
        print(f"Smoke test failed: {e}")
