import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from pycocotools.coco import COCO
except ImportError:
    print("ERROR: pycocotools not installed. Run: pip install pycocotools")
    sys.exit(1)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_CONFIGS = [
    {
        "name": "taping",
        "prompt": "segment taping area",
        "download_dir_keywords": ["drywall", "join", "detect"],
    },
    {
        "name": "cracks",
        "prompt": "segment crack",
        "download_dir_keywords": ["crack"],
    },
]

OUTPUT_ROOT = Path("data")


def find_coco_json(base_dir: Path) -> Path:
    """Recursively search for the COCO annotation JSON file."""
    candidates = list(base_dir.rglob("_annotations.coco.json"))
    if not candidates:
        candidates = list(base_dir.rglob("*.json"))
        candidates = [c for c in candidates if "annotation" in c.name.lower() or "coco" in c.name.lower()]
    if not candidates:
        candidates = list(base_dir.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No COCO JSON found under {base_dir}")
    return candidates[0]


def collect_images_from_roboflow_dir(base_dir: Path):
    results = []
    for sub in ["train", "valid", "test", ""]:
        candidate = base_dir / sub if sub else base_dir
        json_path = None
        try:
            json_path = find_coco_json(candidate)
        except FileNotFoundError:
            continue
        images_folder = json_path.parent
        coco = COCO(str(json_path))
        img_ids = coco.getImgIds()
        imgs = coco.loadImgs(img_ids)
        results.append((coco, imgs, images_folder))
    return results


def generate_mask(coco, img_info, images_folder: Path) -> np.ndarray:
    h, w = img_info["height"], img_info["width"]
    ann_ids = coco.getAnnIds(imgIds=img_info["id"])
    anns = coco.loadAnns(ann_ids)

    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        if "segmentation" in ann and ann["segmentation"]:
            seg = ann["segmentation"]
            if isinstance(seg, list) and len(seg) > 0:
                try:
                    rle_or_poly = coco.annToMask(ann)
                    mask = np.maximum(mask, rle_or_poly)
                    continue
                except Exception:
                    pass
        if "bbox" in ann:
            x, y, bw, bh = [int(round(v)) for v in ann["bbox"]]
            mask[y : y + bh, x : x + bw] = 1

    return mask


def process_dataset(config: dict) -> list:

    name = config["name"]
    prompt = config["prompt"]
    keywords = config["download_dir_keywords"]

    project_root = Path(".")
    candidates = [
        d for d in project_root.iterdir()
        if d.is_dir() and all(kw.lower() in d.name.lower() for kw in keywords)
    ]
    candidates = [d for d in candidates if d.name.lower() != "data"]
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find downloaded dataset directory matching keywords {keywords}. "
            f"Make sure download_data.py ran successfully."
        )
    base_dir = candidates[0]
    print(f"\n  Found dataset directory: {base_dir}")

    groups = collect_images_from_roboflow_dir(base_dir)
    if not groups:
        raise FileNotFoundError(f"No COCO annotations found in {base_dir}")

    tmp_images = OUTPUT_ROOT / name / "_all" / "images"
    tmp_masks = OUTPUT_ROOT / name / "_all" / "masks"
    tmp_images.mkdir(parents=True, exist_ok=True)
    tmp_masks.mkdir(parents=True, exist_ok=True)

    samples = []
    seen_filenames = set()

    for coco, imgs, images_folder in groups:
        for img_info in tqdm(imgs, desc=f"  Generating masks ({name})", leave=False):
            fname = img_info["file_name"]
            if fname in seen_filenames:
                continue
            seen_filenames.add(fname)

            src_image = images_folder / fname
            if not src_image.exists():
                # Try without subfolders
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    alt = images_folder / (Path(fname).stem + ext)
                    if alt.exists():
                        src_image = alt
                        break
            if not src_image.exists():
                continue

            mask_arr = generate_mask(coco, img_info, images_folder)
            mask_arr = (mask_arr > 0).astype(np.uint8) * 255

            dst_img = tmp_images / fname
            dst_mask = tmp_masks / (Path(fname).stem + ".png")

            shutil.copy2(str(src_image), str(dst_img))
            Image.fromarray(mask_arr, mode="L").save(str(dst_mask))

            samples.append(
                {
                    "image_name": fname,
                    "mask_name": Path(fname).stem + ".png",
                    "prompt": prompt,
                }
            )

    print(f"  Total images for '{name}': {len(samples)}")
    return samples


def split_and_save(name: str, samples: list) -> dict:

    random.shuffle(samples)
    train_val, test = train_test_split(samples, test_size=0.10, random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.1111, random_state=SEED)

    splits = {"train": train, "val": val, "test": test}
    counts = {}

    tmp_images = OUTPUT_ROOT / name / "_all" / "images"
    tmp_masks = OUTPUT_ROOT / name / "_all" / "masks"

    for split_name, split_samples in splits.items():
        img_dir = OUTPUT_ROOT / name / split_name / "images"
        msk_dir = OUTPUT_ROOT / name / split_name / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        for s in tqdm(split_samples, desc=f"  Saving {name}/{split_name}", leave=False):
            shutil.copy2(str(tmp_images / s["image_name"]), str(img_dir / s["image_name"]))
            shutil.copy2(str(tmp_masks / s["mask_name"]), str(msk_dir / s["mask_name"]))

        counts[split_name] = len(split_samples)

    shutil.rmtree(str(OUTPUT_ROOT / name / "_all"), ignore_errors=True)

    return counts


def main() -> None:
    print("=" * 60)
    print("  DATA PREPARATION")
    print("=" * 60)

    all_counts = {}

    for config in DATASET_CONFIGS:
        name = config["name"]
        print(f"\nProcessing dataset: {name}")
        try:
            samples = process_dataset(config)
            counts = split_and_save(name, samples)
            all_counts[name] = counts
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")
            raise

    print("\n" + "=" * 60)
    print("  SPLIT COUNTS")
    print("=" * 60)
    total_train, total_val, total_test = 0, 0, 0

    lines = []
    for ds_name, counts in all_counts.items():
        t, v, te = counts["train"], counts["val"], counts["test"]
        total_train += t
        total_val += v
        total_test += te
        line = f"  {ds_name:12s}  train={t:4d}  val={v:4d}  test={te:4d}  total={t+v+te:4d}"
        print(line)
        lines.append(line)

    summary = f"  {'TOTAL':12s}  train={total_train:4d}  val={total_val:4d}  test={total_test:4d}  total={total_train+total_val+total_test:4d}"
    print(summary)
    lines.append(summary)

    with open("split_counts.txt", "w") as f:
        f.write("Split Counts (seed=42, 80/10/10)\n")
        f.write("=" * 60 + "\n")
        for line in lines:
            f.write(line + "\n")

    print("\n  Saved split_counts.txt")
    print("  Data preparation complete.\n")


if __name__ == "__main__":
    main()
