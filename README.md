# Text-Conditioned Image Segmentation for Drywall QA

Fine-tune **CLIPSeg** (`CIDAS/clipseg-rd64-refined`) on two drywall quality-assurance datasets using text prompts to produce binary segmentation masks.

| Prompt | Task |
|---|---|
| `"segment taping area"` | Drywall joint / seam detection |
| `"segment crack"` | Crack detection |

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

### Roboflow API Key

The download script uses the Roboflow API. The key is already embedded in `download_data.py`. If you need to change it, edit the `ROBOFLOW_API_KEY` variable at the top of that file.

---

## Running the Pipeline

### Full pipeline (recommended)

```bash
python run_all.py
```

This runs every step end-to-end in order:

1. **download_data.py** — Downloads both datasets from Roboflow
2. **prepare_data.py** — Converts COCO annotations → binary masks, 80/10/10 split
3. **train.py** — Fine-tunes CLIPSeg for 20 epochs
4. **inference.py** — Generates prediction masks on the test set
5. **evaluate.py** — Computes mIoU & Dice, saves `results.json`
6. **visualize.py** — Creates side-by-side comparison figures

### Running scripts individually

```bash
python download_data.py
python prepare_data.py
python train.py
python inference.py
python evaluate.py
python visualize.py
```

---

## Configuration

| Parameter | Value |
|---|---|
| Model | `CIDAS/clipseg-rd64-refined` |
| Random seed | **42** (set in every script for torch, numpy, random, CUDA) |
| Train / Val / Test split | 80 / 10 / 10 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss function | Binary Cross-Entropy with Logits |
| Epochs | 20 |
| Batch size | 4 |
| Device | Automatically uses CUDA if available, otherwise CPU |

---

## Results

> *Fill in after training completes.*

| Prompt | mIoU | Dice |
|---|---|---|
| `segment crack` | —% | —% |
| `segment taping area` | —% | —% |
| **Overall** | —% | —% |

Full metrics are saved to `results.json` after running `evaluate.py`.

---

## Folder Structure

```
project/
├── data/
│   ├── taping/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   └── test/
│   │       ├── images/
│   │       └── masks/
│   └── cracks/
│       ├── train/
│       │   ├── images/
│       │   └── masks/
│       ├── val/
│       │   ├── images/
│       │   └── masks/
│       └── test/
│           ├── images/
│           └── masks/
├── predictions/          ← binary masks from inference
├── visuals/              ← side-by-side comparison PNGs
├── checkpoints/          ← best_model.pt
├── download_data.py
├── prepare_data.py
├── dataset.py
├── train.py
├── inference.py
├── evaluate.py
├── visualize.py
├── run_all.py
├── requirements.txt
├── results.json
├── split_counts.txt
└── README.md
```

---

## Outputs

| File / Folder | Description |
|---|---|
| `checkpoints/best_model.pt` | Best model weights (lowest val loss) |
| `predictions/` | Binary PNG masks — filename: `{image_id}__{prompt_slug}.png` |
| `visuals/` | 4 comparison PNGs (best & worst per dataset) |
| `results.json` | mIoU and Dice per prompt + overall |
| `split_counts.txt` | Number of images per split |
