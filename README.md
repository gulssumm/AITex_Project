# AITEX Fabric Defect Detection — U-Net & HRNet Segmentation

Bachelor thesis project on automated fabric defect detection using the
[AITEX Fabric Image Database](https://www.aitex.es/afid/). The pipeline covers
dataset preparation, exploratory analysis, a CNN classification baseline
(defect / no-defect), and **pixel-level defect segmentation** with an
**Attention U-Net** and a simplified **HRNet**, trained on patch-extracted,
augmented data.

## Dataset

- **Source:** AITEX fabric image database — 245 images (4096×256 px) across 12
  fabric textures, with 7 defect types and pixel-level ground-truth masks.
- Images are heavily imbalanced (far more no-defect area than defect area),
  which drives most of the design choices below (patch extraction centered on
  defects, class-weighted loss, Dice-based metrics instead of raw accuracy).
- The dataset itself is not included in this repository — see [AFID](https://www.aitex.es/afid/)
  for access, and update the paths noted in [Environment](#environment) below.

## Repository Structure

```
├── aitex_unet_hrnet/                      # Final, consolidated notebooks (Kaggle-run)
│   ├── 01_AITEX_EDA_and_Preprocessing.ipynb
│   ├── 02_AITEX_Segmentation_Models_Training.ipynb
│   └── aitex-unet-hrnet.pdf               # Exported run of the training notebook
├── src/                                   # Standalone scripts — classification baseline & dataset tooling
│   ├── AITex_preprocessing.py
│   ├── AITex_datasetprep.py
│   ├── AITex_detailed_analysis.py
│   ├── AITex_mask_checking.py
│   ├── AITex_classification.py
│   └── AITex_classification_train.py
├── docs/
│   └── Unel_The_Power_of_Tiling...CVPRW_2019_paper.pdf   # Reference paper (tiling for small object detection)
├── assets/                                # Figures referenced below / in the thesis writeup
└── README.md
```

> The final results and write-up in this README are based on `aitex_unet_hrnet/`
> and `src/`. Earlier exploratory notebooks used during development are kept
> locally but are not part of this public repository.

## Pipeline

1. **Preprocessing & dataset prep** (`src/AITex_preprocessing.py`,
   `src/AITex_datasetprep.py`) — inventories the raw dataset, pairs each defect
   image with its mask (skipping defect images with no mask, generating blank
   masks for defect-free images), filters out fabric types with too few
   samples, and creates a stratified train/val split (`aitex_raw_clean/`).
2. **EDA** (`01_AITEX_EDA_and_Preprocessing.ipynb`,
   `src/AITex_detailed_analysis.py`) — class balance, image brightness/contrast
   statistics, defect size distribution, and sample visualizations with mask
   overlays.
3. **Mask cleaning** (`clean_mask_from_path`) — thresholds out compression
   noise, applies morphological opening, and filters out connected regions
   below a minimum area.
4. **Patch extraction** (`extract_defect_centered_patches`) — since defects are
   a tiny fraction of each 4096×256 image, 256×256 patches are sampled per
   image, with 75% of patches forced to be centered on a defect pixel to
   counter class imbalance.
5. **Augmentation** — Albumentations pipeline (flips, rotation,
   shift/scale/rotate, brightness/contrast, Gaussian noise/blur, elastic
   transform).
6. **Models:**
   - **Attention U-Net** — standard encoder–decoder with attention gates on
     the skip connections.
   - **HRNet (simplified)** — multi-resolution encoder with
     upsample-and-concatenate fusion (not the full HRNet spec).
   - Loss: combined BCE + Dice loss. Metrics: Dice, IoU, precision, recall,
     F1, tracked via a custom Keras `dice_coef`/`iou_coef`.
7. **Evaluation** — multi-threshold sweep to pick the best decision threshold
   per model.
8. **Baseline classifier** (`src/AITex_classification*.py`) — a simpler CNN
   that classifies fixed-size crops as defect / no-defect, used as a
   sanity-check baseline before committing to full segmentation.

## Results

Best per-model metrics from the multi-threshold evaluation (segmentation,
validation set):

| Model             | Threshold | Dice   | IoU    | Recall | Precision | F1     |
|--------------------|:---------:|:------:|:------:|:------:|:---------:|:------:|
| Attention U-Net    | 0.10      | 0.621  | 0.451  | 0.643  | 0.601     | 0.621  |
| HRNet (simplified) | 0.40      | 0.611  | 0.440  | 0.611  | 0.611     | 0.611  |

- Attention U-Net slightly edges out HRNet on Dice/IoU and has better recall —
  the more relevant metric for defect detection, since missing a defect is
  costlier than a false alarm.
- Training curves, confusion matrix, and dataset statistics are in `assets/`
  (`training_history.png`, `confusion_matrix.png`,
  `aitex_defect_size_distribution.png`, `aitex_image_quality_analysis.png`,
  `aitex_sample_visualization.png`).

## Reference

- Ünel, F.O. et al., *"The Power of Tiling for Small Object Detection"*,
  CVPRW 2019 — cited as motivation for the patch/tile-based training approach
  (`docs/`).

## Environment

Notebooks were run on Kaggle (GPU, TensorFlow/Keras) with `albumentations`
installed at runtime. Local scripts in `src/` use TensorFlow, OpenCV, and
scikit-learn, and expect data under a local path — edit `data_dir` /
`base_path` at the top of each script before running.

```
tensorflow
opencv-python
albumentations
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
```