# Marine Waste Class-Incremental Learning (CIL)

A class-incremental marine imagery experiment using **TrashCan 1.0** and **ResNet18**. The checked-in demo outputs are smoke tests only; the real-data protocol uses the locally supplied TrashCan-Material annotations and records its mapping/split manifests separately.

- replay memory with iCaRL-style **herding** exemplar selection (the classifier is a learned linear head, so this is not the full iCaRL method)
- frozen-teacher **knowledge distillation** (temperature-scaled KL divergence)
- per-stage evaluation with CIL metrics: **forgetting**, **retention**, old/new class accuracy
- **Grad-CAM** visualizations per class
- checkpoint / resume support
- deterministic class ordering and experiment reproducibility

> The 7.5%, 8.57%, and 6% results currently under `artifacts/` are demo/smoke-test numbers and must not be reported as real results.

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

**Linux / macOS (bash / zsh):**

```bash
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
& ".\.venv\Scripts\Activate.ps1"
```

**Windows CMD:**

```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

The locally supplied raw TrashCan 1.0 tree is under `dataset/`. It includes `instance_version/`, `material_version/`, and `original_data/`; do not modify those source files.

### Variant and image-level labels

This project uses **TrashCan-Material** because the research question is primarily material classification. Its COCO files contain 16 native categories and the official image partitions have 6,008 train and 1,204 validation images. The supplied official train/validation partitions share source-video IDs, so they are not used directly as independent splits.

```text
dataset/material_version/
├── instances_train_trashcan.json
├── instances_val_trashcan.json
├── train/*.jpg
└── val/*.jpg
```

The source contains 7,212 JPEGs (480×270 or 480×360). Images can contain multiple object instances and multiple categories. To keep this a single-label image-classification task, preprocessing retains an image only when **all** its annotations have the same native category; it keeps that image once, regardless of same-category object count, and excludes mixed-category images rather than duplicating them across labels. The filter excludes 2,722 mixed-category images. It selects native categories with at least 100 pure images and at least 15 distinct source videos.

```text
python scripts/prepare_trashcan.py \
    --raw-root dataset/material_version \
    --output-root data/TrashCan_processed_seed42
```

The processed tree uses symlinks to raw JPEGs; image pixels are not copied or changed. A fixed seed-42, stratified group split assigns complete `vid_######` groups to train, validation, or test. The validation split uses seed 43 on the remaining groups. The split manifest, native-label crosswalk, and stage definition are written alongside the processed data. Preparation refuses to overwrite an existing output directory.

```text
data/TrashCan_processed_seed42/
├── train/<native-class>/*.jpg
├── val/<native-class>/*.jpg
├── test/<native-class>/*.jpg
├── split_manifest.json
├── class_mapping.json
├── cil_stages.json
└── dataset_inspection.json
```

The locked processed split contains 3,823 images: 2,147 train, 707 validation, and 969 test. It has zero source-video overlap across splits and zero byte-identical selected images. Exact class/split counts are in `data/TrashCan_processed_seed42/dataset_inspection.json` and `split_manifest.json`.

---

## Dataset inspection

Before any training, inspect the real dataset on disk:

```bash
python verify_dataset.py --data-root data/TrashCan_processed_seed42
python main.py --inspect-dataset --data-root data/TrashCan_processed_seed42
```

This prints a clean report containing:

- total images, total classes, class names
- images per class, train / val / test counts (if an official split exists)
- corrupted / missing / unsupported / duplicate files
- class imbalance ratios

The preparation report and split manifest remain under `data/TrashCan_processed_seed42/`; the CLI inspection report is written under the selected artifacts directory.

To run it from the root CLI entrypoint:

```bash
python main.py --inspect-dataset --data-root /path/to/TrashCan
```

If the real dataset is **not** found, the pipeline explicitly prints `REAL DATASET NOT FOUND` and stops with a non-zero exit code.

---

## Class-Incremental Learning methodology

The experiment uses six native Material/TrashCan categories. The previous project labels do not match this taxonomy one-to-one: `plastic`, `metal`, and `wood` have direct material equivalents; `cloth` corresponds approximately to `trash_fabric`; `fish` maps to `animal_fish`; `rov` is direct; and `other` is not equivalent to TrashCan's narrower `trash_etc`. Categories without enough pure images/source videos are excluded, and no labels are synthesized or merged for the experiment. See `class_mapping.json` for the complete 20-label review.

| Stage | Classes introduced | Cumulative classes |
|-------|--------------------|--------------------|
| **Stage 1** | `trash_plastic`, `trash_metal` | 2 |
| **Stage 2** | `trash_wood`, `trash_etc` | 4 |
| **Stage 3** | `animal_fish`, `rov` | 6 |

Stages are disjoint, cumulative, and saved to `data/TrashCan_processed_seed42/cil_stages.json` before training. The fixed test split is not used for parameter selection; training records validation metrics at each stage and accesses test images only after all stage checkpoints exist.

### Training within each stage

- **Backbone**: `torchvision.models.resnet18`. The local environment has no cached ImageNet weights and this workflow does not download weights, so the recorded real run uses random initialization (`--no-pretrained`).
- **Classifier**: a dynamically expanding linear head. When growing `2 → 4 → 6`, previously learned weights are preserved and only the new rows are freshly initialized.
- **Replay memory**: fixed budget (default 20 exemplars per class, configurable via `--memory-budget-per-class`) storing image paths.
- **Herding exemplar selection** (iCaRL-style): after each stage the pipeline extracts training feature embeddings and iteratively selects exemplars whose running mean best approximates the class mean. Never a naive `[:budget]` slice. This implementation uses a learned linear classifier, not iCaRL's nearest-mean classifier.
- **Imbalance option**: `--balanced-sampling` enables inverse-frequency weighted sampling and is compared against ordinary sampling using validation only.
- **Knowledge distillation**: at Stage 2 and 3, the previous-stage model is **frozen** to serve as a teacher. The combined loss is

  ```text
  L_total = CLASSIFICATION_WEIGHT * L_CE(y, logits)
          + DISTILLATION_WEIGHT   * L_KL( softmax(s_old/T) || softmax(t_old/T) ) * T^2
  ```

  where old-class logits only are distilled. New classes are learned from the ground-truth cross-entropy term only.

Forgetting is measured as

```text
forgetting(c) = previous_stage_accuracy(c) - current_stage_accuracy(c)
retention(c)  = current_stage_accuracy(c)  / previous_stage_accuracy(c)
```

and reported per-class and as mean / max summaries.

---

## Training

### Run the complete end-to-end experiment

```bash
python main.py --full-experiment --data-root data/TrashCan_processed_seed42
```

Equivalently:

```bash
python main.py --train --data-root data/TrashCan_processed_seed42
```

Useful overrides:

```bash
python main.py --full-experiment \
    --data-root data/TrashCan_processed_seed42 \
    --epochs 2 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --memory-budget-per-class 20 \
    --temperature 2.0 \
    --kd-weight 1.0 \
    --seed 42 \
    --device mps \
    --no-pretrained
```

For development/tuning, add `--validation-only`; it does not open the test images. Final training omits that option and evaluates the locked run on test only after all stages finish. Outputs default to `artifacts/real_trashcan_experiment/`, preserving the old demo artifacts.

### Resume from a checkpoint

You can skip already-completed stages. To resume starting after Stage 2:

```bash
python main.py --train --resume artifacts/checkpoints/stage_2.pt --data-root /path/to/TrashCan
```

---

## Evaluation

After training you can evaluate any saved checkpoint against the real test split:

```bash
python main.py --evaluate --checkpoint artifacts/checkpoints/stage_3.pt --data-root /path/to/TrashCan
```

This prints (as JSON):

- overall accuracy, macro / weighted precision, recall, F1
- per-class precision / recall / F1 / support / accuracy
- confusion matrix

Per-stage numerical and visual outputs are produced automatically during training and saved to `artifacts/results/`:

- `stage_1_metrics.json`, `stage_2_metrics.json`, `stage_3_metrics.json`
- `stage_*_per_class_metrics.csv`
- `confusion_matrix_stage_1.png`, `.npy`, `.csv` …
- `summary_by_stage.csv`
- `cil_summary.json` (forgetting / retention / old-vs-new per stage)
- accuracy-across-stages, old-class-accuracy, forgetting-summary, retention-summary plots.

---

## Grad-CAM

Grad-CAM visualizations are generated from the real dataset using the final convolutional layer of ResNet18 (`layer4`).

```bash
python main.py --gradcam \
    --checkpoint artifacts/checkpoints/stage_3.pt \
    --data-root /path/to/TrashCan \
    --samples-per-class 3
```

Outputs are written to `artifacts/gradcam/`:

- `gradcam_<class>_<n>.png` — a three-panel figure showing (1) original image, (2) Grad-CAM heatmap, (3) overlay with predicted class + true class + confidence
- `manifest.json` — an index with `image_path`, `true_class`, `predicted_class`, `confidence`, and `correct` flag for every generated sample.

The default is 3 samples per class (configurable via `--samples-per-class`).

---

## Outputs layout

All artifacts land under the configured output directory (`artifacts/` by default, override with `--output-dir`):

```text
artifacts/
├── class_order.json
├── checkpoints/
│   ├── stage_1.pt
│   ├── stage_2.pt
│   └── stage_3.pt
├── results/
│   ├── experiment_config.json
│   ├── dataset_inspection.json
│   ├── stage_1_metrics.json
│   ├── stage_2_metrics.json
│   ├── stage_3_metrics.json
│   ├── stage_1_per_class_metrics.csv
│   ├── stage_2_per_class_metrics.csv
│   ├── stage_3_per_class_metrics.csv
│   ├── summary_by_stage.csv
│   ├── cil_summary.json
│   ├── final_summary.json
│   ├── confusion_matrix_stage_1.png   (+.npy / .csv)
│   ├── confusion_matrix_stage_2.png   (+.npy / .csv)
│   ├── confusion_matrix_stage_3.png   (+.npy / .csv)
│   ├── accuracy_across_stages.png
│   ├── old_class_accuracy_across_stages.png
│   ├── avg_incremental_accuracy_across_stages.png
│   ├── forgetting_summary_last_stage.png
│   └── retention_summary_last_stage.png
├── gradcam/
│   ├── gradcam_<class>_1.png ...
│   └── manifest.json
└── logs/
    └── train.log
```

Each checkpoint contains:

- `model_state_dict`
- `optimizer_state_dict`
- `stage`, `epoch`
- `class_names`, `class_to_index`, `class_order`
- `config` (full hyperparameter snapshot incl. versions / seed / device)
- `metrics` (last-stage eval)
- `replay_memory` (exemplar paths per class)
- `stage_metrics_history`, `stage_history`, `stage_class_splits`
- `start_stage` (for resuming)

---

## CLI reference

```text
python main.py --inspect-dataset  --data-root <path> [--output-dir <path>]
python main.py --train            --data-root <path> [options]
python main.py --full-experiment  --data-root <path> [options]
python main.py --train --resume <checkpoint> --data-root <path>
python main.py --evaluate --checkpoint <checkpoint> --data-root <path>
python main.py --gradcam  --checkpoint <checkpoint> --data-root <path>
                              [--samples-per-class N]
```

Common options:

```text
--epochs INT                 epochs per stage (default 3)
--batch-size INT             training batch size (default 32)
--learning-rate FLOAT        AdamW LR (default 1e-4)
--memory-size INT            total replay memory budget (optional cap)
--memory-budget-per-class INT exemplars per class (default 20)
--temperature FLOAT          KD temperature T (default 2.0)
--kd-weight FLOAT            distillation loss weight (default 1.0)
--seed INT                   master seed (default 42)
--num-workers INT            DataLoader workers (default 0)
--device cpu|cuda            override auto-selection
--no-pretrained              disable ImageNet pretrained weights
--stage INT                  target stage for --evaluate
--samples-per-class INT      Grad-CAM samples per class (default 3)
```

---

## Running tests

Unit tests use **synthetic** fixtures only.

```bash
pytest -q
```

Additional static checks:

```bash
python -m compileall .
```

---

## Research integrity

- No dummy / synthetic / randomly-generated images are used for final experiments.
- No fabricated metric is ever printed by the pipeline; if no real dataset exists the pipeline stops with `REAL DATASET NOT FOUND`.
- The synthetic debug utility (`generate_dummy_data.py`) is explicitly scoped to smoke testing and cannot feed the final evaluation loop.

---

## References

- Hong, Fulton, Sattar (2020) — *TrashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris* arXiv:2007.08097
- Rebuffi, Kolesnikov, Sperl, Lampert (2017) — iCaRL: Incremental Classifier and Representation Learning (CVPR)
- Hinton, Vinyals, Dean (2015) — Distilling the Knowledge in a Neural Network
- Selvaraju et al. (2017) — Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
