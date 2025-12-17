This repository provides the code for **PRIME-PVTT**, a multimodal transformer-based model for:

- Survival analysis of hepatocellular carcinoma (HCC) patients
- Classification tasks related to portal vein tumor thrombus (PVTT)

## Overview

PRIME-PVTT is a transformer-based multimodal medical prediction model. It processes three modalities:

- Imaging features
- Text features
- Structured clinical metrics

Fusion is achieved via **parallel cross-attention channels** with **cohort-specific processing**:

- **ETS Channel**: For ETS cohort samples (`cohort = 0`)
- **ES Channel**: For ES cohort samples (`cohort = 1`)

Shared self-attention layers operate after cohort-specific fusion, providing a unified representation for prediction.

This repository contains two training scripts:

- `train_survival.py`: Survival analysis (time-to-event) with Cox loss
- `train_classification.py`: Classification task (e.g., PVTT status) with CE loss

## Environment

The original implementation was tested on:

- Windows 10
- Python 3.9
- PyTorch 2.5.1
- CUDA 12.1

You can adjust batch size and other hyperparameters for your own hardware.

## Code Structure

- `models/`
  - `configs.py`: Configuration of PRIME-PVTT transformer (hidden size, heads, etc.)
  - `embed.py`: Multimodal embedding (image, text, lab, age, sex)
  - `attention.py`: Multi-head attention and multimodal cross-attention
  - `block.py`: Transformer blocks with optional gating
  - `encoder_parallel.py`: ETS / ES dual-channel cross-attention encoder
  - `modeling_prime_surv.py`: `PRIME` backbone used for both survival and classification
- `train_survival.py`: Survival training script
- `train_classification.py`: Classification training script
- `data/`: Place your processed `.pkl` data here (or set paths via arguments)

## Data Format

Both tasks expect preprocessed pickle files with the following fields (keys):

Common keys:

- `images`: Image features (tensor/array)
- `text_features`: Text features
- `structured_data`: Structured clinical data
- `age`: Age information
- `sex`: Sex information
- `cohort` (optional): 0 for ETS, 1 for ES (if missing, defaults to 0 in the scripts)
- `patient_ids`: List/array of patient IDs

**Survival task extra keys:**

- `event`: Event indicator (0/1)
- `time`: Time to event

**Classification task extra keys:**

- `labels`: Integer class labels (e.g., 0/1)

## Running Survival Training

```bash
python train_survival.py \
    --train_data ./data/train_surv.pkl \
    --val_data ./data/val_surv.pkl \
    --test_data ./data/test_surv.pkl \
    --save_path ./checkpoints/prime_pvtt_surv
```

Key arguments:

- `--train_data`, `--val_data`, `--test_data`: Paths to `.pkl` files
- `--save_path`: Directory for checkpoints, logs, and results
- `--batch_size`, `--lr`, `--num_epochs`, `--weight_decay`, etc.

Outputs:

- `{save_path}/best_model.pth`
- `{save_path}/model_epoch_*.pth`
- `{save_path}/logs/` (TensorBoard logs)
- `{save_path}/surv_results/`:
  - `train_metrics.csv`, `val_metrics.csv`, `test_metrics.csv`
  - `train_preds.csv`, `val_preds.csv`, `test_preds.csv`

## Running Classification Training

```bash
python train_classification.py \
    --train_data ./data/train_cls.pkl \
    --val_data ./data/val_cls.pkl \
    --test_data ./data/test_cls.pkl \
    --save_path ./checkpoints/prime_pvtt_cls \
    --num_classes 2
```

Key arguments:

- `--train_data`, `--val_data`, `--test_data`: Paths to `.pkl` files
- `--save_path`: Directory for checkpoints, logs, and results
- `--num_classes`: Number of classes (default 2)

Outputs:

- `{save_path}/best_model_cls.pth`
- `{save_path}/model_cls_epoch_*.pth`
- `{save_path}/logs_cls/` (TensorBoard logs)
- `{save_path}/cls_results/`:
  - `train_metrics.csv`, `val_metrics.csv`, `test_metrics.csv`
  - `train_preds.csv`, `val_preds.csv`, `test_preds.csv`

## Notes

1. PRIME-PVTT reuses the same ETS/ES dual-channel backbone for both survival and classification.
2. Only the output head and loss function differ between `train_survival.py` and `train_classification.py`.
3. For GitHub release, you only need to include:
   - `data/` (optionally with small demo files)
   - `models/`
   - `train_survival.py`
   - `train_classification.py`
   - `README.md`



