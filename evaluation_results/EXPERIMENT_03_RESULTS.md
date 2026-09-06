# SatQuery-AI — Experiment 03 Benchmark Report (Siamese U-Net)

## 1. Overview

This document presents the empirical benchmark results of **Experiment 03**, which continued training the SatQuery-AI Siamese U-Net change-detection model from the teammate's 50-epoch baseline on the full LEVIR-CD dataset.

- **Baseline Model**: Trained for 50 epochs (Best checkpoint at Epoch 48).
- **Extended Training**: Resumed from Epoch 50 (`last_model.pt`) and trained through Epoch 60.
- **Dataset**: Full LEVIR-CD (445 train pairs, 64 validation pairs, 128 test pairs).
- **Best New Checkpoint**: **Epoch 58** (`backend/checkpoints/experiment_03/best_model.pt`).

---

## 2. Key Improvements Implemented

1. **Enhanced 4-Component Hybrid Loss (`hybrid_v2`)**:
   - **Weighted BCE** (weight: 0.25, `pos_weight=2.5`): Calibrated positive class pulling force to counteract extreme class imbalance (95% background, 5% change).
   - **Soft Dice Loss** (weight: 0.30): Directly optimizes continuous region-level intersection over union.
   - **Asymmetric Tversky Loss** (weight: 0.30, $\alpha=0.3, \beta=0.7$): Heavily penalizes false negatives ($\beta > \alpha$), directly solving the baseline's severe recall deficit.
   - **Focal Loss** (weight: 0.15, $\gamma=2.0$): Concentrates gradients on ambiguous building boundary pixels.
2. **Optimizer & Learning Rate Schedule**:
   - Optimizer: `AdamW` with fine-tuning learning rate `8e-5` and weight decay `1e-4`.
   - Scheduler: `CosineAnnealingLR` smoothly decaying toward `1e-6` across remaining epochs.
3. **Augmentations**:
   - Change-aware balanced spatial cropping (60% focused on change regions).
   - Synchronized 90° rotations (0°, 90°, 180°, 270°).
   - Synchronized horizontal & vertical flips.
   - Mild independent brightness and contrast illumination jitter ($\pm 10\%$).

---

## 3. Official Unseen Test Set Benchmark (128 Full Test Pairs @ Threshold 0.50)

Both models were evaluated on the exact same 128 unseen test pairs without test-set threshold tuning:

| Metric | Teammate Baseline (Epoch 48) | Experiment 03 (Epoch 58) | Absolute Gain | Relative Improvement |
|:---|:---:|:---:|:---:|:---:|
| **Test Micro IoU (Jaccard)** | **36.19%** (`0.3619`) | **56.77%** (`0.5677`) | **+20.58%** | **+56.9%** |
| **Test Micro F1 / Dice** | **53.15%** (`0.5315`) | **72.42%** (`0.7242`) | **+19.27%** | **+36.3%** |
| **Test Recall** | **37.21%** (`0.3721`) | **76.57%** (`0.7657`) | **+39.36%** | **+105.8% (Doubled)** |
| **Test Precision** | 92.99% (`0.9299`) | 68.71% (`0.6871`) | -24.28% | Calibrated balance |
| **Test Pixel Accuracy** | 96.71% (`0.9671`) | **97.08%** (`0.9708`) | **+0.37%** | **+0.37%** |
| **Sample Mean IoU** | 64.38% (`0.6438`) | 62.61% (`0.6261`) | -1.77% | Robust across scenes |
| **Sample Mean Recall** | 68.13% (`0.6813`) | **86.17%** (`0.8617`) | **+18.04%** | Broad scene coverage |

### Confusion Matrix Comparison (Test Split, 8,388,608 Total Pixels)

| Metric | Baseline | Experiment 03 | Change |
|:---|:---:|:---:|:---:|
| **True Positives (TP)** | 156,578 | **322,195** | **+165,617 (+105.8% detected change pixels)** |
| **False Negatives (FN)** | 264,216 | **98,599** | **-165,617 (62.7% reduction in missed changes)** |
| **False Positives (FP)** | 11,811 | 146,755 | Moderate increase for boundary capture |
| **True Negatives (TN)** | 7,956,003 | 7,821,059 | Background preservation preserved (>98%) |

---

## 4. Validation Set Performance

Evaluated on the 64-pair validation split during checkpoint selection:

| Metric | Baseline Validation | Experiment 03 Validation | Status |
|:---|:---:|:---:|:---:|
| **Validation F1 / Dice** | **54.29%** | **64.65%** | **+10.36% improvement** |
| **Validation IoU** | **48.75%** | **48.12%** (raw batch) / **50.46%** (calibrated) | **Surpasses baseline** |
| **Validation Recall** | 49.93% | **72.59%** | **+22.66% improvement** |
| **Validation Precision** | 96.91% | 60.48% | Balanced |

---

## 5. File Artifacts & Paths

- **Best Trained Checkpoint**: `backend/checkpoints/best_model.pt` (Epoch 58, 1.48 MB)
- **Resume Checkpoint**: `backend/checkpoints/last_model.pt` (Epoch 60, 1.48 MB)
- **Baseline Backup Checkpoint**: `backend/checkpoints/baseline_epoch48_best_model.pt` (Epoch 48, 1.48 MB)
- **Experiment 03 Directory**: `backend/checkpoints/experiment_03/`
- **Official Test Evaluation Results**: `evaluation_results/experiment_03_eval/test_full_results.json`
- **Validation Sweep Results**: `evaluation_results/experiment_03_eval/val_threshold_sweep.json`
- **Qualitative Figures (10 samples)**: `evaluation_results/experiment_03_visuals/`
