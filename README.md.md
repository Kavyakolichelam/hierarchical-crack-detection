# Hierarchical Coarse-to-Fine Crack Detection Framework

> A reliability-aware, three-stage deep learning framework for automated concrete crack detection and geometric characterization. Co-authored research with Dr. Wisam Bukaita (Lawrence Technological University).

## 📊 Headline Results

| Stage | Model | Key Metric | Value |
|---|---|---|---|
| **Stage 1** — Classification | MobileNetV2 | Accuracy | **91.68%** |
| | | ROC-AUC | 0.9341 |
| | | Macro-F1 | 0.8437 |
| **Stage 2** — Segmentation | SegFormer-B2 @ 384×384 + TTA | Dice | **0.7572** |
| | | IoU | 0.6353 |
| | | Pixel Recall | 0.7817 |
| **Pipeline** — End-to-End | Hierarchical gating | Images filtered before segmentation | **85.06%** |
| | | Forwarded to Stage 2 | 14.94% |
| **Domain Shift** | DeepCrack (after adaptation) | Dice | 0.6553 |

---

## 🎯 Problem

Manual inspection of concrete infrastructure — bridge decks, pavements, tunnels, retaining walls — is slow, subjective, and difficult to scale. Most deep learning crack-detection studies evaluate classification or segmentation in isolation, ignoring how errors propagate through a real inspection pipeline. Practical inspection requires a complete workflow that can screen large image collections, localize crack pixels, estimate crack geometry, and explain how errors propagate from one stage to the next.

This project addresses that gap with a **reliability-aware hierarchical pipeline** that connects all four concerns in one framework.

## 🧠 Approach

A three-stage coarse-to-fine framework where each stage feeds the next:

```
Input image
   ↓
[Stage 1] MobileNetV2 → crack / non-crack
   ↓ (only crack-positive forwarded)
[Stage 2] SegFormer-B2 @ 384×384 → pixel-level mask
   ↓
[Stage 3] Geometry analysis → length, width descriptors
```

**Why this design:**
- **Compute efficiency** — Stage 1 filters 85% of images before the expensive segmentation step
- **Pipeline-aware evaluation** — errors at each stage are explicitly tracked (missed-crack rate, pipeline recall)
- **Reliability framing** — geometry outputs are treated as calibration-sensitive, not certified measurements

### Stage 1 — Classification (MobileNetV2)
Lightweight gate selected for its efficiency. Trained with augmentation, scheduler-based optimization, early stopping, and checkpointing. Optimized for the dual role of accuracy *and* downstream forwarding behavior.

### Stage 2 — Segmentation (SegFormer-B2)
The strongest of **10+ architectures benchmarked**:
- U-Net + BCE/Dice (baseline)
- Boundary-Skeleton-Aware U-Net
- Geometry-Width U-Net
- Weighted ensemble + TTA
- Patch-focused U-Net
- Full-image 384 U-Net
- SegFormer-B0
- U-Net++ with ResNet34 encoder
- U-Net++ post-processed ensemble
- EfficientNet-B4 U-Net++ @ 320
- **SegFormer-B2 @ 384×384 + hvflip TTA** ✅ (selected)

Final configuration uses validation-selected thresholding (0.60) and horizontal-vertical flip test-time augmentation.

### Stage 3 — Geometric Characterization
From the predicted binary mask: crack area, skeleton length, and approximate width are estimated in pixel units. When a ruler reference is available, pixel measurements are converted to millimeters. Results are framed as exploratory severity indicators rather than engineering-grade measurements.

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![SegFormer](https://img.shields.io/badge/SegFormer-FFD21E?logoColor=black)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-FF6F00?logoColor=white)
![Transfer Learning](https://img.shields.io/badge/Transfer%20Learning-0F5E5E?logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

## 📦 Datasets

| Dataset | Stage | Size | Label Type |
|---|---|---|---|
| **SDNET2018** | Stage 1 classification | 56,000+ image patches | Image-level crack / non-crack |
| **Project crack segmentation** | Stage 2 segmentation | 11,298 image-mask pairs (8,162 / 1,441 / 1,695) | Pixel-level binary masks |
| **DeepCrack** | Domain-shift testing | 537 RGB images (300 train, 237 test) | Pixel-level masks |
| **Field images** | Stage 3 geometry validation | 45 ruler-calibrated images | Manual length / width |

## 📁 Project Structure

```
hierarchical-crack-detection/
├── Crack_Detection_Project.ipynb     # Full pipeline notebook (all 3 stages)
├── paper/
│   └── Hierarchical_Concrete_Crack_Detection_Final_paper.docx
├── images/                            # README assets (add screenshots here)
└── README.md
```

## 🚀 Reproducing the Results

The full pipeline is implemented in `Crack_Detection_Project.ipynb` and is structured to match the paper sections:

1. Dataset Summary
2. Stage 1 — Crack / Non-Crack Classification
3. Stage 2 — Crack Segmentation
4. Stage 3 — Geometric Characterization
5. Benchmarking
6. Ablation Study
7. Statistical Validation
8. Error Analysis and Failure Case Study
9. Computational Efficiency Analysis
10. Limitations and Future Work

To run locally:
```bash
# Open the notebook in Jupyter or Colab
jupyter notebook Crack_Detection_Project.ipynb
```

Datasets must be downloaded separately from their original sources (SDNET2018 from Utah State University DigitalCommons; DeepCrack from its benchmark repository).

## 📈 Stage-by-Stage Detail

### Stage 1 (MobileNetV2) — Single-Run Test Performance
| Metric | Value |
|---|---|
| Accuracy | 0.9168 |
| Precision | 0.7061 |
| Recall | 0.7704 |
| Specificity | 0.9429 |
| Macro-F1 | 0.8437 |
| ROC-AUC | 0.9341 |
| PR-AUC | 0.8397 |
| MCC | 0.6885 |

### Stage 2 Segmentation Progression
| Configuration | Dice | IoU |
|---|---|---|
| Untuned U-Net baseline | 0.7318 | 0.6126 |
| Tuned U-Net | 0.7350 | 0.6161 |
| Boundary-Skeleton-Aware U-Net | 0.7505 | 0.6322 |
| Geometry-Width U-Net | 0.7495 | 0.6309 |
| Weighted ensemble + TTA | 0.7510 | 0.6326 |
| **SegFormer-B2 384 + hvflip TTA** | **0.7572** | **0.6353** |

### Pipeline Handoff Summary
| Metric | Value |
|---|---|
| Pipeline recall | 0.7461 |
| Pipeline precision | 0.7550 |
| Missed crack rate | 25.39% |
| Images forwarded to Stage 2 | 14.94% |
| Images filtered | 85.06% |

## 🔍 Key Engineering Insights

- **The 25.39% missed-crack rate is the operational risk**, not the classification accuracy. In safety-critical deployments, the Stage 1 threshold should be tuned for recall, not for isolated accuracy.
- **Transformer models outperformed CNNs** at the final benchmark: SegFormer-B2 beat the weighted CNN ensemble (Dice 0.7572 vs 0.7510).
- **Domain shift is real** — DeepCrack Dice dropped substantially under direct transfer, only recovering to 0.6553 after two-stage adaptation. Field deployment must include site-specific fine-tuning.
- **Pixel-level geometry is calibration-sensitive** — without a ruler reference and verified field measurements, mask-derived crack width should not be treated as engineering-grade measurement.

## ⚠️ Limitations

- Stage 1 and Stage 2 use different datasets (SDNET2018 vs project segmentation set) because no single public dataset provides both image-level labels and pixel masks at scale
- Geometry validation used only 45 ruler-calibrated field images
- Some single-run improvements lacked paired per-image outputs for rigorous statistical comparison

Future work: unified inspection dataset with image labels, pixel masks, physical scale references, and consistent field metadata.

## 📄 Paper

The full research paper is included in the `paper/` directory. See `Hierarchical_Concrete_Crack_Detection_Final_paper.docx` for the complete methodology, dataset descriptions, equations, results tables, and references.

**Authors:** Kavya Kolichelam, Wisam Bukaita, Ph.D.
**Affiliation:** Department of Mathematics and Computer Science, Lawrence Technological University, Southfield, MI, USA

## 📬 Contact

**Kavya Kolichelam**
📧 kavyakolichelam@gmail.com
💼 [LinkedIn](https://www.linkedin.com/in/kavya-kolichelam-54452a317)
💻 [GitHub](https://github.com/Kavyakolichelam)

---

⭐ If you found this project interesting, please consider starring the repository.
