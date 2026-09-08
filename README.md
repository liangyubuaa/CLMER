# 🧠 CLMER: Contrastive Learning-based Multi-modal Emotion Recognition

[![Paper](https://img.shields.io/badge/Paper-TNNLS%20-blue)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/liangyubuaa/CLMER)

> **CLMER** is a contrastive learning‑based multi‑modal fusion framework for emotion recognition, integrating physiological signals (EEG‑centric) with visual facial semantics. It achieves **state‑of‑the‑art** performance on DEAP, AMIGOS, and our private **MAN‑II** dataset.

✅ **Accepted by *IEEE Transactions on Neural Networks and Learning Systems (TNNLS)***.

---

## 📖 Overview

Emotion recognition is critical for human‑computer interaction and affective computing. However, fusing heterogeneous modalities—such as continuous physiological signals and spatial visual data—poses significant challenges due to structural mismatches. **CLMER** tackles this by:

- 🔄 **Serializing** visual data into time‑series format via low‑dimensional semantic features (3D landmarks and Action Units), achieving precise temporal alignment with physiological signals.
- 🔬 Applying **self‑supervised contrastive learning** (combining sample‑level and temporal predictive losses) to extract robust, temporally aware representations from both modalities.
- 🤝 Employing **cross‑modal attention** for effective multi‑modal fusion.

The framework substantially outperforms unimodal and traditional fusion baselines across multiple datasets and classification granularities (binary, 4‑class, 9‑class).

---

## ⭐ Key Contributions

1. **🧩 Novel Framework** – A contrastive learning‑based multi‑modal fusion architecture that integrates internal physiological data (EEG, ECG, GSR, etc.) with external visual cues (facial landmarks and AUs).

2. **📐 Cross‑modal Serialization Strategy** – Converts pixel‑level visual inputs into time‑series data via:
   - Low‑dimensional semantic extraction (3D landmarks, Action Units)
   - Tailored temporal windowing (coarse‑grained `k=1` or fine‑grained `k>1`) to synchronize with physiological sampling rates

3. **⏳ Time‑Series Contrastive Learning** – Adapts SeqCLR‑style contrastive loss with both *sample contrastive* and *temporal predictive* objectives, capturing discriminative inter‑sample features and dynamic temporal evolution.

4. **🏆 State‑of‑the‑Art Performance** – Achieves **97.65%** (binary), **96.47%** (4‑class), and **96.09%** (9‑class) accuracy on DEAP; **88.37%** (valence binary) and **81.42%** (4‑class) on AMIGOS short; **97.75%** (4‑class) on our private **MAN‑II** dataset.

---

## 🏗️ Framework Architecture

CLMER consists of three sequential modules:

| Module | Description |
|--------|-------------|
| **📥 Multi‑modal Data Preparation** | Physiological signals: bandpass filtering, resampling to 128 Hz. Visual data: extract 3D landmarks (68 points) and AUs via OpenFace; apply temporal windowing (`k` slices per second) to align with physiological data. |
| **⚡ Contrastive Feature Extraction** | Self‑supervised learning on time‑series data. Augmentations: scaling, jittering. Loss: `L_sequence = λ₁·L_sample + λ₂·L_temporal`, with optimal λ₁=λ₂=1.0. |
| **🔗 Multi‑modal Fusion** | Cross‑modal attention (multi‑head) between physiological and visual modalities. Fuses features and performs classification (binary/4‑class/9‑class). |

---

## 📁 Project Structure

```
CLMER/
├── config_files/                       # ⚙️ Configuration files for datasets
│   ├── deap9classphysio_Configs.py     # DEAP 9-class config
│   └── amigos4classphysio_Configs.py   # AMIGOS 4-class config
├── dataloader/                         # 📂 Data loading utilities
├── models/                             # 🧠 Core model implementations
│   ├── attention.py                    # Attention mechanisms
│   ├── encoder.py                      # Feature encoders
│   ├── fusion.py                       # Multi-modal fusion module
│   ├── loss.py                         # Contrastive loss functions
│   └── timeseries.py                   # Time-series processing
├── modules/                            # 🔧 Reusable building blocks
│   ├── multihead_attention.py          # Multi-head attention
│   ├── position_embedding.py           # Positional encoding
│   └── transformer.py                  # Transformer modules
├── src/                                # 📜 Main scripts
│   ├── cltrain.py                      # Contrastive learning training
│   ├── ftrain.py                       # Fusion module training
│   ├── dataset.py                      # Dataset definitions
│   ├── confusion_matrix.py             # Confusion matrix utilities
│   ├── eval_metrics.py                 # Evaluation metrics
│   └── utils.py                        # Helper functions
├── clmain.py                           # 🚀 Entry point for feature extraction phase
├── fmain.py                            # 🚀 Entry point for fusion phase
└── requirements.txt                    # 📦 Dependency list
```

---

## 📊 Datasets

| Dataset | Modalities | Details |
|---------|------------|---------|
| **DEAP** | 32‑ch EEG + peripheral (GSR, EMG, BVP, temp, respiration) + visual | 40 channels, 32 subjects, 40 trials. Labels: valence/arousal (binary, 4‑quadrant, 9‑grid). Visual alignment: `k=1`. |
| **AMIGOS** | 14‑ch EEG + ECG + GSR + visual | Short videos (40 subjects, 16 clips) and long videos (17 subjects, 4 sequences). Labels: valence/arousal (binary, 4‑class). Visual alignment: `k=1`. |
| **MAN‑II (private)** | 14‑ch EEG + visual | 4 emotional states (moved, angry, nervous, reproachful). Visual includes 3D landmarks + eye gaze. Fine‑grained alignment: `k=4`. |

---

## 📈 Results

| Dataset | Task | Accuracy (%) | F1 (%) |
|---------|------|--------------|--------|
| DEAP | Binary (Valence/Arousal) | **97.65 ± 0.71** | 97.65 ± 0.72 |
| DEAP | 4‑class (VA quadrants) | **96.47 ± 1.29** | 96.47 ± 1.28 |
| DEAP | 9‑class (VA grid) | **96.09 ± 0.55** | 96.09 ± 0.55 |
| AMIGOS (short) | Valence (binary) | **88.37 ± 0.61** | 88.37 ± 0.58 |
| AMIGOS (short) | Arousal (binary) | **83.56 ± 0.97** | 83.53 ± 0.96 |
| AMIGOS (short) | 4‑class VA | **81.42 ± 1.19** | 81.41 ± 1.16 |
| AMIGOS (long) | 4‑class VA | **90.83 ± 1.01** | 90.82 ± 1.03 |
| **MAN‑II (private)** | 4‑class | **97.75 ± 1.63** | 97.75 ± 1.62 |

---

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/liangyubuaa/CLMER.git
cd CLMER

# Create a virtual environment (optional)
conda create -n clmer python=3.8
conda activate clmer

# Install dependencies
pip install -r requirements.txt
```

**📦 Requirements:**
- PyTorch 1.8.1+
- CUDA 11.1+
- numpy, scipy
- opencv‑python
- OpenFace (for visual feature extraction)

---

## 🚀 Usage

### 1. Data Preparation

Prepare your dataset following the structure expected by the dataloader. For DEAP and AMIGOS, download the raw data and run preprocessing scripts to extract:
- Physiological signals (EEG, ECG, GSR, etc.)
- Visual semantics (3D landmarks, Action Units) via OpenFace

### 2. Training

CLMER has **two training phases**:

#### Phase 1: Contrastive Feature Extraction

```bash
python clmain.py --config config_files/deap9classphysio_Configs.py
```

This phase:
- Loads serialized time‑series data from both modalities
- Applies self‑supervised contrastive learning (sample‑level + temporal losses)
- Saves pre‑trained encoder weights for the fusion phase

#### Phase 2: Multi‑modal Fusion

```bash
python fmain.py --config config_files/deap9classphysio_Configs.py
```

This phase:
- Loads pre‑trained features from Phase 1
- Applies cross‑modal attention for modality fusion
- Performs emotion classification (binary/4‑class/9‑class)

### 3. Evaluation

```bash
python src/eval_metrics.py --model_path checkpoints/best_model.pth
```

### ⚙️ Configuration

Configuration files are located in `config_files/`:
- `deap9classphysio_Configs.py` – DEAP dataset, 9‑class classification
- `amigos4classphysio_Configs.py` – AMIGOS dataset, 4‑class classification

Key configurable parameters:
- `lambda1`, `lambda2` – Contrastive loss weights (optimal: 1.0, 1.0)
- `token_length` – Fusion token length (DEAP: 50–70; AMIGOS: 10–20)
- `batch_size`, `learning_rate`, `epochs`
- `k` – Temporal windowing parameter (coarse: 1, fine: >1)

---

## ⚡ Computational Efficiency

| Method | GPU Memory (MiB) | Training Time (s) | Accuracy (DEAP 9‑class) |
|--------|------------------|-------------------|--------------------------|
| Downsampling | 2172 | 1908 | 81.21 |
| MobileViT_s  | 825  | 267  | 87.58 |
| **Serialization (CLMER)** | **2534** | **2015** | **94.91** (w/o CL) / **96.09** (w/ CL) |

> CLMER achieves the best accuracy with moderate resource consumption, proving its lightweight and practical deployability.

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{niu2024clmer,
  author  = {Shuang Niu, Jian He, Yu Liang, Xiyuan Hu, Shahid Mumtaz, Kaixin Wang, Tianhao Peng, Chenlong Zhang, Zaitian Wang},
  title   = {CLMER: a Framework for Contrastive Learning‑based Multi‑modal Emotion Recognition},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2024},
  note    = {Accepted for publication}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 📬 Contact

For any questions, please open an issue or contact the corresponding author via the paper's affiliation email.

---

*🔗 Code and pre‑trained models are available at [https://github.com/liangyubuaa/CLMER](https://github.com/liangyubuaa/CLMER).*
