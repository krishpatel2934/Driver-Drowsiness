# 🧠 Driver Drowsiness Detection
> Real-time driver fatigue monitoring using BiGRU + Multi-Head Attention + Focal Loss

[![Python](https://img.shields.io/badge/Python-3.9--3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.13-green)](https://mediapipe.dev/)

A real-time drowsiness detection system that classifies driver vigilance into three levels — **Alert**, **Low-vigilant**, and **Drowsy** — directly from a live webcam feed. Built with a BiGRU + Multi-Head Attention architecture trained with Focal Loss to handle class imbalance, deployed as an interactive Streamlit dashboard. 

---

## Table of Contents
- [Demo](#demo)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Feature Engineering](#feature-engineering)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [System Requirements](#system-requirements)
- [Dependencies](#dependencies)

---

## Demo

Once running, the app provides:
- Live webcam feed with facial landmark overlays (eyes, mouth, head axis)
- Real-time fatigue score gauge (0–100)
- Per-class probability bars (Alert / Low-vigilant / Drowsy)
- Live score trend chart
- Per-subject baseline calibration

| Score Range | Status | Meaning |
|:-----------:|:------:|---------|
| 0 – 34 | ✅ ALERT | Driver is fully awake |
| 35 – 64 | ⚠️ CAUTION | Early signs of fatigue detected |
| 65 – 100 | 🚨 DROWSY | Immediate attention required |

---

## Dataset

This project uses the **UTA Real-Life Drowsiness Dataset (UTA-RLDD)**.

> 📎 Dataset page: [https://sites.google.com/view/utarldd/home](https://sites.google.com/view/utarldd/home)

### About UTA-RLDD

UTA-RLDD is a real-life drowsiness dataset collected under naturalistic driving conditions. Unlike lab-based datasets, subjects were recorded in their own vehicles without scripted behaviour, making it significantly more challenging and realistic.

| Property | Details |
|----------|---------|
| Subjects | 60 participants |
| Videos per subject | 3 (one per drowsiness level) |
| Total videos | 180 |
| Recording condition | Real vehicle, natural lighting |
| Labels | 0 = Alert, 5 = Low-vigilant, 10 = Drowsy |
| Format | RGB video (.mov / .mp4) |
| Folds | Fold1, Fold2 (used in this project) |

### Label Mapping

The dataset uses numeric folder names as labels:

| Folder Name | Mapped Label | Class |
|:-----------:|:------------:|-------|
| `0` | 0 | Alert |
| `5` | 1 | Low-vigilant |
| `10` | 2 | Drowsy |

### Accessing the Dataset

The dataset must be requested directly from the authors via the link above. Once downloaded, organise it as:

```
DL_project/
└── Dataset/
    ├── Fold1_part1/
    │   ├── subject_01/
    │   │   ├── 0.mov
    │   │   ├── 5.mov
    │   │   └── 10.mov
    │   └── ...
    ├── Fold1_part2/
    ├── Fold2_part1/
    └── Fold2_part2/
```

> The dataset is **not included** in this repository. You must obtain it independently from the authors.

---

## Model Architecture

**BiGRU + Multi-Head Additive Attention + Focal Loss**

```
Input Sequence  (600 frames × 16 features)   ← training
                (150 frames × 16 features)   ← inference (first score in ~5 sec)
        │
        ▼
┌─────────────────────────────────────┐
│  Bidirectional GRU                  │
│  hidden_size = 128,  num_layers = 2 │
│  output = 256  (128 × 2 directions) │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Multi-Head Additive Attention      │
│  num_heads = 4,  head_dim = 64      │
│  score = W2 · tanh(W1 · h_t)       │
│  → softmax → weighted context       │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Fully-Connected Head               │
│  256 → 128 → 64 → 3 classes         │
│  ReLU + Dropout(0.3)                │
└─────────────────────────────────────┘
        │
        ▼
   [Alert,  Low-vigilant,  Drowsy]
```

**Design choices:**

- **Bidirectional GRU** — processes the frame sequence in both forward and backward directions, capturing context from both past and future frames. More parameter-efficient than LSTM with comparable performance
- **Multi-Head Additive Attention** — learns which frames in the sequence are most indicative of drowsiness. 4 independent heads attend to different temporal patterns simultaneously (e.g., one head may focus on sustained eye closure, another on yawning bursts)
- **Focal Loss (γ=2.0)** — down-weights easy examples (Alert is the majority class) and forces the model to focus on hard-to-classify Low-vigilant frames. Combined with inverse-frequency class weights for maximum imbalance correction

---

## Feature Engineering

Each frame is represented as a **16-dimensional feature vector** derived from 4 base facial measurements extracted using MediaPipe Face Landmarker (478 landmarks):

| Group | Features | Description |
|-------|----------|-------------|
| Raw | `ear`, `mar`, `head_tilt`, `perclos` | Direct per-frame facial measurements |
| Per-subject normalised | `ear_pn`, `mar_pn`, `head_tilt_pn`, `perclos_pn` | Z-scored against each subject's own alert-state baseline |
| Frame delta | `ear_Δ`, `mar_Δ`, `head_tilt_Δ`, `perclos_Δ` | Frame-to-frame rate of change |
| Rolling mean (90 frames) | `ear_r90`, `mar_r90`, `head_tilt_r90`, `perclos_r90` | Smoothed 3-second trend |

**Base feature definitions:**

| Feature | Computation | Drowsiness Signal |
|---------|------------|-------------------|
| **EAR** — Eye Aspect Ratio | `(‖p1−p5‖ + ‖p2−p4‖) / (2 × ‖p0−p3‖)` using 6 eye landmarks | ↓ as eyes close |
| **MAR** — Mouth Aspect Ratio | `vertical opening / horizontal width` using 8 mouth landmarks | ↑ during yawning |
| **Head Tilt** | `‖nose−chin‖ / ‖left_ear−right_ear‖` | ↑ as head nods forward |
| **PERCLOS** | `% of frames with EAR < 0.21` over a 60-second rolling window | ↑ with sustained eye closure |

The per-subject normalisation is critical — it removes inter-subject variability (e.g., naturally small eyes vs. large eyes) so the model learns deviation from each person's own alert baseline rather than absolute values.

---

## Project Structure

```
Driver_Drowsiness/
│
├── app.py                  # Streamlit real-time detection dashboard
├── models.py               # BiGRU_MHA model + attention architecture definition
├── train.py                # LOSO-CV training script with Focal Loss (Kaggle/Colab)
├── extract_features.py     # Feature extraction from raw UTA-RLDD videos (Colab)
├── eda_visualise.py        # EDA: distributions, time-series, boxplots (Colab)
├── setup_and_unzip.py      # Colab environment setup: install libs + unzip dataset
├── requirements.txt        # Python dependencies
│
├── v5_focal_models/        # ⚠️ Create this folder manually — not tracked in git
│   ├── drowsiness_bigru_mha_v5_focal.pth   # Trained model weights (~5 MB)
│   └── scaler_v5.pkl                        # Fitted StandardScaler
│
└── face_landmarker.task    # ⚠️ Download separately — not tracked in git (~29 MB)
```

> `extract_features.py`, `eda_visualise.py`, and `setup_and_unzip.py` are **Colab/Kaggle scripts** for reproducing the training pipeline. They are **not needed** to run the app locally — only `app.py`, `models.py`, the model weights, scaler, and face landmarker task file are required.

---

## Quick Start

### Prerequisites
- Python **3.9, 3.10, or 3.11** — mediapipe 0.10.13 does **not** support Python 3.12+
- A working webcam
- ~500 MB free disk space

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Driver_Drowsiness
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If you have a CUDA GPU, install the matching PyTorch version from [pytorch.org](https://pytorch.org/get-started/locally/) for faster inference. The app works fine on CPU.

### 4. Create the models folder

```bash
# Windows
mkdir v5_focal_models

# macOS / Linux
mkdir v5_focal_models
```

### 5. Download required binary files

**MediaPipe face landmarker model** (~29 MB) — paste this in your terminal:

```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker.task"

# macOS / Linux
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**Trained model weights + scaler** — download from the shared link provided separately and place them inside `v5_focal_models/`:

```
v5_focal_models/
├── drowsiness_bigru_mha_v5_focal.pth
└── scaler_v5.pkl
```

### 6. Verify your folder looks like this

```
Driver_Drowsiness/
├── app.py
├── models.py
├── face_landmarker.task          ✅ downloaded
├── v5_focal_models/
│   ├── drowsiness_bigru_mha_v5_focal.pth   ✅ downloaded
│   └── scaler_v5.pkl                        ✅ downloaded
└── ...
```

### 7. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

**How to use:**
1. Click **▶ Start** and allow webcam access when prompted
2. Sit normally — the app calibrates to your face for ~15 seconds (baseline collection)
3. Once calibrated, the fatigue score updates every ~3 frames in real time
4. The gauge, probability bars, and trend chart all update live
5. Click **■ Stop** to end the session

---

## Training Pipeline

> Training scripts are designed for **Kaggle** (with GPU) or **Google Colab** with Google Drive. You do not need to retrain to use the app — use the provided weights.

The model was trained using **Leave-One-Subject-Out Cross-Validation (LOSO-CV)** on the UTA-RLDD dataset. Each subject is held out as the test set in turn, ensuring the model is evaluated on completely unseen drivers — the most rigorous evaluation for person-independent drowsiness detection.

### Step 1 — Setup environment (Colab only)

```bash
python setup_and_unzip.py
```

Installs all libraries, downloads the MediaPipe face landmarker model, and extracts the UTA-RLDD zip files from Google Drive into Colab's local SSD.

### Step 2 — Extract features from raw videos

```bash
python extract_features.py
```

Processes every video frame-by-frame using MediaPipe, computes EAR / MAR / Head Tilt / PERCLOS for each frame, and saves per-fold CSVs to Google Drive. Supports **smart resume** — already-processed folds are detected and skipped automatically.

Output: `features_all.csv` saved to your Drive processed folder.

### Step 3 — Clean the dataset

> ⚠️ **This step is manual and required before training.**

Open `features_all.csv` and:
- Remove subjects with face detection rate below ~80% (check `face_detected` column)
- Remove any corrupted or incomplete sessions

Save the cleaned file as **`features_all_clean1.csv`** in the same Drive folder. The training script reads this exact filename.

### Step 4 — EDA (optional but recommended)

```bash
python eda_visualise.py
```

Generates 4 diagnostic plots saved to Drive:
- Feature distributions per drowsiness class
- Per-subject feature time-series
- Box plots comparing classes
- Face detection rate per subject

### Step 5 — Train the model

```bash
python train.py
```

> Before running, update the paths at the top of `train.py` to match your Kaggle dataset input path and working directory.

Key training configuration:

| Parameter | Value |
|-----------|-------|
| Sequence length | 600 frames (~20 sec at 30fps) |
| Stride | 30 frames |
| Batch size | 32 |
| Epochs | 60 (early stopping, patience = 10) |
| Optimizer | Adam (lr = 5e-4, weight_decay = 1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | Focal Loss (γ = 2.0) + inverse-frequency class weights |
| Validation | LOSO-CV |
| Metric | Macro F1-score |

Outputs saved to `v5_focal_models/`:
- `drowsiness_bigru_mha_v5_focal.pth` — model weights, config, and per-subject F1 scores
- `scaler_v5.pkl` — fitted StandardScaler for inference
- `metrics_bigru_mha_v5_focal.json` — full LOSO results per subject

---

## Results

Model progression across all experiments on UTA-RLDD (LOSO Macro F1-score):

| Version | Architecture | Loss Function | LOSO Macro F1 |
|:-------:|-------------|--------------|:-------------:|
| v3 | BiGRU | Cross-Entropy | 0.6799 |
| v4 | BiGRU + Multi-Head Attention | Cross-Entropy | 0.6799 |
| **v5** | **BiGRU + Multi-Head Attention** | **Focal Loss (γ=2.0)** | **see metrics JSON** |

The key improvement in v5 is on the **Low-vigilant** class — the most safety-critical state and the hardest to detect due to its subtle, transitional visual cues. Focal Loss specifically targets this by penalising misclassifications of hard examples more heavily.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.10 or 3.11 |
| RAM | 4 GB | 8 GB |
| GPU | Not required | CUDA-capable (faster inference) |
| Webcam | Any USB / built-in | 720p at 30 fps |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | — |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.28.0 | Real-time web dashboard |
| `mediapipe` | == 0.10.13 | Face landmark detection (478 landmarks) |
| `opencv-python` | latest | Webcam capture and frame processing |
| `torch` | latest stable | Model definition and inference |
| `torchvision` | latest stable | PyTorch vision utilities |
| `scikit-learn` | latest | StandardScaler, F1-score, classification report |
| `pandas` | latest | Data loading and feature engineering |
| `numpy` | latest | Numerical operations |
| `matplotlib` | latest | Training plots and EDA visualisations |
| `tqdm` | latest | Training progress bars |

```bash
pip install -r requirements.txt
```

---

## Citation

If you use the UTA-RLDD dataset in your work, please cite the original authors:

```
Ghoddoosian, R., Galib, M., & Athitsos, V. (2019).
A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection.
CVPR Workshops.
https://sites.google.com/view/utarldd/home
```
