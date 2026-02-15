# 🚗 Dashcam Raindrop Removal

## 📌 Project Overview
**Real-time video restoration for autonomous driving.**
This project implements a deep learning pipeline to remove raindrops, streaks, and fog from dashcam footage, restoring visibility for both human drivers and downstream computer vision tasks (e.g., object detection).

The model is built on a **MobileNetV3-UNet** architecture, optimized for speed and deployed with a custom **"Crapification"** pipeline to generate synthetic training data from clean driving sequences.

---

## 🚀 Key Features
- **⚡ Real-Time Performance**: Achieves **>30 FPS** on standard GPUs (RTX 3060/4090).
- **🏗️ Two-Stage Curriculum Training**:
  - **Stage 1**: Structural Recovery (Pixel Loss only).
  - **Stage 2**: Texture Refinement (SSIM + Edge + Perceptual Loss).
- **🌧️ Synthetic "Crapification" Pipeline**:
  - Generates realistic rain streaks, droplets, and depth-based fog.
  - Uses **MiDaS** for depth estimation and **RainStreakDB** for masks.
- **🎥 Temporal Consistency**:
  - Includes experimental support for **ConvLSTM** to enforce frame-to-frame coherency.

---

## 📂 Dataset
This project utilizes high-quality urban driving sequences from the **Wayve Open Dataset**.
- **Source**: [Wayve.ai Data](https://wayve.ai/data/)
- **Role**: The "Clean" frames from Wayve serve as the ground truth. Our pipeline artificially degrades them to create paired `(Rainy, Clean)` training samples.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/DashcamRaindropsRemoval.git
   cd DashcamRaindropsRemoval
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Requires Python 3.8+ and PyTorch with CUDA support.*

---

## 🏃 Usage

### 1. Data Generation ("Crapification")
Generate synthetic rainy data from your clean videos.
```bash
python crapification/per_video_crapification_pipeline.py
```

### 2. Training
The model is trained in two stages for stability.

**Stage 1: Structural Warmup**
```bash
python training/train_stage_1.py
```
*Goal: Remove rain and restore basic scene structure (loss: L1).*

**Stage 2: Fine-Tuning**
```bash
python training/train_stage_2.py
# OR for ConvLSTM experiment:
python training/training_attempts/train_with_convlstm.py
```
*Goal: Recover fine textures (asphalt, signs) and sharpen edges using Perceptual + SSIM losses.*

### 3. Testing & Evaluation
Run inference on the test set and calculate metrics (PSNR, SSIM).
```bash
python testing/testing.py
```

---

## 📊 Results

| Metric | Stage 1 (Baseline) | Stage 2 (Final) |
| :--- | :---: | :---: |
| **PSNR** | ~26 dB | **~30 dB** |
| **SSIM** | ~0.82 | **~0.90** |
| **FPS** | 35+ | **33+** |

*Visual results show significant removal of occluding droplets while preserving road markings and traffic lights.*

---

## 🔮 Future Work (Potential Improvements)
- **Vision Transformers (SwinIR)**: To capture long-range dependencies.
- **Unsupervised Domain Adaptation (CycleGAN)**: To bridge the gap between synthetic and real-world rain.
- **End-to-End Optimization**: Training jointly with YOLO ensuring restoration improves detection accuracy.

---

**University Project | 2026**
