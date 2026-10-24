# Handwritten Kanji Recognition System
An end-to-end computer vision and deep learning project for recognizing handwritten Japanese Kanji characters, featuring custom architectures trained from scratch and an interactive local web demo.

[🇻🇳 Bản tiếng Việt](README_VI.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?logo=flask)](https://flask.palletsprojects.com/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.73%25-brightgreen)](#6-evaluation--comparative-analysis)

<p align="center">
  <img src="assets/Screenshot%202026-10-24%20205057.png" width="900" alt="Web Demo Interface Preview"/>
  <br>
  <em>Interactive Web Interface: Real-time dual-model comparison on user-drawn Kanji character 時 (Time).</em>
</p>

---

## 1. Overview

Handwritten Kanji recognition is challenging due to the large character inventory, intricate stroke arrangements, and significant handwriting distortion, especially in calligraphic or cursive (Kuzushiji) styles. Characters sharing identical radicals can be difficult to separate under loose handwriting.

This repository covers the complete machine learning lifecycle:
- **Dataset engineering:** Class filtering, stratified splitting, and an adaptive polarity conversion pipeline to reconcile drawing-canvas strokes with training data.
- **Model development from scratch:** 
  - A 4-stage **Custom CNN** designed for fast inference with small memory footprint.
  - A custom **ResNet-18** implementation built with residual blocks to mitigate vanishing gradients without using pre-trained weights.
  - A comprehensive benchmark against a **previous baseline ResNet-18** model to rigorously validate optimizations.
- **Local deployment:** A Flask backend serving predictions sequentially to a responsive HTML5 Canvas frontend for real-time comparison between models.

---

## 2. Key Features

- **Trained From Scratch:** Both primary models (CNN and ResNet-18) were designed and trained without transfer learning to evaluate pure feature learning capacity on historical character forms.
- **Outperforming Baseline Research:** The proposed ResNet-18 achieves **98.73% test accuracy** across 100 classes, outperforming the previous ResNet-18 baseline (**98.27%** on 92 classes) while lowering test loss from **0.0672 to 0.0556** (-17.3%).
- **Adaptive Stroke Polarity Preprocessing:** Automatically detects drawing contrast and inverts canvas input (black strokes on white canvas) to match training format (white strokes on black background).
- **Dual-Model Inference:** A single user drawing is evaluated sequentially by both CNN and ResNet-18, displaying Top-1 predictions, confidence scores, and Top-3 candidate distributions.
- **Handling Class Imbalance:** Implementation of inverse-frequency weighted cross-entropy in PyTorch for high-confidence classification across unequal class distributions.
- **Interactive Web Interface:** HTML5 drawing canvas ($400 \times 400$) with crosshair grid lines, adjustable brush size (2px–20px), and dark/light mode toggle.

---

## 3. System Architecture & Data Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           CLIENT (Browser UI)                             │
│  User draws on HTML5 Canvas (400x400) -> Export base64 PNG data URL       │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ HTTP POST /predict { image: base64 }
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           FLASK BACKEND (app.py)                          │
│                                                                           │
│  1. Decode base64 string -> PIL Grayscale Image ('L')                     │
│  2. Polarity Check: if mean(pixel) > 127 => invert (255 - pixel)          │
│  3. Lanczos Resize to target 64x64 resolution                             │
│  4. Normalize pixel intensities to float range [0.0, 1.0]                 │
│                                                                           │
│  Sequential Model Inference:                                              │
│  ├── Reshape to (1, 64, 64, 1) -> Custom CNN -> Top-1 & Top-3 Candidates  │
│  └── Stack to   (1, 64, 64, 3) -> ResNet-18  -> Top-1 & Top-3 Candidates  │
│                                                                           │
│  Return JSON: { cnn: { top1, confidence, top3 }, resnet: { ... } }        │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ JSON Response
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           RESULT VISUALIZATION                            │
│  Side-by-side comparison cards with confidence bars & Top-3 rank list     │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Dataset & Preprocessing

The experiments use the open-source **Kuzushiji-Kanji** dataset (ROIS-DS CODH):

- **Full Raw Dataset:** Contains 2,472 distinct character classes with severe frequency imbalance.
- **100-Class Benchmark Dataset (`100class/`):**
  - **Classes:** 100 selected high-frequency characters.
  - **Total Samples:** 29,365 grayscale images.
  - **Partitioning:** Stratified random split into Train (70% — 20,555 images), Validation (15% — 4,405 images), and Test (15% — 4,405 images).
- **Previous Baseline Dataset:**
  - Evaluated on a 92-class subset ($\ge 300$ samples/class), typically trained for only 15 epochs on CPU.

<p align="center">
  <img src="assets/class_distribution.png" width="850" alt="Class Distribution in Training Set"/>
  <br>
  <em>Figure 1: Long-tailed frequency distribution across character classes in the dataset. Sample counts drop steeply from >1,200 down to ~210 images, illustrating the class imbalance problem addressed by weighted loss.</em>
</p>

### Preprocessing & Augmentation Pipeline
1. **Polarity Adaptation:**
   The training dataset stores white strokes on a black background ($\text{mean} < 127$). In contrast, web canvas elements generate dark strokes on a light background ($\text{mean} > 127$). The backend applies:
   $$\text{img} = 255 - \text{img} \quad \text{if} \quad \frac{1}{HW}\sum_{i,j} \text{img}_{i,j} > 127$$
2. **Resizing:** Lanczos interpolation to $64 \times 64$ pixels.
3. **Normalization:** Intensities scaled to $[0.0, 1.0]$.
4. **Data Augmentation (Training only):**
   - Random rotation: $\pm 10^\circ$
   - Random translation: $\pm 10\%$ horizontal and vertical
   - Random zoom: $\pm 10\%$

---

## 5. Model Architectures

Three primary model configurations studied in the project:

### 1. Custom CNN (`training/CNN.ipynb`)
- **Framework:** TensorFlow / Keras
- **Input Shape:** $(64, 64, 1)$ — Single-channel grayscale
- **Structure:**
  - 4 convolutional blocks, each containing $2 \times \text{Conv2D}(3 \times 3)$ layers, Batch Normalization, ReLU activation, $2 \times 2$ Max Pooling, and 25% Spatial Dropout.
  - Channel progression: $32 \rightarrow 64 \rightarrow 128 \rightarrow 256$.
  - Classification head: `Flatten` $\rightarrow$ `Dense(512)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Dense(100, softmax)`.
- **Total Parameters:** 3,326,532 (~3.3M)

### 2. Proposed ResNet-18 (`training/RESNET18.ipynb`)
- **Framework:** TensorFlow / Keras (implemented from scratch)
- **Input Shape:** $(64, 64, 3)$ — Grayscale repeated across 3 channels
- **Structure:**
  - Initial convolutional layer followed by Batch Normalization, ReLU, and Max Pooling.
  - 4 residual stages with filter counts $[64, 128, 256, 512]$, each holding 2 residual blocks with identity shortcuts and projection shortcuts ($1 \times 1$ convolutions) where spatial dimensions downsample.
  - Global Average Pooling layer connected to a 100-class Softmax layer.
- **Total Parameters:** 11,229,732 (~11.2M)

### 3. Previous ResNet-18 Baseline (`training/training_notebook.ipynb`)
- **Framework:** PyTorch (CPU)
- **Classes:** 92 classes
- **Training Strategy:** 15 epochs, inverse-frequency class weights $w_c = \frac{N}{K \cdot N_c}$.

---

## 6. Evaluation & Comparative Analysis

### 6.1. Performance Summary of Proposed Models
The following metrics are verified directly from completed notebook runs and project final reports (Tables 2 & 3 in final evaluation documentation):

| Metric | Custom CNN (Keras) | Proposed ResNet-18 (Keras) | Architectural Comparison |
| :--- | :---: | :---: | :--- |
| **Number of Classes** | 100 classes | 100 classes | Same 100-class evaluation benchmark |
| **Test Set Size** | 4,405 samples | 4,405 samples | Unseen independent test set |
| **Accuracy (%)** | **98.52%** | **98.73%** | ResNet-18 achieves +0.21% higher accuracy |
| **Test Loss** | 0.0597 | **0.0556** | ResNet-18 achieves lower generalization loss |
| **Precision (%)** | 98.39% | **98.64%** | ResNet-18 has fewer false positives |
| **Recall (%)** | 98.26% | **98.45%** | ResNet-18 captures higher character coverage |
| **Macro Average F1** | 98.29% | **98.52%** | Balanced performance across all 100 classes |
| **Total Parameters** | **3,326,532 (3.3M)** | 11,229,732 (11.2M) | CNN is 3.3× more compact in memory |
| **Training Time** | **8.5 minutes** | 44.0 minutes | CNN trains 5.2× faster |
| **Epochs Trained** | 31 (Early Stopped at best ep 24) | 35 (Early Stopped at best ep 28) | Both early-stop cleanly via scheduler |
| **Model File Size** | **38.2 MB** | 128.7 MB | CNN requires 3.3× less disk/RAM footprint |

---

### 6.2. Comparison with Previous ResNet-18 Baseline
Performance comparison between the proposed ResNet-18 model and the previous baseline study:

| Metric | Previous ResNet-18 (Baseline) | Proposed ResNet-18 (Current Study) | Improvement Achieved |
| :--- | :---: | :---: | :--- |
| **Number of Classes** | 92 classes | **100 classes** | **Expanded search space (+8 classes)** |
| **Training Epochs** | 15 epochs (CPU) | **35 epochs** (Early Stopping) | Deeper training & complete convergence |
| **Test Loss** | 0.0672 | **0.0556** | **-17.3% lower loss** (better generalization) |
| **Test Accuracy** | 98.27% | **98.73%** | **+0.46% improvement** |
| **Macro F1-Score** | 98.07% | **98.52%** | **+0.45% improvement** |

#### Why the Proposed Model Outperformed the Baseline:
1. **Higher Accuracy in a Larger Hypothesis Space:** Expanding from 92 to 100 classes inherently increases decision boundary complexity. Despite this, the proposed ResNet-18 improved on all three primary metrics (Accuracy, Loss, Macro F1).
2. **Extended Optimization & Adaptive Scheduling:** The previous study terminated after 15 epochs, before weights fully stabilized in late-stage convergence. The proposed system extended the ceiling to 50 epochs and automatically stopped at epoch 35 (restoring best weights from epoch 28), guided by `ReduceLROnPlateau` to avoid plateau entrapment.
3. **Robust Augmentation & Polarity Adaptation:** Incorporating multi-angle rotation, spatial jittering, and contrast adaptation generated invariant feature maps that generalized significantly better to distorted real-world strokes.

---

### 6.3. Training Dynamics & Convergence Curves

<p align="center">
  <img src="assets/CNN.png" width="850" alt="Custom CNN Training Loss and Accuracy Curves"/>
  <br>
  <em>Figure 2: Custom CNN training dynamics over 31 epochs. Loss declines steadily from >2.1 to 0.03, while validation accuracy stabilizes around 98.5% with no overfitting gap.</em>
</p>

<p align="center">
  <img src="assets/REsnet18.png" width="850" alt="Custom ResNet-18 Training Loss and Accuracy Curves"/>
  <br>
  <em>Figure 3: Custom ResNet-18 training dynamics over 35 epochs. Residual connections accelerate convergence, reaching 99.9% training accuracy and 98.73% test accuracy.</em>
</p>

---

### 6.4. Computational Cost & Trade-off Analysis
- **Quality Benchmark:** ResNet-18 is the superior choice when absolute recognition accuracy is critical, particularly for intricate historical glyphs with dense, overlapping strokes.
- **Resource Benchmark:** Custom CNN provides an outstanding efficiency trade-off for edge and mobile environments:
  - **3.3× smaller parameter volume** (3.3M vs 11.2M).
  - **3.3× smaller model footprint** (38.2 MB vs 128.7 MB, easily hosted within GitHub's standard limits).
  - **5.2× faster training duration** (8.5 minutes vs 44.0 minutes).
  - Significantly lower RAM consumption during real-time serving.

---

## 7. Confusion Matrix & Error Analysis

Analysis of the Normalized Confusion Matrix across the **Top 20 most confusing character classes** reveals distinct architectural differences:

1. **True Positive Concentrations (Diagonal Values):**
   - **Custom CNN:** Diagonal values range between **0.85 and 0.97**.
   - **Proposed ResNet-18:** Demonstrates noticeably tighter, higher diagonal clustering between **0.93 and 0.97**.
2. **Suppression of Off-Diagonal Errors:**
   - In Custom CNN, confusing pairs exhibit off-diagonal error rates ranging between **0.04 and 0.15** (4%–15%).
   - In ResNet-18, these off-diagonal error rates are largely suppressed or reduced below 0.02.
3. **Primary Morphological Failure Modes:**
   - **Shared Radicals:** E.g., **今** (Now) confused with **分** (Divide) due to the shared top radical ($\boldsymbol{\hat{人}}$).
   - **Cursive Stroke Simplification:** E.g., **國** (Country) in cursive Kuzushiji being abbreviated into a single curved sweep, leading to occasional misclassification as **人** (Person).
   - **Radical Clusters:** Characters with the water radical `氵` (海, 汁, 江) occasionally swapped positions between Top-1 and Top-2, but consistently remained within the Top-3 candidates.

---

## 8. Interactive Web Demo & Qualitative Case Studies

The web demo serves as an empirical testbed for evaluating model behavior on real user drawings. Below are five verified qualitative case studies from live inference:

### Case 1: Baseline High-Confidence Recognition — 時 (Time)
<p align="center">
  <img src="assets/Screenshot%202026-10-24%20205057.png" width="850" alt="Recognition of 時"/>
</p>

- **Input:** Standard composition of **時** (Time - 日 + 寺).
- **Result:** Both models exhibit decisive high-confidence classification: **Custom CNN achieves 100%** confidence, while **ResNet-18 reaches 99%**. Top-3 rankings show negligible competitor probabilities.

---

### Case 2: Structural Radical Family & Candidate Ranking — 木 (Tree/Wood)
<p align="center">
  <img src="assets/Screenshot%202026-10-24%210531.png" width="850" alt="Recognition of 木"/>
</p>

- **Input:** 4-stroke character **木** (Tree) drawn with a thick brush (16px).
- **Result:** Both models accurately predict **木** as Top-1 (CNN: 93%, ResNet: 59%). Notably, the Top-3 candidates capture structurally related characters sharing the same radical base: **本** (Book/Origin, 7%) and **来** (Come), demonstrating meaningful semantic clustering in feature space.

---

### Case 3: Radical Feature Detection — 近 (Near)
<p align="center">
  <img src="assets/Screenshot%202026-10-24%20210942.png" width="850" alt="Recognition of 近"/>
</p>

- **Input:** Character **近** containing the movement radical 辶 (Quai xước) alongside 斤 (Cân).
- **Result:** Custom CNN predicts **近** at **98%**, and ResNet-18 at **95%**. Candidate #2 is **道** (Road), which also shares the 辶 radical, validating that intermediate convolutional filters successfully locate and encode morphological radical subunits.

---

### Case 4: Architecture Comparison on Complex Multi-Stroke Glyph — 春 (Spring)
<p align="center">
  <img src="assets/Screenshot%202026-10-24%20211153.png" width="850" alt="Comparison on 春"/>
</p>

- **Input:** Dense, 9-stroke character **春** (Spring) with parallel horizontal strokes and the lower 日 radical.
- **Result:** 
  - **Custom CNN struggles:** CNN gets confused by the stacked horizontal strokes and lower box structure, predicting **書** at **39%**, with **春** relegated to #2 (**21%**) and **蚕** at #3 (**15%**).
  - **ResNet-18 decisively succeeds:** ResNet-18 correctly and robustly predicts **春 at 97% confidence**!
- **Technical Takeaway:** This case visually validates the architectural advantage of ResNet-18. The residual skip connections preserve high-resolution spatial relationships across deeper layers, preventing stroke blurring that causes shallower CNNs to fail on dense characters.

---

### Case 5: Subtle Stroke Disambiguation — 松 (Pine)
<p align="center">
  <img src="assets/Screenshot%202026-10-24%20211441.png" width="850" alt="Recognition of 松"/>
</p>

- **Input:** Character **松** (composed of 木 + 公).
- **Result:** Both models deliver confident predictions for **松** (CNN: 92%, ResNet: 95%). Furthermore, candidate #3 captures **私** (composed of 禾 + 厶), demonstrating the system's sensitivity to fine radical boundaries between the tree radical (木) and the grain radical (禾).

---

## 9. Repository Structure

```
├── assets/                        # Visual documentation assets
│   ├── class_distribution.png     # Long-tailed class distribution plot (93 classes)
│   ├── CNN.png                    # Training loss and accuracy curves for Custom CNN
│   ├── REsnet18.png               # Training loss and accuracy curves for ResNet-18
│   ├── Screenshot 2026-10-24 205057.png  # Web demo: recognition of 時 (Time)
│   ├── Screenshot 2026-10-24 210531.png  # Web demo: recognition of 木 (Tree)
│   ├── Screenshot 2026-10-24 210942.png  # Web demo: recognition of 近 (Near)
│   ├── Screenshot 2026-10-24 211153.png  # Web demo: architectural comparison on 春 (Spring)
│   └── Screenshot 2026-10-24 211441.png  # Web demo: disambiguation of 松 (Pine)
│
├── demo/
│   ├── app.py                     # Flask server with preprocessing & sequential inference
│   ├── requirements.txt           # Python package dependencies for web demo
│   ├── frontend/
│   │   ├── index.html             # User interface with dual model comparison cards
│   │   ├── script.js              # Canvas interaction, base64 encoding & API calls
│   │   └── style.css              # Responsive layout with dark/light theme support
│   ├── label/
│   │   ├── labelscnn.json         # 100-class label map for Custom CNN (1 channel)
│   │   └── labelsresnet18.json    # 100-class label map for ResNet-18 (3 channels)
│   └── model/
│       └── .gitkeep               # Directory placeholder (weights excluded by .gitignore)
│
├── training/
│   ├── CNN.ipynb                  # 100-class Custom CNN training & evaluation notebook
│   ├── RESNET18.ipynb             # 100-class ResNet-18 training & evaluation notebook
│   └── training_notebook.ipynb    # 92-class PyTorch training with Weighted Loss
│
├── .gitignore                     # Git exclusion rules for datasets, binaries, and caches
├── README.md                      # English documentation
└── README_VI.md                   # Vietnamese documentation
```

---

## 10. Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `pip` package manager

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/kanji-handwriting-recognition.git
cd kanji-handwriting-recognition
```

### 2. Set up virtual environment
```bash
# Create environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r demo/requirements.txt
```

*Contents of `demo/requirements.txt`:*
```
flask>=3.0.0
flask-cors>=4.0.0
tensorflow>=2.15.0
numpy>=1.24.0
Pillow>=10.0.0
```

### 4. Set up model weights
Because `model_resnet18.h5` exceeds GitHub's 100 MB file limit (~128.7 MB), model weights are omitted from regular git commits. 

Place your trained weights inside `demo/model/`:
- `demo/model/model_cnn.h5`
- `demo/model/model_resnet18.h5`

*(You can export these weights by executing cells in `training/CNN.ipynb` and `training/RESNET18.ipynb`, or download them from repository releases if published).*

### 5. Run the web server
```bash
cd demo
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

### 6. Verify server health
You can check API status via:
```bash
curl http://localhost:5000/status
```
Expected output:
```json
{
  "server": "running",
  "cnn_model": "loaded",
  "resnet_model": "loaded",
  "cnn_labels": "loaded",
  "resnet_labels": "loaded"
}
```

---

## 11. Technical Notes & Troubleshooting

- **Keras Serialization Compatibility:** Models trained on recent Google Colab environments occasionally save a `quantization_config` key in layer configurations. Local Keras versions without this argument will raise a `TypeError` during deserialization. `demo/app.py` includes an automatic monkey-patch on `Dense.__init__` to strip this parameter before instantiation, ensuring reliable loading across different Keras versions.
- **Sequential Execution:** The backend runs inference on the CNN model first, followed by ResNet-18 within the same request lifecycle. For production scaling, inference could be dispatched concurrently or converted to an asynchronous worker queue.
- **Port Conflicts:** If port 5000 is occupied by another service, modify the port parameter in the last line of `demo/app.py`:
  ```python
  app.run(host='0.0.0.0', port=5001, debug=False)
  ```

---

## 12. Current Limitations

- **Isolated Character Recognition Only:** The current pipeline accepts individual character drawings. It does not perform document-level line or text segmentation.
- **Class Scope:** Evaluated on 100 (and 92) selected characters rather than the full 2,472+ classes of historical Kanji.
- **Fixed Input Resolution:** Resizing non-square drawings directly to $64 \times 64$ can distort aspect ratios if characters are drawn near canvas boundaries.

---

## 13. Future Improvements

- [ ] **Full Dataset Expansion:** Scale training across all 2,472 Kanji classes using hierarchical classification or metric learning (e.g., ArcFace / CosFace loss).
- [ ] **Vision Transformer (ViT) Exploration:** Evaluate compact ViT architectures (e.g., MobileViT) to assess global stroke attention mechanisms compared to local convolutional receptive fields.
- [ ] **Line-Level Sequence Recognition:** Combine YOLO / text detection with CRNN (Convolutional Recurrent Neural Network) and CTC Loss for continuous line recognition.
- [ ] **Model Quantization & ONNX Export:** Convert trained Keras/PyTorch weights to ONNX / TensorFlow Lite format for optimized client-side in-browser inference via ONNX Runtime Web.

---

## 14. Acknowledgements

- **Dataset:** Kuzushiji-Kanji dataset provided by the [Center for Open Data in the Humanities (CODH)](http://codh.rois.ac.jp/kmnist/), National Institute of Informatics (ROIS-DS).
- **Inspiration:** Research on historical document transcription and Japanese character recognition benchmarks.
