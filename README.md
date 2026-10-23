# Handwritten Kanji Recognition System
An end-to-end computer vision and deep learning project for recognizing handwritten Japanese Kanji characters, featuring custom architectures trained from scratch and an interactive local web demo.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?logo=flask)](https://flask.palletsprojects.com/)

---

## 1. Overview

Handwritten Kanji recognition is challenging due to the large character inventory, intricate stroke arrangements, and significant handwriting distortion, especially in calligraphic or cursive (Kuzushiji) styles. Characters sharing identical radicals can be difficult to separate under loose handwriting.

This repository covers the complete machine learning lifecycle:
- **Dataset engineering:** Class filtering, stratified splitting, and an adaptive polarity conversion pipeline to reconcile drawing-canvas strokes with training data.
- **Model development from scratch:** 
  - A 4-stage **Custom CNN** designed for fast inference with small memory footprint.
  - A custom **ResNet-18** implementation built with residual blocks to mitigate vanishing gradients without using pre-trained weights.
  - A **PyTorch pipeline** applying class-weighted cross-entropy loss to address long-tailed class distribution.
- **Local deployment:** A Flask backend serving predictions sequentially to a responsive HTML5 Canvas frontend for real-time comparison between models.

---

## 2. Key Features

- **Trained From Scratch:** Both primary models (CNN and ResNet-18) were designed and trained without transfer learning to evaluate pure feature learning capacity on historical character forms.
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
- **93-Class PyTorch Dataset:**
  - Filtered subset of classes containing $\ge 300$ samples each.
  - Total of 57,130 images (Train: 39,990 / Val: 8,570 / Test: 8,570).

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

Three distinct model pipelines are included in the repository:

### 1. Custom CNN (`training/CNN.ipynb`)
- **Framework:** TensorFlow / Keras
- **Input Shape:** $(64, 64, 1)$ — Single-channel grayscale
- **Structure:**
  - 4 convolutional blocks, each containing $2 \times \text{Conv2D}(3 \times 3)$ layers, Batch Normalization, ReLU activation, $2 \times 2$ Max Pooling, and 25% Spatial Dropout.
  - Channel progression: $32 \rightarrow 64 \rightarrow 128 \rightarrow 256$.
  - Classification head: `Flatten` $\rightarrow$ `Dense(512)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Dense(100, softmax)`.
- **Total Parameters:** 3,326,532 (~3.3M)

### 2. Custom ResNet-18 (`training/RESNET18.ipynb`)
- **Framework:** TensorFlow / Keras (implemented from scratch)
- **Input Shape:** $(64, 64, 3)$ — Grayscale repeated across 3 channels
- **Structure:**
  - Initial convolutional layer ($3 \times 3$ or $7 \times 7$) followed by Batch Normalization, ReLU, and Max Pooling.
  - 4 residual stages with filter counts $[64, 128, 256, 512]$, each holding 2 residual blocks with identity shortcuts and projection shortcuts ($1 \times 1$ convolutions) where spatial dimensions downsample.
  - Global Average Pooling layer connected to a 100-class Softmax layer.
- **Total Parameters:** 11,229,732 (~11.2M)

### 3. ResNet-18 with Class-Weighted Loss (`training/training_notebook.ipynb`)
- **Framework:** PyTorch
- **Input Shape:** $(3, 64, 64)$
- **Classes:** 93 classes ($\ge 300$ samples/class)
- **Loss Strategy:** Weighted Cross-Entropy Loss to counter uneven sample distribution:
  $$w_c = \frac{N}{K \cdot N_c}$$
  where $N$ is total images, $K$ is number of classes, and $N_c$ is image count for class $c$.

---

## 6. Evaluation & Benchmark Results

The following metrics are verified directly from completed notebook runs on the unseen test sets:

| Metric | Custom CNN (Keras) | Custom ResNet-18 (Keras) | ResNet-18 (PyTorch) |
| :--- | :---: | :---: | :---: |
| **Target Classes** | 100 classes | 100 classes | 93 classes |
| **Test Set Size** | 4,405 samples | 4,405 samples | 8,570 samples |
| **Total Parameters** | 3,326,532 | 11,229,732 | ~11.2M |
| **Training Epochs** | 31 (Early Stopped at best ep 24) | 35 (Early Stopped at best ep 28) | 15 epochs (CPU) |
| **Test Accuracy** | **98.52%** | **98.73%** | **98.27%** |
| **Test Loss** | 0.0597 | **0.0556** | 0.0672 |
| **Macro Average F1** | 0.9829 | **0.9852** | 0.9807 |
| **Top-5 Accuracy** | — | — | **99.68%** |
| **Model File Size** | 38.2 MB | 128.7 MB | ~45 MB (PyTorch state_dict) |

### Key Observations
- ResNet-18 achieves higher classification accuracy (98.73% vs 98.52%) and lower test loss, demonstrating stronger generalization on intricate characters with dense strokes.
- The Custom CNN model requires only 29.6% of ResNet-18's parameter volume (3.3M vs 11.2M) and one-third of the storage space (38.2 MB vs 128.7 MB), making it more practical for low-resource environments.
- On the PyTorch 93-class dataset, incorporating class weights maintained a high Macro F1 score of 98.07%, confirming robust performance across minor and major classes.

---

## 7. Error Analysis

Examination of confusion matrices in the training notebooks reveals distinct failure patterns:
1. **Shared Radicals with Stroke Simplification:**
   - Example: **今** (Now) confused with **分** (Divide). Both characters share the top "person" radical ($\boldsymbol{\hat{人}}$). In hurried handwriting, the bottom horizontal and vertical hook of 今 can compress into two detached ticks resembling the lower part of 分.
2. **Heavy Cursive Contraction:**
   - Example: **國** (Country). In classical cursive Kuzushiji, the complex multi-stroke internal structure is frequently abstracted into a simplified wavy stroke, causing occasional misclassification against simpler glyphs.
3. **Class Radical Clusters:**
   - Characters sharing the water radical `氵` (e.g., 海, 汁, 江) occasionally swapped positions between Top-1 and Top-2, but consistently remained within the Top-3 candidates.

---

## 8. Repository Structure

```
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
│   └── training_notebook.ipynb    # 93-class PyTorch training with Weighted Loss & Gradio
│
├── .gitignore                     # Git exclusion rules for datasets, binaries, and caches
└── README.md                      # Technical project documentation
```

---

## 9. Installation & Quickstart

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

## 10. Technical Notes & Troubleshooting

- **Keras Serialization Compatibility:** Models trained on recent Google Colab environments occasionally save a `quantization_config` key in layer configurations. Local Keras versions without this argument will raise a `TypeError` during deserialization. `demo/app.py` includes an automatic monkey-patch on `Dense.__init__` to strip this parameter before instantiation, ensuring reliable loading across different Keras versions.
- **Sequential Execution:** The backend runs inference on the CNN model first, followed by ResNet-18 within the same request lifecycle. For production scaling, inference could be dispatched concurrently or converted to an asynchronous worker queue.
- **Port Conflicts:** If port 5000 is occupied by another service, modify the port parameter in the last line of `demo/app.py`:
  ```python
  app.run(host='0.0.0.0', port=5001, debug=False)
  ```

---

## 11. Current Limitations

- **Isolated Character Recognition Only:** The current pipeline accepts individual character drawings. It does not perform document-level line or text segmentation.
- **Class Scope:** Evaluated on 100 (and 93) selected characters rather than the full 2,472+ classes of historical Kanji.
- **Fixed Input Resolution:** Resizing non-square drawings directly to $64 \times 64$ can distort aspect ratios if characters are drawn near canvas boundaries.

---

## 12. Future Improvements

- [ ] **Full Dataset Expansion:** Scale training across all 2,472 Kanji classes using hierarchical classification or metric learning (e.g., ArcFace / CosFace loss).
- [ ] **Vision Transformer (ViT) Exploration:** Evaluate compact ViT architectures (e.g., MobileViT) to assess global stroke attention mechanisms compared to local convolutional receptive fields.
- [ ] **Line-Level Sequence Recognition:** Combine YOLO / text detection with CRNN (Convolutional Recurrent Neural Network) and CTC Loss for continuous line recognition.
- [ ] **Model Quantization & ONNX Export:** Convert trained Keras/PyTorch weights to ONNX / TensorFlow Lite format for optimized client-side in-browser inference via ONNX Runtime Web.

---

## 13. Acknowledgements

- **Dataset:** Kuzushiji-Kanji dataset provided by the [Center for Open Data in the Humanities (CODH)](http://codh.rois.ac.jp/kmnist/), National Institute of Informatics (ROIS-DS).
- **Inspiration:** Research on historical document transcription and Japanese character recognition benchmarks.
