# Hệ Thống Nhận Dạng Chữ Kanji Viết Tay
Dự án thị giác máy tính và học sâu (end-to-end) nhận dạng chữ Hán (Kanji) viết tay tiếng Nhật, xây dựng các kiến trúc mạng nơ-ron huấn luyện từ đầu (from scratch) và ứng dụng web demo tương tác cục bộ.

[🇬🇧 English Version](README.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?logo=flask)](https://flask.palletsprojects.com/)

---

## 1. Tổng Quan

Bài toán nhận dạng chữ Kanji viết tay gặp nhiều thách thức lớn do số lượng ký tự khổng lồ, cấu trúc nét phức tạp và độ biến dạng chữ viết tay cao, đặc biệt là phong cách thảo thư/thư pháp cổ (Kuzushiji). Nhiều ký tự có cùng bộ thủ rất dễ bị nhầm lẫn khi người dùng viết nhanh hoặc lướt nét.

Repository này bao quát toàn bộ quy trình phát triển học máy:
- **Xử lý dữ liệu (Dataset Engineering):** Lọc lớp ký tự, phân chia phân tầng (stratified split), và pipeline chuyển đổi cực màu nét vẽ thích ứng giữa Canvas người dùng và tập huấn luyện.
- **Xây dựng mô hình từ đầu (Trained from Scratch):**
  - Mô hình **Custom CNN** gồm 4 khối tích chập, thiết kế để suy luận nhanh với bộ nhớ gọn nhẹ.
  - Mô hình **ResNet-18** tùy biến tự lập trình khối Residual (Skip Connection) nhằm triệt tiêu hiện tượng suy giảm gradient mà không sử dụng trọng số tiền huấn luyện (pre-trained weights).
  - Pipeline **PyTorch** áp dụng hàm mất mát Weighted Cross-Entropy Loss giải quyết tình trạng mất cân bằng dữ liệu (long-tailed distribution).
- **Triển khai ứng dụng cục bộ:** Backend Flask phục vụ suy luận tuần tự cả hai mô hình, trả kết quả về giao diện HTML5 Canvas thời gian thực để so sánh trực quan.

---

## 2. Tính Năng Nổi Bật

- **Huấn luyện từ đầu (From Scratch):** Cả hai mô hình chính (CNN và ResNet-18) đều được thiết kế và huấn luyện không dùng Transfer Learning, nhằm đánh giá chuẩn xác năng lực học đặc trưng hình học của từng kiến trúc.
- **Tiền xử lý đảo cực màu thích ứng (Adaptive Stroke Polarity):** Tự động phát hiện độ tương phản và đảo màu nét vẽ từ Canvas (nét đen trên nền trắng) về định dạng của tập huấn luyện (nét trắng trên nền đen).
- **Suy luận đối sánh hai mô hình (Dual-Model Inference):** Một lần vẽ của người dùng được suy luận tuần tự qua cả CNN và ResNet-18, hiển thị ký tự Top-1, độ tự tin (confidence score) và phân phối Top-3 ứng viên.
- **Xử lý mất cân bằng lớp:** Ứng dụng trọng số nghịch đảo tần suất lớp trong PyTorch giúp mô hình nhận dạng chuẩn xác cả những lớp có ít dữ liệu.
- **Giao diện Web tương tác:** Canvas vẽ HTML5 ($400 \times 400$) hỗ trợ chuột và màn hình cảm ứng, có lưới trợ nét chữ điền, tùy chỉnh cỡ nét cọ (2px–20px) và nút chuyển đổi giao diện Sáng / Tối.

---

## 3. Kiến Trúc Hệ Thống & Luồng Dữ Liệu

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           CLIENT (Giao diện Web)                          │
│  Người dùng vẽ trên Canvas (400x400) -> Xuất dữ liệu ảnh Base64 PNG       │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ HTTP POST /predict { image: base64 }
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           FLASK BACKEND (app.py)                          │
│                                                                           │
│  1. Giải mã chuỗi Base64 -> Ảnh xám PIL Grayscale ('L')                   │
│  2. Kiểm tra cực màu: if mean(pixel) > 127 => đảo màu (255 - pixel)       │
│  3. Thu phóng bằng thuật toán Lanczos về kích thước chuẩn 64x64           │
│  4. Chuẩn hóa giá trị điểm ảnh về khoảng thực [0.0, 1.0]                  │
│                                                                           │
│  Suy luận tuần tự hai mô hình:                                            │
│  ├── Reshape về (1, 64, 64, 1) -> Custom CNN -> Kết quả Top-1 & Top-3     │
│  └── Nhân bản lên (1, 64, 64, 3) -> ResNet-18  -> Kết quả Top-1 & Top-3   │
│                                                                           │
│  Trả về JSON: { cnn: { top1, confidence, top3 }, resnet: { ... } }        │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ Phản hồi JSON
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           TRỰC QUAN HÓA KẾT QUẢ                           │
│  Thẻ so sánh song song với thanh phần trăm độ tự tin & bảng xếp hạng Top-3│
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tập Dữ Liệu & Tiền Xử Lý

Thực nghiệm sử dụng tập dữ liệu nguồn mở chuẩn học thuật **Kuzushiji-Kanji** (ROIS-DS CODH):

- **Tập dữ liệu thô đầy đủ:** Chứa 2,472 lớp ký tự Unicode với mức độ mất cân bằng tần suất xuất hiện rất lớn.
- **Tập dữ liệu chuẩn 100 lớp (`100class/`):**
  - **Số lớp:** 100 lớp ký tự có tần suất xuất hiện cao và tiêu biểu.
  - **Tổng số mẫu:** 29,365 ảnh xám.
  - **Phân chia tập:** Chia ngẫu nhiên phân tầng (Stratified split) thành tập Huấn luyện (70% — 20,555 ảnh), tập Kiểm định (15% — 4,405 ảnh), và tập Kiểm thử độc lập (15% — 4,405 ảnh).
- **Tập dữ liệu PyTorch 93 lớp:**
  - Tập con lọc các lớp có từ 300 mẫu trở lên ($\ge 300$).
  - Tổng số 57,130 ảnh (Train: 39,990 / Val: 8,570 / Test: 8,570).

### Quy trình Tiền Xử Lý & Tăng Cường Dữ Liệu
1. **Thích ứng cực màu nét vẽ (Polarity Adaptation):**
   Dữ liệu huấn luyện lưu trữ nét chữ màu trắng trên nền đen ($\text{mean} < 127$). Ngược lại, nét vẽ trên Canvas web là nét đen trên nền trắng ($\text{mean} > 127$). Backend tự động xử lý:
   $$\text{img} = 255 - \text{img} \quad \text{nếu} \quad \frac{1}{HW}\sum_{i,j} \text{img}_{i,j} > 127$$
2. **Thu phóng (Resizing):** Sử dụng thuật toán nội suy Lanczos về $64 \times 64$ pixels.
3. **Chuẩn hóa (Normalization):** Đưa khoảng giá trị pixel về $[0.0, 1.0]$.
4. **Tăng cường dữ liệu (Chỉ áp dụng khi Train):**
   - Xoay ngẫu nhiên: $\pm 10^\circ$
   - Dịch chuyển ngẫu nhiên: $\pm 10\%$ theo chiều ngang và dọc
   - Thu phóng ngẫu nhiên (Zoom): $\pm 10\%$

---

## 5. Kiến Trúc Các Mô Hình

Ba pipeline mô hình được lưu trữ trong thư mục `training/`:

### 1. Custom CNN (`training/CNN.ipynb`)
- **Framework:** TensorFlow / Keras
- **Kích thước đầu vào:** $(64, 64, 1)$ — Ảnh xám đơn kênh
- **Cấu trúc:**
  - 4 khối tích chập lặp lại, mỗi khối gồm $2 \times \text{Conv2D}(3 \times 3)$, Batch Normalization, hàm kích hoạt ReLU, Max Pooling $2 \times 2$, và Spatial Dropout 25%.
  - Tiến trình số lượng kênh: $32 \rightarrow 64 \rightarrow 128 \rightarrow 256$.
  - Khối phân loại: `Flatten` $\rightarrow$ `Dense(512)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Dense(100, softmax)`.
- **Tổng số tham số:** 3,326,532 (~3.3 triệu)

### 2. Custom ResNet-18 (`training/RESNET18.ipynb`)
- **Framework:** TensorFlow / Keras (tự xây dựng từ đầu)
- **Kích thước đầu vào:** $(64, 64, 3)$ — Ảnh xám nhân bản thành 3 kênh
- **Cấu trúc:**
  - Lớp tích chập đầu vào kèm Batch Normalization, ReLU và Max Pooling.
  - 4 stage phần dư với số bộ lọc $[64, 128, 256, 512]$, mỗi stage gồm 2 khối Residual Block chứa đường tắt đồng nhất (Identity Shortcut) hoặc đường tắt chiếu tích chập $1 \times 1$ (Projection Shortcut) khi giảm kích thước không gian.
  - Lớp gộp trung bình toàn cục (Global Average Pooling) nối vào lớp Dense 100 ngõ ra Softmax.
- **Tổng số tham số:** 11,229,732 (~11.2 triệu)

### 3. ResNet-18 với Weighted Loss (`training/training_notebook.ipynb`)
- **Framework:** PyTorch
- **Kích thước đầu vào:** $(3, 64, 64)$
- **Số lớp:** 93 lớp ($\ge 300$ ảnh/lớp)
- **Chiến lược Loss:** Weighted Cross-Entropy Loss nhằm bù đắp cho sự chênh lệch số lượng mẫu giữa các lớp:
  $$w_c = \frac{N}{K \cdot N_c}$$
  trong đó $N$ là tổng số ảnh, $K$ là số lớp, và $N_c$ là số ảnh của lớp $c$.

---

## 6. Kết Quả Thực Nghiệm & Đánh Giá

Các chỉ số dưới đây được trích xuất trực tiếp từ kết quả chạy thực tế trên tập kiểm thử độc lập (Test Set):

| Chỉ số đánh giá | Custom CNN (Keras) | Custom ResNet-18 (Keras) | ResNet-18 (PyTorch) |
| :--- | :---: | :---: | :---: |
| **Số lớp mục tiêu** | 100 lớp | 100 lớp | 93 lớp |
| **Kích thước tập Test** | 4,405 mẫu | 4,405 mẫu | 8,570 mẫu |
| **Tổng số tham số** | 3,326,532 | 11,229,732 | ~11.2M |
| **Số Epoch huấn luyện** | 31 (Dừng sớm tại epoch 24) | 35 (Dừng sớm tại epoch 28) | 15 epochs (CPU) |
| **Độ chính xác (Test Acc)** | **98.52%** | **98.73%** | **98.27%** |
| **Mất mát (Test Loss)** | 0.0597 | **0.0556** | 0.0672 |
| **Macro Average F1** | 0.9829 | **0.9852** | 0.9807 |
| **Top-5 Accuracy** | — | — | **99.68%** |
| **Kích thước file mô hình** | 38.2 MB | 128.7 MB | ~45 MB (PyTorch state_dict) |

### Nhận xét kỹ thuật
- ResNet-18 đạt độ chính xác cao hơn (98.73% so với 98.52%) và độ mất mát thấp hơn, cho thấy khả năng trích xuất các đặc trưng phân cấp sâu vượt trội đối với các chữ Hán có nhiều nét mảnh và phức tạp.
- Custom CNN có ưu thế lớn về kích thước tham số (chỉ chiếm 29.6% so với ResNet-18) và dung lượng file nhẹ hơn 3.3 lần (38.2 MB so với 128.7 MB), tối ưu cho các thiết bị hạn chế tài nguyên phần cứng.
- Trên tập 93 lớp với PyTorch, việc áp dụng trọng số lớp đã duy trì chỉ số Macro F1 đạt 98.07%, chứng minh tính ổn định cao trên cả các lớp thiểu số.

---

## 7. Phân Tích Lỗi (Error Analysis)

Phân tích ma trận nhầm lẫn (Confusion Matrix) trong notebook cho thấy các dạng sai sót chính:
1. **Trùng bộ thủ kèm tinh giản nét:**
   - Ví dụ: Chữ **今** (Kim - Nay) bị nhầm với chữ **分** (Phân - Chia). Cả hai chữ đều chung bộ "Nhân nón" ($\boldsymbol{\hat{人}}$) phía trên. Khi viết tay nhanh, nét phẩy và móc dưới của chữ 今 có xu hướng co lại thành hai vệt ngắn giống phần dưới của chữ 分.
2. **Ký tự chữ thảo bị lược nét mạnh:**
   - Ví dụ: Chữ **國** (Quốc). Trong lối viết thảo cổ Kuzushiji, cấu trúc khung viền phức tạp bên trong thường được vẽ lướt thành một nét uốn lượn đơn, dẫn đến khả năng bị nhầm với các chữ ít nét hơn.
3. **Cụm ký tự cùng bộ thủ:**
   - Các chữ có chung bộ Thủy `氵` (như 海, 汁, 江) đôi khi bị tráo đổi vị trí giữa Top-1 và Top-2, nhưng luôn nằm chắc chắn trong **Top-3** dự đoán có độ tự tin cao nhất.

---

## 8. Cấu Trúc Thư Mục Repository

```
├── demo/
│   ├── app.py                     # Flask server xử lý tiền xử lý & suy luận tuần tự
│   ├── requirements.txt           # Danh sách thư viện Python cần thiết cho demo
│   ├── frontend/
│   │   ├── index.html             # Giao diện web chính với các thẻ hiển thị đối sánh
│   │   ├── script.js              # Logic thao tác Canvas, mã hóa Base64 & gọi API
│   │   └── style.css              # Giao diện hiện đại, hỗ trợ chế độ Dark/Light mode
│   ├── label/
│   │   ├── labelscnn.json         # Danh mục 100 nhãn cho Custom CNN (1 kênh)
│   │   └── labelsresnet18.json    # Danh mục 100 nhãn cho ResNet-18 (3 kênh)
│   └── model/
│       └── .gitkeep               # Giữ thư mục trên Git (weights được bỏ qua bởi .gitignore)
│
├── training/
│   ├── CNN.ipynb                  # Notebook huấn luyện & đánh giá Custom CNN 100 lớp
│   ├── RESNET18.ipynb             # Notebook huấn luyện & đánh giá ResNet-18 100 lớp
│   └── training_notebook.ipynb    # Notebook PyTorch huấn luyện 93 lớp kèm Weighted Loss & Gradio
│
├── .gitignore                     # Cấu hình bỏ qua dataset, file binary nặng và file tạm
├── README.md                      # Tài liệu dự án bằng Tiếng Anh
└── README_VN.md                   # Tài liệu dự án bằng Tiếng Việt
```

---

## 9. Hướng Dẫn Cài Đặt & Chạy Cục Bộ

### Yêu cầu môi trường
- Python 3.10, 3.11, hoặc 3.12
- Trình quản lý gói `pip`

### 1. Sao chép (Clone) kho lưu trữ
```bash
git clone https://github.com/<tai-khoan-cua-ban>/kanji-handwriting-recognition.git
cd kanji-handwriting-recognition
```

### 2. Thiết lập môi trường ảo
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt trên Windows:
.venv\Scripts\activate

# Kích hoạt trên Linux/macOS:
source .venv/bin/activate
```

### 3. Cài đặt các thư viện cần thiết
```bash
pip install -r demo/requirements.txt
```

*Nội dung file `demo/requirements.txt`:*
```
flask>=3.0.0
flask-cors>=4.0.0
tensorflow>=2.15.0
numpy>=1.24.0
Pillow>=10.0.0
```

### 4. Chuẩn bị file trọng số mô hình
Do file `model_resnet18.h5` nặng ~128.7 MB vượt giới hạn 100 MB của GitHub, các file trọng số không được commit trực tiếp vào git.

Hãy đặt file trọng số đã huấn luyện vào thư mục `demo/model/`:
- `demo/model/model_cnn.h5`
- `demo/model/model_resnet18.h5`

*(Bạn có thể tự xuất các file này bằng cách chạy các cell lưu model trong notebook `training/CNN.ipynb` và `training/RESNET18.ipynb`, hoặc tải về từ mục **Releases** của repo nếu có).*

### 5. Khởi chạy máy chủ Web
```bash
cd demo
python app.py
```

Mở trình duyệt web và truy cập địa chỉ:
```
http://localhost:5000
```

### 6. Kiểm tra trạng thái máy chủ
Bạn có thể kiểm tra API bằng lệnh:
```bash
curl http://localhost:5000/status
```
Kết quả phản hồi mẫu:
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

## 10. Ghi Chú Kỹ Thuật & Xử Lý Sự Cố

- **Xử lý tương thích Keras Colab:** Các model lưu từ môi trường Google Colab mới thường chứa khóa `quantization_config` trong cấu hình layer. Khi nạp lại trên các bản Keras local cũ hơn, hàm `load_model()` sẽ báo lỗi `TypeError`. File `demo/app.py` đã tích hợp sẵn đoạn monkey-patch loại bỏ thuộc tính này trước khi khởi tạo `Dense`, giúp nạp model trơn tru trên mọi phiên bản.
- **Thực thi suy luận tuần tự:** Server thực hiện suy luận trên mô hình CNN trước, sau đó tiếp tục suy luận trên ResNet-18 trong cùng một chu kỳ request. Khi triển khai production chịu tải lớn, có thể chuyển sang cơ chế suy luận bất đồng bộ hoặc hàng đợi worker.
- **Trùng cổng mạng (Port Conflict):** Nếu cổng 5000 đã bị chiếm dụng bởi ứng dụng khác, bạn có thể chỉnh sửa dòng cuối cùng của `demo/app.py`:
  ```python
  app.run(host='0.0.0.0', port=5001, debug=False)
  ```

---

## 11. Các Giới Hạn Hiện Tại

- **Chỉ nhận dạng từng chữ đơn lẻ:** Hệ thống hiện chỉ nhận ảnh một chữ Kanji vẽ đơn lẻ, chưa tích hợp thuật toán phân đoạn dòng chữ để đọc cả văn bản dài.
- **Quy mô lớp huấn luyện:** Mô hình hiện được đánh giá trên 100 lớp (và 93 lớp) đại diện, chưa mở rộng ra toàn bộ 2,472+ lớp chữ Hán cổ.
- **Kích thước ảnh cố định:** Thao tác co giãn trực tiếp về $64 \times 64$ có thể làm méo tỷ lệ nét nếu người dùng vẽ chữ lệch sát mép khung Canvas.

---

## 12. Định Hướng Nâng Cấp Tiếp Theo

- [ ] **Mở rộng toàn bộ 2,472 lớp:** Áp dụng phương pháp phân loại phân cấp (Hierarchical Classification) hoặc Metric Learning (ArcFace / CosFace) để học các lớp hiếm mẫu.
- [ ] **Thử nghiệm Vision Transformer (ViT):** Đánh giá các biến thể gọn nhẹ (như MobileViT) để khai thác cơ chế Self-Attention nắm bắt mối tương quan nét vẽ toàn cục.
- [ ] **Nhận dạng theo dòng văn bản:** Kết hợp mô hình phát hiện chữ (như YOLO) cùng mạng **CRNN + CTC Loss** để đọc văn bản viết tay liên tục trên tài liệu cổ.
- [ ] **Tối ưu hóa mô hình & Xuất ONNX:** Chuyển đổi trọng số sang định dạng ONNX / TensorFlow Lite để thực hiện suy luận trực tiếp trên trình duyệt bằng ONNX Runtime Web.

---

## 13. Lời Cảm Ơn & Nguồn Tham Khảo

- **Tập dữ liệu:** Bộ dữ liệu Kuzushiji-Kanji được cung cấp bởi [Center for Open Data in the Humanities (CODH)](http://codh.rois.ac.jp/kmnist/), National Institute of Informatics (ROIS-DS).
- **Cảm hứng đề tài:** Các nghiên cứu về số hóa di sản tài liệu cổ và các chuẩn đánh giá nhận dạng chữ viết tay tiếng Nhật.

