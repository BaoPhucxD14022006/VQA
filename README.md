# 👁️‍🗨️ Visual Question Answering (VQA) - Nghiên cứu Khoa học (NCKH)

Dự án nghiên cứu và thử nghiệm các mô hình học sâu cho bài toán **Visual Question Answering (VQA)** — bài toán đa phương thức (Multimodal Learning) kết hợp giữa **Thị giác Máy tính (Computer Vision)** và **Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)** nhằm tự động trả lời các câu hỏi dựa trên nội dung hình ảnh đầu vào.

---

## 📑 Mục lục
1. [Tổng quan Đề tài](#-tổng-quan-đề-tài)
2. [Kiến trúc Mô hình](#-kiến-trúc-mô-hình)
   - [1. Mô hình Baseline: ResNet-18 + LSTM](#1-mô-hình-baseline-resnet-18--lstm)
   - [2. Mô hình Nâng cao: ViT-B/16 + Transformer + Cross-Attention](#2-mô-hình-nâng-cao-vit-b16--transformer--cross-attention)
3. [Cấu trúc Thư mục Dự án](#-cấu-trúc-thư-mục-dự-án)
4. [Cài đặt Môi trường](#-cài-đặt-môi-trường)
5. [Chuẩn bị Dữ liệu](#-chuẩn-bị-dữ-liệu)
6. [Cấu hình & Hướng dẫn Sử dụng](#-cấu-hình--hướng-dẫn-sử-dụng)
   - [Chuyển đổi giữa Kaggle và Local](#chuyển-đổi-giữa-kaggle-và-local)
   - [Huấn luyện (Training)](#huấn-luyện-training)
   - [Trực quan hóa Lịch sử Huấn luyện (Plot History)](#trực-quan-hóa-lịch-sử-huấn-luyện-plot-history)
   - [Dự đoán & Tạo Submission (Inference)](#dự-đoán--tạo-submission-inference)
7. [Kỹ thuật Tối ưu Huấn luyện](#-kỹ-thuật-tối-ưu-huấn-luyện)
8. [Định hướng Phát triển & Mở rộng (NCKH)](#-định-hướng-phát-triển--mở-rộng-nckh)

---

## 📌 Tổng quan Đề tài

Bài toán VQA nhận đầu vào gồm:
- **Hình ảnh** $I \in \mathbb{R}^{3 \times H \times W}$
- **Câu hỏi văn bản** $Q = (w_1, w_2, \dots, w_L)$

Mục tiêu là dự đoán câu trả lời $A$ chính xác nhất từ không gian nhãn câu trả lời (Answer Vocabulary):
$$\hat{A} = \arg\max_{a \in \mathcal{A}} P(a \mid I, Q; \theta)$$

Notebook chính triển khai toàn bộ quy trình: [vqa-reseach.ipynb](file:///d:/VQA%20-%20NCKH/vqa-reseach.ipynb).

---

## 🧱 Kiến trúc Mô hình

### 1. Mô hình Baseline: ResNet-18 + LSTM
- **Nhánh hình ảnh**: ResNet-18 (pre-trained trên ImageNet-1K), loại bỏ 2 tầng cuối (AdaptiveAvgPool và FC), áp dụng Global Average Pooling trên spatial dimensions để thu về feature vector $v \in \mathbb{R}^{512}$.
- **Nhánh câu hỏi**: Embedding layer ($128$ chiều) đưa chuỗi token vào mạng LSTM ($hidden\_size=256$). Trạng thái ẩn cuối cùng của LSTM đại diện cho câu hỏi $q \in \mathbb{R}^{256}$.
- **Hợp nhất đa phương thức (Multimodal Fusion)**: Ghép nối (Concatenation) hai vector đặc trưng:
  $$x = [v \,\|\, q] \in \mathbb{R}^{768}$$
- **Phân loại**: Mạng MLP gồm `Linear(768, 512) -> ReLU -> Linear(512, num_answers)`.

```
[Hình ảnh] --> ResNet-18 ------------> Vector (512) ----\
                                                         [Concat] --> MLP (512) --> Logits (num_answers)
[Câu hỏi]  --> Embedding --> LSTM ---> Vector (256) ----/
```

### 2. Mô hình Nâng cao: ViT-B/16 + Transformer + Cross-Attention
- **Nhánh hình ảnh (Vision Transformer)**:
  - Sử dụng backbone `vit_b_16` pre-trained để chia ảnh thành các patch $16 \times 16$, tạo chuỗi token ảnh kèm `[CLS]` token và vị trí.
  - Trích xuất representation chuỗi token $i \in \mathbb{R}^{B \times 197 \times 768}$.
  - Đóng băng (freeze) các tầng đầu của ViT và chỉ **fine-tune 4 block encoder cuối** để tối ưu tài nguyên tính toán.
  - Sử dụng `Linear(768, 256)` để chiếu đặc trưng ảnh về cùng không gian nhúng với câu hỏi.
- **Nhánh câu hỏi (Transformer Encoder)**:
  - Word Embedding ($256$ chiều) kết hợp thêm học vector `[CLS]` token và Position Embedding.
  - Chuỗi câu hỏi đi qua **4 tầng Transformer Encoder** (8 heads, $d_{model}=256$) có áp dụng padding mask.
- **Cơ chế Tương tác Cross-Attention**:
  - Dùng câu hỏi làm **Query** ($Q = q$), dùng đặc trưng ảnh làm **Key & Value** ($K = i, V = i$).
  - Multi-Head Cross-Attention (8 heads, $256$ chiều) giúp câu hỏi "chú ý" (attend) vào các vùng ảnh có thông tin liên quan trực tiếp đến câu hỏi.
  - Áp dụng kết nối tắt (Residual Connection), Layer Normalization và khối Feed-Forward Network (GELU activation, Dropout 0.1).
- **Phân loại**: Trích xuất token `[CLS]` vị trí đầu tiên sau khối Cross-Attention đi qua bộ phân loại `Linear(256, 512) -> ReLU -> Linear(512, num_answers)`.

```
[Hình ảnh]  --> ViT-B/16 (freeze early + unfreeze last 4 layers) --> Linear(768->256) = K, V --\
                                                                                                 \
                                                                                                  --> [Cross-Attention] --> FFN --> [CLS] --> Classifier
                                                                                                 /
[Câu hỏi]   --> Embedding + PosEmbed + [CLS] --> 4x Transformer Encoder = Q --------------------/
```

---

## 📂 Cấu trúc Thư mục Dự án

```
d:/VQA - NCKH/
│
├── vqa-reseach.ipynb          # Jupyter Notebook chính chứa pipeline hoàn chỉnh
├── requirements.txt           # Danh sách các thư viện phụ thuộc
├── README.md                  # Tài liệu hướng dẫn chi tiết dự án
│
└── data/                      # Thư mục dữ liệu (nếu chạy trên máy cục bộ)
    ├── questions/
    │   ├── train.json       # Tập câu hỏi huấn luyện
    │   ├── val.json           # Tập câu hỏi kiểm định
    │   └── test.json          # Tập câu hỏi kiểm thử (chỉ chứa question & qid)
    ├── images/
    │   ├── train/             # Hình ảnh huấn luyện & kiểm định
    │   └── test/              # Hình ảnh kiểm thử
    └── vocab/
        ├── vocab.pth          # Từ điển câu hỏi (word -> token_id)
        └── ans_vocab.pth      # Từ điển câu trả lời (answer -> class_id)
```

---

## ⚙️ Cài đặt Môi trường

### 1. Yêu cầu Hệ thống
- **Python**: `>= 3.10` (Khuyến nghị `Python 3.11`)
- **GPU**: NVIDIA GPU với tối thiểu 8GB VRAM (để huấn luyện ViT-B/16 mượt mà) hoặc chạy trên Google Colab / Kaggle GPU T4 / P100.

### 2. Cài đặt chi tiết
Tạo và kích hoạt môi trường ảo:

```powershell
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường trên Windows PowerShell
.venv\Scripts\Activate.ps1
```

Cài đặt PyTorch phù hợp với phiên bản GPU CUDA:
```powershell
# Trường hợp máy có GPU NVIDIA CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Hoặc máy có CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Nếu chỉ chạy trên CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Cài đặt các gói phụ thuộc còn lại từ [requirements.txt](file:///d:/VQA%20-%20NCKH/requirements.txt):
```powershell
pip install -r requirements.txt
```

---

## 📊 Chuẩn bị Dữ liệu

### Định dạng File Câu hỏi (JSON)

- **Tập Train / Validation (`train_1.json`, `val.json`)**:
  ```json
  {
    "root": {
      "questions": [
        {
          "qid": 1001,
          "image_filename": "image_0001.jpg",
          "question": "Con mèo đang nằm ở đâu?",
          "answer": "tren ghe"
        }
      ]
    }
  }
  ```

- **Tập Test (`test.json`)**:
  ```json
  {
    "questions": [
      {
        "qid": 5001,
        "image_filename": "test_0001.jpg",
        "question": "Bầu trời có màu gì?"
      }
    ]
  }
  ```

---

## 🚀 Cấu hình & Hướng dẫn Sử dụng

Mở notebook [vqa-reseach.ipynb](file:///d:/VQA%20-%20NCKH/vqa-reseach.ipynb) bằng Jupyter Notebook hoặc VS Code.

### Chuyển đổi giữa Kaggle và Local
Trong cell **Config** của notebook, cấu hình lại các biến trong lớp `Path` tùy theo môi trường:

```python
class Path:
    # --- Khi chạy trên Kaggle ---
    # BASE_DIR = "/kaggle/input/datasets/lbophc/nckh-vqa"
    # SAVE_PATH = "/kaggle/working/best_model.pth"

    # --- Khi chạy trên máy Local (Windows) ---
    BASE_DIR = "d:/VQA - NCKH/data"
    TRAIN_JSON = os.path.join(BASE_DIR, "questions/train.json")
    VAL_JSON   = os.path.join(BASE_DIR, "questions/val.json")
    TEST_JSON  = os.path.join(BASE_DIR, "questions/test.json")

    TRAIN_IMG_DIR = os.path.join(BASE_DIR, "images/train")
    TEST_IMG_DIR  = os.path.join(BASE_DIR, "images/test")

    SAVE_PATH = "best_model.pth"
    HISTORY_JSON_PATH = "training_history.json"
    HISTORY_PLOT_PATH = "training_history.png"
    SUBMISSION_PATH = "submission.csv"
```

### Huấn luyện (Training)
Tại cell cuối của notebook, bỏ comment các dòng liên quan đến `trainer.train()`:

```python
if __name__ == "__main__":
    dataloader = Data_Loader()
    train_loader = dataloader.load_data(Path.TRAIN_JSON, Path.TRAIN_IMG_DIR, train=True)
    val_loader   = dataloader.load_data(Path.VAL_JSON, Path.TRAIN_IMG_DIR, train=False, val=True)

    # Lựa chọn mô hình:
    # 1. Baseline:
    # model = Model_ResNet_LSTM(len(Config.VOCAB), len(Config.ANS_VOCAB))
    # 2. Advanced ViT + Cross Attention:
    model = Model_ViT_Cross_attn(len(Config.VOCAB), len(Config.ANS_VOCAB))

    trainer = Tranning(model, train_loader, val_loader)
    trainer.train()
    trainer.plot_history()
```

### Trực quan hóa Lịch sử Huấn luyện (Plot History)
Sau khi quá trình huấn luyện hoàn tất, hàm `trainer.plot_history()` sẽ tạo và lưu file ảnh `training_history.png` gồm 3 đồ thị:
1. **Loss (Train Loss vs. Val Loss)**
2. **Accuracy (Train Acc vs. Val Acc)**
3. **F1-Score (Weighted Train F1 vs. Weighted Val F1)**

### Dự đoán & Tạo Submission (Inference)
Chạy hàm `inference(model)` để:
1. Nạp trọng số tốt nhất từ file checkpoint `best_model.pth`.
2. Dự đoán câu trả lời cho toàn bộ mẫu trong tập `test.json`.
3. Ghi kết quả ra file `submission.csv` với cấu trúc `qid,answer`.
4. Nén tự động thành `submission.zip` sẵn sàng nộp bài / đánh giá.

---

## ⚡ Kỹ thuật Tối ưu Huấn luyện

Dự án áp dụng nhiều kỹ thuật tối ưu nhằm tăng tốc độ huấn luyện và cải thiện độ ổn định:
1. **Mixed Precision Training (AMP)**: Sử dụng `torch.amp.autocast('cuda')` và `torch.amp.GradScaler('cuda')` giúp giảm tiêu hao VRAM khoảng 40-50% và tăng tốc 1.5 - 2x trên các GPU có Tensor Cores (T4, RTX series, A100).
2. **Tối ưu Pipeline Dữ liệu**:
   - `pin_memory=True`: Chuyển dữ liệu từ CPU RAM sang GPU VRAM nhanh hơn thông qua non-blocking transfers.
   - `num_workers=4` và `prefetch_factor=2`: Nạp trước các batch dữ liệu trong khi GPU đang tính toán.
3. **Đánh giá Đa chỉ số**: Không chỉ theo dõi Accuracy mà còn đánh giá **Weighted F1-Score** qua thư viện `scikit-learn`, đặc biệt quan trọng với bài toán VQA khi phân phối tần suất các câu trả lời thường bị mất cân bằng (imbalanced class distribution).
4. **Reproducibility**: Cài đặt seed cố định (`SEED = 42`) cho `random`, `numpy`, `torch` và cuDNN để kết quả thực nghiệm có thể tái lập được.

---

## 🔬 Định hướng Phát triển & Mở rộng (NCKH)

Để nâng cao chất lượng đề tài nghiên cứu khoa học hoặc cải thiện thứ hạng cuộc thi, một số hướng mở rộng tiềm năng bao gồm:

1. **Pretrained Language Models**:
   - Thay thế LSTM / Transformer cơ bản bằng các mô hình ngôn ngữ tiếng Việt tiền huấn luyện mạnh như **PhoBERT**, **BARTpho** hoặc mô hình đa ngôn ngữ **XLM-RoBERTa**.
2. **Vision-Language Foundation Models**:
   - Tận dụng các mô hình nền tảng đa phương thức tiên tiến như **CLIP**, **BLIP-2**, hoặc **ViLT** để trích xuất biểu diễn liên kết trực tiếp giữa ảnh và văn bản.
3. **Cơ chế Co-Attention 2 chiều**:
   - Mở rộng cơ chế Cross-Attention thành **Bi-directional Co-Attention** (câu hỏi dẫn hướng tập trung vào vùng ảnh, đồng thời các vùng ảnh quan trọng dẫn hướng việc hiểu câu hỏi).
4. **Data Augmentation**:
   - Hình ảnh: Random Crop, Color Jitter, AutoAugment.
   - Câu hỏi: Back-translation, hoán đổi từ đồng nghĩa tiếng Việt để làm phong phú dữ liệu huấn luyện.
5. **Ensemble Models**:
   - Kết hợp đầu ra xác suất của nhiều kiến trúc mô hình (ResNet + LSTM, ViT + Cross-Attention, PhoBERT + Swin Transformer) để gia tăng độ chính xác tổng thể.
