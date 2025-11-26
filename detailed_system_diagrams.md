# Sơ Đồ Chi Tiết Hệ Thống Phân Loại Hoa

## 1. Kiến Trúc Tổng Thể Hệ Thống

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HỆ THỐNG PHÂN LOẠI HOA                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │   ĐẦU VÀO    │──▶│ TIỀN XỬ LÝ  │──▶│   MÔ HÌNH   │──▶│   ĐẦU RA    │        │
│  │             │   │             │   │             │   │             │        │
│  │ • Ảnh Thô   │   │ • Thay Đổi  │   │ • CNN       │   │ • Lớp       │        │
│  │ • 5 Lớp     │   │   Kích Thước│   │ • Đặc Trưng │   │ • Độ Tin Cậy│        │
│  │ • Nhiều     │   │ • Chuẩn Hóa │   │ • Phân Loại │   │ • Dự Đoán   │        │
│  │   Định Dạng │   │ • Tăng Cường│   │             │   │             │        │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘        │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        QUÁ TRÌNH HUẤN LUYỆN                           │   │
│  │                                                                         │   │
│  │  Tải Dữ Liệu ──▶ Tiền Xử Lý ──▶ Huấn Luyện Mô Hình ──▶ Đánh Giá      │   │
│  │       │                 │                 │                │           │   │
│  │   4345 Ảnh        Tăng Cường        20 Epochs       Chỉ Số &          │   │
│  │   5 Lớp           Chuẩn Hóa         Kiểm Tra        Trực Quan Hóa      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         GIAO DIỆN NGƯỜI DÙNG                          │   │
│  │                                                                         │   │
│  │  Tải Ảnh Lên ──▶ Dự Đoán Thời Gian Thực ──▶ Hiển Thị Kết Quả         │   │
│  │       │                    │                      │                    │   │
│  │   Giao Diện           Suy Luận Mô Hình        Điểm Tin Cậy            │   │
│  │   Kéo & Thả           Phản Hồi Nhanh          Phản Hồi Trực Quan       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Sơ Đồ Luồng Xử Lý Chi Tiết

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW DIAGRAM                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

    📁 Ảnh Thô                        🔄 Tiền Xử Lý                   🧠 Học Sâu
┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│                 │              │                 │              │                 │
│ • File JPG/PNG  │─────────────▶│ • Tải Ảnh       │─────────────▶│ • MobileNetV2   │
│ • Kích Thước    │              │ • Đổi Kích Thước│              │   (Xương Sống)  │
│   Khác Nhau     │              │   128x128       │              │                 │
│ • 5 Danh Mục    │              │ • BGR→RGB       │              │ • Trích Xuất    │
│   - Cúc         │              │ • Chuẩn Hóa/255 │              │   Đặc Trưng     │
│   - Bồ Công Anh │              │ • Thêm Chiều    │              │                 │
│   - Hồng        │              │   Batch         │              │ • Gộp Trung     │
│   - Hướng Dương │              │                 │              │   Bình Toàn Cục │
│   - Tulip       │              │ Chỉ Huấn Luyện: │              │                 │
└─────────────────┘              │ • Xoay ±20°     │              │ • Dense(128)    │
                                 │ • Dịch Chuyển   │              │   ReLU          │
                                 │   Rộng/Cao ±20% │              │                 │
                                 │ • Lật Ngang     │              │ • Dropout(0.2)  │
                                 │ • Phóng To ±20% │              │                 │
                                 └─────────────────┘              │ • Dense(5)      │
                                                                  │   Softmax       │
                                                                  │                 │
                                                                  └─────────────────┘
                                                                           │
                                                                           ▼
    📊 Kết Quả                     🎯 Dự Đoán                    ⚡ Đầu Ra Mô Hình
┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│                 │              │                 │              │                 │
│ • Tên Lớp       │◀─────────────│ • ArgMax        │◀─────────────│ • Xác Suất     │
│ • % Tin Cậy     │              │ • Điểm Tin Cậy  │              │   [0.1, 0.8,    │
│ • Hiển Thị      │              │                 │              │    0.05, 0.03,  │
│   Trực Quan     │              │                 │              │    0.02]        │
│ Chế Độ Huấn    │              │ • Ánh Xạ Lớp   │              │                 │
│ Luyện:          │              │   0→Cúc         │              │ • Kích Thước:   │
│ • Biểu Đồ      │              │   1→Bồ Công Anh │              │   (1,5)         │
│   Độ Chính Xác │              │   2→Hồng        │              │ • Kiểu: float32 │
│ • Biểu Đồ Loss │              │   3→Hướng Dương │              │                 │
│ • Ma Trận       │              │   4→Tulip       │              │                 │
│   Nhầm Lẫn     │              │                 │              │                 │
│ • Báo Cáo      │              │                 │              │                 │
│   Phân Loại    │              │                 │              │                 │
│                 │              │                 │              │                 │
└─────────────────┘              └─────────────────┘              └─────────────────┘
```

## 3. Kiến Trúc Mô Hình Neural Network Chi Tiết

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            CNN MODEL ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────────┘

📥 LớP ĐẦU VÀO                   🏗️ XƯƠNG SỐNG MOBILENETV2              🎯 BỘ PHÂN LOẠI
┌─────────────────┐              ┌─────────────────────────────────┐   ┌─────────────┐
│                 │              │                                 │   │             │
│ Kích Thước      │─────────────▶│ 🔒 Trọng Số Đã Huấn Luyện    │──▶│ Gộp Trung   │
│ Đầu Vào:       │              │    (Bị Đông Băng)           │   │ Bình Toàn   │
│ (128, 128, 3)   │              │                                 │   │ Cục 2D      │
│                 │              │ • Tích Chập Tách Theo Độ Sâu   │   │             │
│ • Kênh RGB      │              │ • Khối Dư Thừa Đảo Ngược     │   │ Đầu Ra:     │
│ • Đã Chuẩn Hóa  │              │ • Kiến Trúc Di Động Hiệu Quả │   │ (None, 1280)│
│   [0, 1]        │              │                                 │   │             │
│ • Kích Thước    │              │ Bản Đồ Đặc Trưng:          │   └─────────────┘
│   Batch: Động   │              │ 128x128 → 64x64 → 32x32 →      │           │
│                 │              │ 16x16 → 8x8 → 4x4 → 1280       │           ▼
└─────────────────┘              │                                 │   ┌─────────────┐
                                 │ Tham Số: ~2.2M (đông băng)   │   │             │
                                 │                                 │   │ Lớp Kết Nối │
                                 └─────────────────────────────────┘   │ 128 đơn vị  │
                                                                       │ ReLU        │
                                                                       │             │
                                                                       │ Có Thể Huấn: │
                                                                       │ 163,968     │
                                                                       │ tham số     │
                                                                       │             │
                                                                       └─────────────┘
                                                                               │
                                                                               ▼
                                                                       ┌─────────────┐
                                                                       │             │
                                                                       │ Dropout     │
                                                                       │ Tỉ Lệ: 0.2   │
                                                                       │             │
                                                                       │ Ngăn Chặn   │
                                                                       │ Quá Khớp     │
                                                                       │             │
                                                                       └─────────────┘
                                                                               │
                                                                               ▼
📤 LớP ĐẦU RA                                                       ┌─────────────┐
┌─────────────────┐                                                    │             │
│                 │◀───────────────────────────────────────────────────│ Lớp Kết Nối │
│ Kích Thước      │                                                    │ 5 đơn vị    │
│ Đầu Ra:        │                                                    │ Softmax     │
│ (None, 5)       │                                                    │             │
│                 │                                                    │ Có Thể Huấn: │
│ Xác Suất Lớp:   │                                                    │ 645 tham số │
│ [P(cúc),        │                                                    │             │
│  P(bồ công anh),│                                                    │ ∑ = 1.0     │
│  P(hồng),       │                                                    │             │
│  P(hướng dương),│                                                    └─────────────┘
│  P(tulip)]      │
│                 │
│ Tổng Tham Số:   │
│ • Có Thể Huấn: 164,613
│ • Đông Băng: 2,257,984
│ • Tổng Cộng: 2,422,597
└─────────────────┘
```

## 4. Cấu Trúc Dữ Liệu & Thống Kê

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATASET STRUCTURE                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

📁 data/flowers/                    📊 Dataset Statistics
├── 🌼 daisy/                       ┌─────────────────────────────────┐
│   ├── image_001.jpg               │ Total Images: 4,345            │
│   ├── image_002.jpg               │ Classes: 5                     │
│   └── ... (633 images)            │                                │
│                                   │ Distribution:                  │
├── 🌻 dandelion/                   │ • Daisy: 633 (14.6%)          │
│   ├── image_001.jpg               │ • Dandelion: 898 (20.7%)      │
│   ├── image_002.jpg               │ • Rose: 641 (14.7%)           │
│   └── ... (898 images)            │ • Sunflower: 699 (16.1%)      │
│                                   │ • Tulip: 799 (18.4%)          │
├── 🌹 rose/                        │                                │
│   ├── image_001.jpg               │ Split Ratio:                   │
│   ├── image_002.jpg               │ • Training: 80% (3,476)       │
│   └── ... (641 images)            │ • Testing: 20% (869)          │
│                                   │                                │
├── 🌻 sunflower/                   │ Image Properties:              │
│   ├── image_001.jpg               │ • Format: JPG/PNG             │
│   ├── image_002.jpg               │ • Input Size: Variable        │
│   └── ... (699 images)            │ • Processed: 128x128x3        │
│                                   │ • Color Space: RGB            │
└── 🌷 tulip/                       │ • Pixel Range: [0, 1]         │
    ├── image_001.jpg               │                                │
    ├── image_002.jpg               └─────────────────────────────────┘
    └── ... (799 images)
```

## 5. Sơ Đồ Luồng Training & Inference

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING vs INFERENCE FLOW                           │
└─────────────────────────────────────────────────────────────────────────────────┘

🏋️ TRAINING MODE                              🚀 INFERENCE MODE
┌─────────────────────────────┐          ┌─────────────────────────────┐
│                             │          │                             │
│ 📁 Load Dataset (4,345 images)  │          │ 📷 Single Image Upload         │
│           │                 │          │           │                 │
│           ▼                 │          │           ▼                 │
│ 🔄 Data Augmentation            │          │ 🔄 Basic Preprocessing          │
│   • Rotation ±20°              │          │   • Resize to 128x128           │
│   • Width/Height Shift ±20%    │          │   • BGR → RGB                   │
│   • Horizontal Flip             │          │   • Normalize /255              │
│   • Zoom ±20%                  │          │   • Add Batch Dimension         │
│           │                 │          │           │                 │
│           ▼                 │          │           ▼                 │
│ 📊 Train/Test Split (80/20)     │          │ 🧠 Model.predict()             │
│           │                 │          │           │                 │
│           ▼                 │          │           ▼                 │
│ 🧠 Model Training               │          │ 🎯 Get Probabilities           │
│   • 20 Epochs                  │          │   • Shape: (1, 5)               │
│   • Batch Size: 32             │          │   • Softmax Output              │
│   • Adam Optimizer             │          │           │                 │
│   • Categorical Crossentropy   │          │           ▼                 │
│           │                 │          │ 📈 ArgMax + Confidence          │
│           ▼                 │          │   • Class Index                 │
│ 📈 Model Evaluation             │          │   • Confidence Score            │
│   • Test Accuracy              │          │           │                 │
│   • Confusion Matrix           │          │           ▼                 │
│   • Classification Report      │          │ 🎨 Display Results              │
│           │                 │          │   • Class Name                 │
│           ▼                 │          │   • Confidence %               │
│ 💾 Save Model (.h5)             │          │   • Visual Feedback            │
│ 💾 Save Plots & Reports         │          │                             │
│                             │          │                             │
└─────────────────────────────┘          └─────────────────────────────┘

⏱️ Time: ~30-60 minutes                      ⏱️ Time: ~1-2 seconds
💻 Resources: GPU recommended                 💻 Resources: CPU sufficient
🎯 Goal: Optimize accuracy                    🎯 Goal: Fast prediction
```

## 6. Giao Diện Người Dùng (Streamlit UI)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT INTERFACE                               │
└─────────────────────────────────────────────────────────────────────────────────┘

🌸 Header Section
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🌺 [Icon]  NHẬN DIỆN LOÀI HOA                                                 │
│             Upload ảnh hoa để nhận diện và phân loại loài hoa bằng công nghệ AI │
└─────────────────────────────────────────────────────────────────────────────────┘

📤 Upload Section                           🎯 Results Section
┌─────────────────────────────┐       ┌─────────────────────────────┐
│  🌸 Upload Ảnh                  │       │  🌼 Kết Quả Phân Loại          │
│  Kéo thả hoặc chọn ảnh hoa      │       │  Kết quả nhận diện loài hoa    │
│                                 │       │                                 │
│  ┌─────────────────────────┐    │       │  ┌─────────────────────────┐    │
│  │                         │    │       │  │                         │    │
│  │    📁 File Uploader     │    │       │  │   🌷 Đây là hoa:        │    │
│  │    Drag & Drop Zone     │    │       │  │   ROSE (85.6% tự tin)   │    │
│  │                         │    │       │  │                         │    │
│  └─────────────────────────┘    │       │  └─────────────────────────┘    │
│                                 │       │                                 │
│  [Uploaded Image Preview]       │       │  [Confidence Bar Chart]        │
│                                 │       │                                 │
└─────────────────────────────┘       └─────────────────────────────┘

🚀 Features Section
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  📤 Upload Dễ   │  │  🤖 AI Thông    │  │  🖼️ Kết Quả     │
│     Dàng        │  │     Minh        │  │     Chi Tiết    │
│                 │  │                 │  │                 │
│ Kéo thả hoặc    │  │ Công nghệ AI    │  │ Hiển thị tên,   │
│ chọn ảnh từ     │  │ nhận diện chính │  │ độ chính xác    │
│ thiết bị        │  │ xác các loài hoa│  │ và thông tin    │
└─────────────────┘  └─────────────────┘  └─────────────────┘

📊 Training Results Section
┌─────────────────────────────────────────────────────────────────────────────────┐
│  📊 Kết Quả Huấn Luyện Mô Hình                                                 │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │  📈 Accuracy    │  │  📉 Loss Plot   │  │  🎯 Confusion   │                │
│  │     Plot        │  │                 │  │     Matrix      │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  📋 Classification Report                                              │   │
│  │      precision    recall  f1-score   support                          │   │
│  │                                                                         │   │
│  │      daisy         0.85      0.82      0.83       127                 │   │
│  │      dandelion     0.89      0.91      0.90       180                 │   │
│  │      rose          0.83      0.85      0.84       128                 │   │
│  │      sunflower     0.87      0.84      0.85       140                 │   │
│  │      tulip         0.86      0.88      0.87       160                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```