# Sơ Đồ Tổng Thể Hệ Thống Phân Loại Hoa

## 1. Mô Hình Tổng Thể

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dữ liệu       │    │   Tiền xử lý    │    │   Mô hình       │
│   đầu vào       │───▶│   ảnh           │───▶│   học sâu       │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   Giao diện     │    │   Huấn luyện    │◀────────────┘
│   người dùng    │◀───│   và đánh giá   │
│                 │    │                 │
└─────────────────┘    └─────────────────┘
```

## 2. Sơ Đồ Luồng Chi Tiết

```
Input Image
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    TIỀN XỬ LÝ ẢNH                           │
│  • Resize ảnh (128x128)                                     │
│  • Chuẩn hóa pixel (0-1)                                    │
│  • Augmentation (training)                                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                   MÔ HÌNH HỌC SÂU                          │
│  • Base: MobileNetV2 (pretrained)                          │
│  • GlobalAveragePooling2D                                  │
│  • Dense(128, relu)                                        │
│  • Dense(5, softmax) → 5 classes                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                 HUẤN LUYỆN & ĐÁNH GIÁ                      │
│  • Train/Validation Split                                   │
│  • Metrics: Accuracy, Loss                                 │
│  • Confusion Matrix                                        │
│  • Classification Report                                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                 GIAO DIỆN NGƯỜI DÙNG                       │
│  • Upload ảnh (Streamlit)                                  │
│  • Hiển thị kết quả dự đoán                               │
│  • Độ tin cậy (confidence)                                │
│  • Biểu đồ training results                               │
└─────────────────────────────────────────────────────────────┘
```

## 3. Cấu Trúc Dữ Liệu

```
data/flowers/
├── daisy/       (Hoa cúc)
├── dandelion/   (Hoa bồ công anh)  
├── rose/        (Hoa hồng)
├── sunflower/   (Hoa hướng dương)
└── tulip/       (Hoa tulip)
```

## 4. Kiến Trúc Mô Hình

- **Input**: Ảnh RGB 128x128x3
- **Backbone**: MobileNetV2 (ImageNet pretrained)
- **Classifier**: Dense layers với 5 outputs
- **Output**: Xác suất cho 5 loại hoa