import subprocess
from train import train_model
from evaluate import evaluate_model
from preprocess import load_test_data
from tensorflow.keras.models import load_model

# === 1. Huấn luyện mô hình ===
print("🌼 Đang huấn luyện mô hình...")
model, history, class_names = train_model()

# === 2. Đánh giá mô hình trên bộ test ===
print("🔍 Đang đánh giá mô hình trên bộ test...")
X_test, y_test = load_test_data("data/flower_test", img_size=128, class_names=class_names)

# Load lại model đã lưu (đảm bảo dùng đúng weights)
model = load_model("models/flower_cnn_model.h5")

# Đánh giá
evaluate_model(model, X_test, y_test, class_names, history)

# === 3. Mở giao diện Streamlit ===
print("🚀 Đang khởi chạy giao diện Streamlit...")
subprocess.run(["streamlit", "run", "main.py"])