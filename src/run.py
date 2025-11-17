import subprocess
from train import train_model
from evaluate import evaluate_model
from preprocess import load_data
from model_cnn import build_model

# === 1. Huấn luyện mô hình ===
print("Đang huấn luyện mô hình...")
train_model()

# === 2. Đánh giá mô hình ===
print("Đang đánh giá mô hình...")
(X_train, X_test, y_train, y_test), class_names = load_data()
model = build_model()
model.load_weights("models/flower_cnn_model.h5")
evaluate_model(model, X_test, y_test, class_names)

# === 3. Mở giao diện Streamlit ===
print("Đang khởi chạy giao diện Streamlit...")
subprocess.run(["streamlit", "run", "main.py"])