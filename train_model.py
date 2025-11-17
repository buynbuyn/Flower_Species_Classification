import os
import matplotlib.pyplot as plt
from model import FlowerClassificationModel
from preprocess import ImagePreprocessor
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def train_flower_classification_model():
    """Huấn luyện mô hình phân loại hoa"""
    
    # 1. Tiền xử lý dữ liệu
    print("=== BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU ===")
    preprocessor = ImagePreprocessor(img_size=128, test_size=0.2)
    X, y = preprocessor.load_and_preprocess_data('data/flowers')
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 2. Xây dựng mô hình
    print("\n=== BƯỚC 2: XÂY DỰNG MÔ HÌNH ===")
    flower_model = FlowerClassificationModel(num_classes=len(preprocessor.class_names))
    model = flower_model.build_model()
    
    print("Kiến trúc mô hình:")
    model.summary()
    
    # 3. Huấn luyện mô hình
    print("\n=== BƯỚC 3: HUẤN LUYỆN MÔ HÌNH ===")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=20,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # 4. Đánh giá mô hình
    print("\n=== BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH ===")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # 5. Dự đoán và tạo báo cáo
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    report = classification_report(
        y_true_classes, y_pred_classes, 
        target_names=preprocessor.class_names
    )
    print("\nClassification Report:")
    print(report)
    
    # 6. Lưu kết quả
    print("\n=== BƯỚC 5: LÀM KẾT QUẢ ===")
    os.makedirs('outputs', exist_ok=True)
    
    # Lưu mô hình
    os.makedirs('models', exist_ok=True)
    model.save('models/flower_cnn_model.h5')
    
    # Lưu báo cáo
    with open('outputs/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Vẽ biểu đồ training
    plot_training_history(history)
    
    # Vẽ confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, preprocessor.class_names)
    
    print("Hoàn thành! Các file đã được lưu trong thư mục outputs/ và models/")

def plot_training_history(history):
    """Vẽ biểu đồ quá trình huấn luyện"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('outputs/loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Vẽ confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    train_flower_classification_model()