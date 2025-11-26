import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from preprocess import load_test_data, load_train_val_data   # gom chung import

def evaluate_model(model, X_test, y_test, class_names, history=None):
    # Tắt warning của TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Tạo thư mục outputs nếu chưa có
    os.makedirs('outputs', exist_ok=True)

    # Dự đoán trên tập test
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # In và lưu classification report
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    print("🔍 Classification report trên bộ TEST:")
    print(report)

    with open('outputs/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write("🔍 Classification report trên bộ TEST:\n")
        f.write(report)

    # Vẽ và lưu confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()

    # Nếu có history thì vẽ thêm Accuracy và Loss
    if history:
        # Accuracy
        plt.figure()
        plt.plot(history.history['accuracy'], label='Test Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('outputs/accuracy_plot.png')
        plt.close()

        # Loss
        plt.figure()
        plt.plot(history.history['loss'], label='Test Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('outputs/loss_plot.png')
        plt.close()

    print("✅ Đánh giá TEST xong. Báo cáo và biểu đồ đã lưu trong thư mục outputs.")


if __name__ == "__main__":
    # Load model đã train
    model = load_model("models/flower_cnn_model.h5")

    # Load class_names từ train data
    (_, _, _, _), class_names = load_train_val_data("data/flowers")

    # Load test data
    X_test, y_test = load_test_data("data/flower_test", class_names=class_names)

    # Chạy đánh giá trên TEST
    evaluate_model(model, X_test, y_test, class_names)