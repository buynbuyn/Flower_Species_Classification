def evaluate_model(model, X_test, y_test, class_names, history=None):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    # Tạo thư mục nếu chưa có
    os.makedirs('outputs', exist_ok=True)

    # Dự đoán
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # In và lưu báo cáo đánh giá
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    print(report)

    with open('outputs/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # Vẽ và lưu confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()

    # Nếu có history thì vẽ thêm biểu đồ Accuracy và Loss
    if history:
        # Accuracy
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('outputs/accuracy_plot.png')
        plt.close()

        # Loss
        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('outputs/loss_plot.png')
        plt.close()

    print("Đánh giá xong. Đã lưu báo cáo và biểu đồ vào thư mục Outputs.")