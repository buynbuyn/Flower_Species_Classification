def train_model():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import matplotlib.pyplot as plt
    from preprocess import load_train_val_data   # dùng hàm mới
    from model_cnn import build_model
    from tensorflow.keras.callbacks import EarlyStopping

    # Tạo thư mục nếu chưa có
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load dữ liệu train/val và class_names
    (X_train, X_val, y_train, y_val), class_names = load_train_val_data("data/flowers", img_size=128)

    # Xây mô hình (num_classes động theo số lớp hoa)
    model = build_model(num_classes=len(class_names))

    # Callback dừng sớm
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Huấn luyện mô hình
    history = model.fit(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    # Lưu mô hình
    model.save('models/flower_cnn_model.h5')

    # Vẽ biểu đồ Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy_plot.png')
    plt.close()

    # Vẽ biểu đồ Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.savefig('outputs/loss_plot.png')
    plt.close()

    print("✅ Huấn luyện xong. Mô hình và biểu đồ đã được lưu.")

    return model, history, class_names


if __name__ == "__main__":
    train_model()