def train_model():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import matplotlib.pyplot as plt
    from preprocess import load_data
    from model_cnn import build_model
    from tensorflow.keras.callbacks import EarlyStopping

    # Tạo thư mục nếu chưa có
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load dữ liệu và class_names
    (X_train, X_test, y_train, y_test), class_names = load_data()

    # Xây mô hình
    model = build_model()

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
        validation_data=(X_test, y_test),
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

    print("Huấn luyện xong. Mô hình và biểu đồ đã được lưu.")