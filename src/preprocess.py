import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_train_val_data(data_dir='data/flowers', img_size=128, test_size=0.15):
    categories = sorted(os.listdir(data_dir))
    data = []

    for label, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        if not os.path.isdir(path):
            continue
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                data.append([img, label])
            except:
                continue

    if not data:
        raise ValueError("Không có ảnh nào được load từ thư mục train/val.")

    X = np.array([item[0] for item in data], dtype=np.float32) / 255.0
    y = np.array([item[1] for item in data])

    num_classes = len(categories)
    y = to_categorical(y, num_classes=num_classes)

    class_names = categories
    (X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=test_size, random_state=42)

    print("✅ Train/Val shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    return (X_train, X_val, y_train, y_val), class_names


def load_test_data(test_dir='data/flower_test', img_size=128, class_names=None):
    categories = sorted(os.listdir(test_dir)) if class_names is None else class_names
    data = []

    for label, category in enumerate(categories):
        path = os.path.join(test_dir, category)
        if not os.path.isdir(path):
            continue
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                data.append([img, label])
            except:
                continue

    if not data:
        raise ValueError("Không có ảnh nào được load từ thư mục test.")

    X_test = np.array([item[0] for item in data], dtype=np.float32) / 255.0
    y_test = np.array([item[1] for item in data])

    if class_names is not None:
        y_test = to_categorical(y_test, num_classes=len(class_names))

    print("✅ Test shapes:", X_test.shape, y_test.shape)
    return X_test, y_test