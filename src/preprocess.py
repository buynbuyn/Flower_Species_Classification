import os, cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(data_dir='data/flowers', img_size=128):
    categories = sorted(os.listdir(data_dir))  # sắp xếp để nhãn nhất quán
    data = []

    for label, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        if not os.path.isdir(path):
            continue  # bỏ qua nếu không phải thư mục

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue  # bỏ qua nếu ảnh hỏng
                img = cv2.resize(img, (img_size, img_size))
                data.append([img, label])
            except:
                continue

    if not data:
        raise ValueError("Không có ảnh nào được load từ thư mục.")

    # Tách ảnh và nhãn
    X = np.array([item[0] for item in data], dtype=np.float32) / 255.0
    y = np.array([item[1] for item in data])

    # Chuyển nhãn sang one-hot
    num_classes = len(categories)
    y = to_categorical(y, num_classes=num_classes)

    # Trả về thêm class_names để dùng trong báo cáo
    class_names = categories

    return train_test_split(X, y, test_size=0.15, random_state=42), class_names