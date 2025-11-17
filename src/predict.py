import numpy as np
from PIL import Image
import tensorflow as tf

# Load model và class một lần
model = tf.keras.models.load_model('models/flower_cnn_model.h5')
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def preprocess_image(image, target_size=(128, 128)):
    # Chuyển ảnh sang RGB nếu cần
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, target_size[0], target_size[1], 3)
    return img_array

def predict_flower(image, class_names=classes):
    try:
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        return predicted_class, confidence
    except Exception as e:
        print(f"Lỗi khi dự đoán: {e}")
        return "Không xác định", 0.0