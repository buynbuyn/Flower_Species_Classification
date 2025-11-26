import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImagePreprocessor:
    def __init__(self, img_size=128, test_size=0.2, random_state=42):
        self.img_size = img_size
        self.test_size = test_size
        self.random_state = random_state
        self.class_names = []
    
    def load_and_preprocess_data(self, data_dir='data/flowers'):
        """Tải và tiền xử lý dữ liệu ảnh"""
        print(f"Đang tải dữ liệu từ: {data_dir}")
        
        # Lấy danh sách các class
        self.class_names = sorted([d for d in os.listdir(data_dir) 
                                 if os.path.isdir(os.path.join(data_dir, d))])
        print(f"Các loại hoa: {self.class_names}")
        
        images = []
        labels = []
        
        # Duyệt qua từng class
        for label, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_dir, class_name)
            print(f"Đang xử lý class: {class_name}")
            
            # Đếm số ảnh trong class
            img_count = 0
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                
                # Kiểm tra file ảnh hợp lệ
                if not self._is_valid_image(img_path):
                    continue
                
                # Đọc và xử lý ảnh
                img = self._process_single_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)
                    img_count += 1
            
            print(f"  - Đã tải {img_count} ảnh")
        
        if not images:
            raise ValueError("Không tìm thấy ảnh hợp lệ nào!")
        
        # Chuyển đổi sang numpy arrays
        X = np.array(images, dtype=np.float32)
        y = np.array(labels)
        
        # Chuẩn hóa pixel values (0-1)
        X = X / 255.0
        
        # One-hot encoding cho labels
        y = to_categorical(y, num_classes=len(self.class_names))
        
        print(f"Tổng cộng: {len(X)} ảnh, Shape: {X.shape}")
        return X, y
    
    def _is_valid_image(self, img_path):
        """Kiểm tra file có phải ảnh hợp lệ không"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        return any(img_path.lower().endswith(ext) for ext in valid_extensions)
    
    def _process_single_image(self, img_path):
        """Xử lý một ảnh đơn lẻ"""
        try:
            # Đọc ảnh
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Chuyển BGR sang RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize ảnh
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            return img
        except Exception as e:
            print(f"Lỗi xử lý ảnh {img_path}: {e}")
            return None
    
    def split_data(self, X, y):
        """Chia dữ liệu train/test"""
        return train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
    
    def create_data_generator(self, validation_split=0.2):
        """Tạo data generator với augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=validation_split
        )
        
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        return train_datagen, val_datagen
    
    def preprocess_single_image(self, img_path_or_array):
        """Tiền xử lý một ảnh để dự đoán"""
        if isinstance(img_path_or_array, str):
            # Nếu là đường dẫn file
            img = self._process_single_image(img_path_or_array)
        else:
            # Nếu là numpy array hoặc PIL Image
            img = np.array(img_path_or_array)
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        if img is None:
            return None
        
        # Chuẩn hóa và thêm batch dimension
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img

def load_flower_data(data_dir='data/flowers', img_size=128):
    """Hàm tiện ích để tải dữ liệu nhanh"""
    preprocessor = ImagePreprocessor(img_size=img_size)
    X, y = preprocessor.load_and_preprocess_data(data_dir)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    return (X_train, X_test, y_train, y_test), preprocessor.class_names