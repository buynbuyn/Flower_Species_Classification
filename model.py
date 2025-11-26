import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

class FlowerClassificationModel:
    def __init__(self, input_shape=(128, 128, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        """Xây dựng mô hình CNN với MobileNetV2 backbone"""
        # Base model với pretrained weights
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Thêm custom classifier
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Tạo model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_model_summary(self):
        """Hiển thị thông tin mô hình"""
        if self.model:
            return self.model.summary()
        return "Model chưa được xây dựng"
    
    def save_model(self, filepath):
        """Lưu mô hình"""
        if self.model:
            self.model.save(filepath)
            print(f"Model đã được lưu tại: {filepath}")
    
    def load_model(self, filepath):
        """Tải mô hình đã lưu"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model đã được tải từ: {filepath}")
        return self.model

def create_flower_model():
    """Hàm tiện ích để tạo model nhanh"""
    flower_model = FlowerClassificationModel()
    return flower_model.build_model()