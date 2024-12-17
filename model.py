import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

class CrowdCounter:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        return self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split
        )
    
    def convert_to_tflite(self, output_path='pedestrian_detector.tflite'):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {output_path}")
    
    def predict(self, image):
        return self.model.predict(np.expand_dims(image, axis=0))


if __name__ == "__main__":
    detector = CrowdCounter()
    
    X_dummy = np.random.random((100, 224, 224, 3))
    y_dummy = np.random.random((100, 1))
    
    # Train model
    history = detector.train(X_dummy, y_dummy)
    
    # Convert to TFLite
    detector.convert_to_tflite()
