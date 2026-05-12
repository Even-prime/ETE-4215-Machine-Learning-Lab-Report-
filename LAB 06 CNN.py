# 1. Install and Import Libraries
!pip install medmnist

import os
import numpy as np
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 2. Load PneumoniaMNIST Dataset
data_flag = 'pneumoniamnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# Downloading and loading splits
train_dataset = DataClass(split='train', download=True)
test_dataset = DataClass(split='test', download=True)

# Preprocessing: Normalize pixel values to [0, 1]
X_train = train_dataset.imgs.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = train_dataset.labels
X_test = test_dataset.imgs.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_test = test_dataset.labels

print(f"Data Loaded: {len(X_train)} Train images, {len(X_test)} Test images")

# 3. Define CNN Model
# Preserves spatial hierarchies through convolutional and pooling layers
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)), 
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(), 
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid') 
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Training
print("\n--- Training CNN ---")
history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# 5. Evaluation
cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)

print("\n" + "="*40)
print(f"Final CNN Accuracy: {cnn_acc * 100:.2f}%")
print("="*40)

# 6. Visualize Learning Curves
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Performance on PneumoniaMNIST')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
