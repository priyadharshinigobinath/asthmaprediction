# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 00:17:46 2024

@author: siddh
""" 

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to your train and test datasets
train_data_dir = 'C:/Users/priya/OneDrive/Desktop/AshtmaPrediction/dataset/train'
test_data_dir = 'C:/Users/priya/OneDrive/Desktop/AshtmaPrediction/dataset/test'

# Define image dimensions 
image_height, image_width = 150, 150
num_channels = 3  # Assuming RGB images
num_classes = 2  # Adjust based on your problem

# Define batch size
batch_size = 32

# Define data generators for train and test datasets
train_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values
test_datagen = ImageDataGenerator(rescale=1./255)   # Rescale pixel values

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='binary')  # Use 'categorical' for multi-class classification

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='binary')  # Use 'categorical' for multi-class classification

# Define your CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using fit_generator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=2,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model
test_loss, test_accuracy = model.evaluate_generator(test_generator, steps=test_generator.samples // batch_size)
print("Test Accuracy:", test_accuracy)

model.save("asthma_model.h5")

# Generate predictions
predictions = model.predict_generator(test_generator)
predicted_classes = predictions.argmax(axis=1)
true_classes = test_generator.classes
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Get class labels from the generator
class_labels = list(test_generator.class_indices.keys())

# Generate predictions
predictions = model.predict_generator(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Get true classes
true_classes = test_generator.classes

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()