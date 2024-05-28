import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator
import tensorflow as tf
from tensorflow.keras import layers, models

# Define a wrapper class for Keras model
class KerasClassifierWrapper(BaseEstimator):
    def __init__(self, model, batch_size=32, epochs=10):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
    
    def fit(self, X, y):
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2)
        return self
    
    def score(self, X, y):
        _, accuracy = self.model.evaluate(X, y)
        return accuracy

# Function to load images and labels
def load_data(data_dir):
    images = []
    labels = []
    class_names = ["Fractured", "Non-Fractured"]
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_height, image_width))  # Resizing images
            images.append(image)
            labels.append(class_names.index(class_name))
    
    return np.array(images), np.array(labels)

size_images = 64
data_dir = "data/hand"
image_height, image_width = size_images, size_images  # Image size

# Load data
images, labels = load_data(data_dir)

# Split data into train, validation, and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Normalize pixel values to [0, 1]
train_images = train_images.astype('float32') / 255.0
validation_images = validation_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Define CNN model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define hyperparameters for tuning
parameters = {'batch_size': [32, 64, 128], 'epochs': [10, 20, 30]}

# Create GridSearchCV instance with Keras model wrapper
grid_search = GridSearchCV(KerasClassifierWrapper(model), parameters, cv=5)

# Train the model
print("Training model...")
grid_search.fit(train_images, train_labels)
print("Training completed!")

# Best parameters found
print("Best parameters found:")
print(grid_search.best_params_)

# Evaluate the model
test_accuracy = grid_search.best_estimator_.score(test_images, test_labels)
print('Test accuracy:', test_accuracy)
