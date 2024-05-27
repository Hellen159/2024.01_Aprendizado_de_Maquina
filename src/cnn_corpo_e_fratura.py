import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models

def load_data(data_dirs):
    images = []
    labels = []
    
    for idx, data_dir in enumerate(data_dirs):
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist.")
            continue
        for image_name in os.listdir(data_dir):
            image_path = os.path.join(data_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_height, image_width))  # Resize the images
            images.append(image)
            labels.append(idx)
    
    return np.array(images), np.array(labels)

# Define os diretórios de dados e os tamanhos das imagens
size_images = 64
data_dirs = ["data/hand/Fractured", "data/hand/Non-Fractured", "data/hip/Fractured", "data/hip/Non-Fractured", "data/leg/Fractured", "data/leg/Non-Fractured", "data/mixed/Fractured", "data/mixed/Non-Fractured", "data/shoulder/Fractured", "data/shoulder/Non-Fractured"]
image_height, image_width = size_images, size_images

# Carrega os dados
images, labels = load_data(data_dirs)

# Divide os dados em conjuntos de treinamento, validação e teste
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Normaliza os pixels das imagens para o intervalo [0, 1]
train_images = train_images.astype('float32') / 255.0
validation_images = validation_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Define o modelo CNN
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compila o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define métricas personalizadas
def f1_metric(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average='weighted')

# Treina o modelo
model.fit(train_images, train_labels, epochs=20, validation_data=(validation_images, validation_labels))

# Avalia o modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_predictions = np.argmax(model.predict(test_images), axis=1)
f1 = f1_score(test_labels, test_predictions, average='weighted')
error_rate = 1 - accuracy_score(test_labels, test_predictions)

print('Test accuracy:', test_acc)
print('F1-score:', f1)
print('Error rate:', error_rate)
