import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Função para carregar imagens e rótulos
def load_data(data_dir):
    images = []
    labels = []
    class_names = ["Fractured", "Non-Fractured"]
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_height, image_width))  # Redimensionando as imagens
            images.append(image)
            labels.append(class_names.index(class_name))
    
    return np.array(images), np.array(labels)


size_images = 64
data_dir = "data/hand"
image_height, image_width = size_images, size_images  # Definindo o tamanho das imagens

# Carregando os dados
images, labels = load_data(data_dir)
#images = np.expand_dims(images, axis=-1)

# Dividindo os dados em conjuntos de treinamento, validacao e teste
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Normalizando os pixels das imagens para o intervalo [0, 1]
train_images = train_images.astype('float32') / 255.0
validation_images = validation_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Definindo o modelo CNN
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

# Compilando o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinando o modelo
model.fit(train_images, train_labels, epochs=10, validation_data=(validation_images, validation_labels))

# Avaliando o modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
