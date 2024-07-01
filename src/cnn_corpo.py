import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model

def load_data(data_dirs):
    images = []
    body_parts = []
    labels = []
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist.")
            continue
        for image_name in os.listdir(data_dir):
            image_path = os.path.join(data_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_height, image_width))  # Resize the images
            images.append(image)
            
            if 'hand' in data_dir:
                body_parts.append(0)
            elif 'hip' in data_dir:
                body_parts.append(1)
            elif 'leg' in data_dir:
                body_parts.append(2)
            elif 'mixed' in data_dir:
                body_parts.append(3)
            elif 'shoulder' in data_dir:
                body_parts.append(4)
                
            labels.append(0 if 'Non-Fractured' in data_dir else 1)
    
    return np.array(images), np.array(body_parts), np.array(labels)

# Define os diretórios de dados e os tamanhos das imagens
size_images = 64
data_dirs = ["../data/hand/Fractured", "../data/hand/Non-Fractured", "../data/hip/Fractured", "../data/hip/Non-Fractured", "../data/leg/Fractured", "../data/leg/Non-Fractured", "../data/mixed/Fractured", "../data/mixed/Non-Fractured", "../data/shoulder/Fractured", "../data/shoulder/Non-Fractured"]
image_height, image_width = size_images, size_images

# Carrega os dados
images, body_parts, labels = load_data(data_dirs)

# Divide os dados em conjuntos de treinamento, validação e teste
train_images, test_images, train_body_parts, test_body_parts, train_labels, test_labels = train_test_split(images, body_parts, labels, test_size=0.2, random_state=42)
train_images, validation_images, train_body_parts, validation_body_parts, train_labels, validation_labels = train_test_split(train_images, train_body_parts, train_labels, test_size=0.2, random_state=42)

# Normaliza os pixels das imagens para o intervalo [0, 1]
train_images = train_images.astype('float32') / 255.0
validation_images = validation_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Define o modelo CNN com entrada adicional para a parte do corpo
input_image = Input(shape=(image_height, image_width, 3))
x = layers.Conv2D(64, (3, 3), activation='relu')(input_image)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(256, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)

input_body_part = Input(shape=(1,))
body_part_embedding = layers.Embedding(input_dim=5, output_dim=4)(input_body_part)
body_part_flat = layers.Flatten()(body_part_embedding)

combined = layers.concatenate([x, body_part_flat])
combined = layers.Dense(256, activation='relu')(combined)
combined = layers.Dropout(0.5)(combined)
combined = layers.Dense(128, activation='relu')(combined)
combined = layers.Dropout(0.5)(combined)
output = layers.Dense(1, activation='sigmoid')(combined)

model = Model(inputs=[input_image, input_body_part], outputs=output)

# Compila o modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treina o modelo
model.fit([train_images, train_body_parts], train_labels, epochs=20, validation_data=([validation_images, validation_body_parts], validation_labels))

# Avalia o modelo
test_loss, test_acc = model.evaluate([test_images, test_body_parts], test_labels)
test_predictions = (model.predict([test_images, test_body_parts]) > 0.5).astype("int32")
f1 = f1_score(test_labels, test_predictions, average='weighted')
error_rate = 1 - accuracy_score(test_labels, test_predictions)

print('Test accuracy:', test_acc)
print('F1-score:', f1)
print('Error rate:', error_rate)

# Salva o modelo treinado
model.save('trained_model_with_body_part.h5')
