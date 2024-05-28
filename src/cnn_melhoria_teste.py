import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf

def load_new_data(data_dir, target_height=64, target_width=64):
    images = []
    labels = []
    class_names =  {"Fractured": 0, "Non-Fractured": 1}  # Mapear nomes de diretórios para rótulos
    
    for class_name, label in class_names.items():
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Carrega a imagem em escala de cinza
                image = cv2.resize(image, (target_height, target_width))
                image = np.expand_dims(image, axis=-1)  # Adiciona uma dimensão para o canal
                images.append(image)
                labels.append(label)
        else:
            print(f"Diretório {class_dir} não encontrado, pulando...")

    return np.array(images), np.array(labels)

# Define o diretório da nova base de dados e os tamanhos das imagens
new_data_dir = "../data-teste"
image_height, image_width = 64, 64

# Carrega a nova base de dados
new_images, new_labels = load_new_data(new_data_dir, image_height, image_width)

# Verifica se há dados carregados
if len(new_images) == 0 or len(new_labels) == 0:
    raise ValueError("Nenhum dado foi carregado. Verifique o diretório da base de dados.")

# Normaliza os pixels das imagens para o intervalo [0, 1]
new_images = new_images.astype('float32') / 255.0

# Carrega o modelo salvo
model = tf.keras.models.load_model('fracture_detection_model.h5')

# Avalia o modelo na nova base de dados
new_predictions = np.argmax(model.predict(new_images), axis=1)
new_f1 = f1_score(new_labels, new_predictions)
new_accuracy = accuracy_score(new_labels, new_predictions)
new_error_rate = 1 - new_accuracy

print('New data accuracy:', new_accuracy)
print('New data F1-score:', new_f1)
print('New data Error rate:', new_error_rate)
