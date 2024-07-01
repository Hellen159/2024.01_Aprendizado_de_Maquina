import os
import cv2
import numpy as np
import tensorflow as tf

# Define os tamanhos das imagens
size_images = 64
image_height, image_width = size_images, size_images

# Carrega o modelo treinado
model = tf.keras.models.load_model('trained_model_with_body_part.h5')

def predict_and_save(image_path, save_dir):
    # Inferir a parte do corpo a partir do caminho do arquivo
    body_part = None
    body_part_dict = {'hand': 0, 'hip': 1, 'leg': 2, 'mixed': 3, 'shoulder': 4}
    for part in body_part_dict.keys():
        if part in image_path:
            body_part = body_part_dict[part]
            break

    if body_part is None:
        print(f"Could not determine body part from path: {image_path}")
        return
    
    # Carrega e processa a imagem
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (image_height, image_width))
    image_normalized = image_resized.astype('float32') / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    
    # Faz a previsão
    prediction = model.predict([image_expanded, np.array([[body_part]])])
    predicted_fracture = (prediction > 0.5).astype("int32")[0][0]
    
    # Verifica se a pasta existe, caso contrário, cria
    part_dir = os.path.join(save_dir, list(body_part_dict.keys())[body_part])
    if not os.path.exists(part_dir):
        os.makedirs(part_dir)
    
    # Define o nome do arquivo para salvar
    base_name = os.path.basename(image_path)
    save_path = os.path.join(part_dir, base_name)
    
    # Salva a imagem na pasta correspondente
    cv2.imwrite(save_path, image)
    
    # Imprime o resultado da previsão
    fracture_status = "Fracture detected" if predicted_fracture == 1 else "No fracture detected"
    print(f"Prediction for {base_name} ({list(body_part_dict.keys())[body_part]}): {fracture_status}")

# Exemplo de uso da função
predict_and_save('../image-teste-cnn/hand/teste11.jpg', '../data-teste-cnn')

