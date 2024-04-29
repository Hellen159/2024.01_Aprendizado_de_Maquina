import cv2
import os


diretorio_origem = "../data/fracAtlas/Fractured"
diretorio_destino = "../data/dataset1-Fractured"


novo_tamanho = (224, 224)

# Lista para armazenar os valores de intensidade de pixel
valores_pixel = []


for filename in os.listdir(diretorio_origem):
    if filename.endswith(".jpg"):  
        # Carregar a imagem em escala de cinza
        imagem = cv2.imread(os.path.join(diretorio_origem, filename), cv2.IMREAD_GRAYSCALE)
        
        # Redimensionar a imagem
        imagem_redimensionada = cv2.resize(imagem, novo_tamanho)
        
        # Normalizar a intensidade de pixel da imagem
        valores_pixel.extend(imagem_redimensionada.flatten())

# Calcular o valor mínimo e máximo dos pixels
valor_minimo = min(valores_pixel)
valor_maximo = max(valores_pixel)


for filename in os.listdir(diretorio_origem):
    if filename.endswith(".jpg"): 
        # Carregar a imagem em escala de cinza
        imagem = cv2.imread(os.path.join(diretorio_origem, filename), cv2.IMREAD_GRAYSCALE)
        
        # Redimensionar a imagem
        imagem_redimensionada = cv2.resize(imagem, novo_tamanho)
        
        # Normalizar a intensidade de pixel da imagem
        imagem_normalizada = (imagem_redimensionada - valor_minimo) / (valor_maximo - valor_minimo)
        
        # Salvar a imagem processada no diretório de destino
        cv2.imwrite(os.path.join(diretorio_destino, filename), imagem_normalizada * 255.0)

print("Processamento concluído!")
