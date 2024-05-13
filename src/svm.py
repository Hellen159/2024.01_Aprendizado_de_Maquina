import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.io import imread

# Função para carregar as imagens e seus rótulos
def load_images_and_labels(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img = imread(os.path.join(directory, filename), as_gray=True)
            images.append(img.flatten())  # Achatando a imagem para um vetor unidimensional
            if "Non" in directory:
                labels.append(0)  # 0 para ossos não fraturados
            else:
                labels.append(1)  # 1 para ossos  fraturados
    return np.array(images), np.array(labels)

# Carregando imagens e rótulos para mãos fraturadas e não fraturadas
fractured_images, fractured_labels = load_images_and_labels("data/hand/Fractured")
non_fractured_images, non_fractured_labels = load_images_and_labels("data/hand/Non-Fractured")

# Concatenando as imagens e rótulos
X = np.concatenate((fractured_images, non_fractured_images))
y = np.concatenate((fractured_labels, non_fractured_labels))

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificando se há pelo menos duas classes nos rótulos de treinamento
if len(np.unique(y_train)) < 2:
    raise ValueError("Há apenas uma classe nos dados de treinamento. Certifique-se de que há duas classes presentes.")

# Definindo os hiperparâmetros para ajustar
parameters = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['linear', 'rbf']}

# Criando Instância do modelo svm
svm_model = SVC()

# Criando uma instância do GridSearchCV com a métrica F1-score
grid_search = GridSearchCV(svm_model, parameters, cv=5, scoring='f1')

# Tereino de modelo
print("Iniciando treino de modelo...")
grid_search.fit(X_train, y_train)
print("Treino finalizado!")

# Melhores parâmetros encontrados
print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)

# Fazendo previsões no conjunto de teste
print("Iniciando testes...")
y_pred = grid_search.predict(X_test)
print("Testes Finalizados!\n")

# Calculando métricas de desempenho
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Acuracia:", accuracy)
print("Precisao:", precision)
print("Recall:", recall)
print("F1-score:", f1)