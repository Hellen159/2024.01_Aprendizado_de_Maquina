import pandas as pd
import os
import shutil

df = pd.read_csv('data/dataset.csv')

print(df.sort_values(['hip', 'fractured', 'mixed'], ascending=[False, False, True])[0:20])

path = 'data/dataset1-Fractured'
path2 = 'data/dataset1-Non-fractured'

for index, row in df.iterrows():
    nome_arquivo = row['image_id']
    if(row['fractured']):
        caminho_arquivo = os.path.join(path, nome_arquivo)
        if os.path.exists(caminho_arquivo):
            if(row['mixed']):
                shutil.move(caminho_arquivo, 'data/mixed/Fractured')
            elif(row['shoulder']):
                shutil.move(caminho_arquivo, 'data/shoulder/Fractured')
            elif(row['hip']):
                shutil.move(caminho_arquivo, 'data/hip/Fractured')
            elif(row['leg']):
                shutil.move(caminho_arquivo, 'data/leg/Fractured')
            elif(row['hand']):
                shutil.move(caminho_arquivo, 'data/hand/Fractured')
    else:
        caminho_arquivo = os.path.join(path2, nome_arquivo)
        if os.path.exists(caminho_arquivo):
            if(row['mixed']):
                shutil.move(caminho_arquivo, 'data/mixed/Non-Fractured')
            elif(row['shoulder']):
                shutil.move(caminho_arquivo, 'data/shoulder/Non-Fractured')
            elif(row['hip']):
                shutil.move(caminho_arquivo, 'data/hip/Non-Fractured')
            elif(row['leg']):
                shutil.move(caminho_arquivo, 'data/leg/Non-Fractured')
            elif(row['hand']):
                shutil.move(caminho_arquivo, 'data/hand/Non-Fractured')

print('Fim')