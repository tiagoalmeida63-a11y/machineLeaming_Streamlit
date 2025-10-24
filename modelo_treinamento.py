import pandas as pd
import joblib
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics

#etapa 01 - carregar dados
def carregar_dados(caminho_arquivo = "historicoAcademico.csv"):
    try:
        if os.path.exists(caminho_arquivo):

            df = pd.read_csv(caminho_arquivo, )

            print("Oarquivo foi carregado com sucesso!")

            return df
        else:
            print("O aquivo não foi encontrado dentro da pasta!")

            return None
    except Exception as e:
        print("Erro inesperado ao carregar o arquivo: ", e)

        return None

#----------chamar a funcao pra armazenar o resultado--------

dados = carregar_dados()
print(dados)

