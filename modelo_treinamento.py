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

#--------------Preparacao e divisao dos dados----------------
#---Definicao de X(FEATURES) e Y(TARGET)-------------

if dados is not None:
    print("\nTotal de Registros carregados: {len(dados)}")
    print("Iniciando o pipeline de treinamento")
    TARGET_COLUMN = "Status_Final" 
#etapa 2.1 - definicao das reatures e target
    try:
        X = dados.drop(TARGET_COLUMN, axis=1)
        Y = dados[TARGET_COLUMN]

        print(f"FEATURES (X) DEFINIDAS: {list(X.columns)}")
        print(f"Features (Y) definidas: {TARGET_COLUMN}")

    except KeyError:
    
        print(f"\n----------Erro critico--------")
        print(f"A coluna {TARGET_COLUMN} não foi encontrada no CSV")
        print(f"colunas disponiveis: {list(dados.columns)}")
        print(f"por favor ajuste a variavel 'TARGET_COLUMN' e tente novamente!")
        #se o target não for encontrado , ira encerrar o script!

        exit()

#etapa 2.2 - divisao entre treino e teste
    print("\n------Dividindo dados em treino e teste...----------")
    
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X,Y,
        test_size=0.2, # 20% dos dados serao para teste
        random_state= 42,  #Garantir a reprotutividade
        stratify=Y         #mater a proporcao de aprovados e reprovados
    )

    print(f"Dados de treino: {len(X_train)} | Dados teste: {len(X_test)}")

    #ETAPA 03: CEIACAO DA PIPELINE DE ML

    print("\n ---------Criando a pipeline de ML-------")
    #scaler -> normalizacao dos dados (colocando tudo na mesma escala)
    #model ->
    pipeline_model = pipeline.Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('model', linear_model.LogisticRegression(random_state=42))
    ])
    #ETAPA 04: Treinamneto e avaliacao dos dados

    print("\n----------------TREINAMENTO DO MODELO......-----------")
#Treina a pipeline com os dados de treino
    pipeline_model.fit(X_train, Y_train)

    print("modelo treinado. Avaliando com os dados de teste....")
    Y_pred = pipeline_model.predict(X_test)

    #AVALIACAO DE DESEMPENHO
    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    print(accuracy)
    report = metrics.classification_report(Y_test, Y_pred)

    print("\n -------------- Relatorio de avaliacao geral...-------------")
    print(f"Acuracia geral: {accuracy * 100:.2f}%")
    print("\nRelatorio de classificacao detalhado:")
    print(report)


    #ETAPA 05: Salvando o modelo
    model_filename = 'modelo_previsao_desempenho.joblib'

    print(f"\nSalvando  pipeline treinado em.. {model_filename}")
    joblib.dump(pipeline_model, model_filename)

    print("Processo concluido com sucesso!")
    print(f"O arquivo '{model_filename}' esta para ser ultilizado!")

else:
    print("O pipeline nao pode continuar pois os dados nao foram carregados!")