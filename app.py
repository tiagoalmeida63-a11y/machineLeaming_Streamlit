import streamlit as st
import pandas as pd
import joblib
import os

# nome_usuario = st.text_input("Informe seu nome: ", placeholder="Digite aqui... ")


# if st.button(label="Clique aqui:"):
#     st.success("Seja bem-vindo, " + nome_usuario)

#ETAPA 01 - DEFINICAO DAS FEATURES
FEATURES_NAMES =[
    'Nota_P1',
    'Nota_P2',
    'Media_trabalhos',
    "Frequencia",
    'Reprovacoes_Anteriores',
    'Acesso_plataforma_Mes'
]

COLUNAS_HISTORICO = FEATURES_NAMES + ["Previsao_Resultado", 'Prob_Aprovado', 'Prob_Reprovado']

#Criar uma sessao st.session
if 'historico_previsoes' not in st.session_state:
    st.session_state.historico_previsoes = pd.DataFrame(columns=COLUNAS_HISTORICO)

    #ETAPA02 - CARREGAMENTO DO MODELO PARA NOSSO FROT-END
    #st.cache_resource para carregar o modelo apenas uma vez
    #otimizando o desempenho do app
@st.cache_resource
def carregar_modelo(caminho_modelo = "modelo_previsao_desempenho.joblib"):
    """
        Carregar o pipeline de ML treinado (modelo_previsao_desempenho.joblib)
    """

    try:
        if os.path.exists(caminho_modelo):
            modelo = joblib.load(caminho_modelo)
            return modelo
        else:
            st.error(f"Erro Arquivo do modelo '{caminho_modelo}' não foi  encontrado!")
            st.warning(f'Por favor, execute o script "modelo_treinamento.py" para gerar o modelo')
    except Exception as e:
        st.error(f"Error inesperado ao carregar o modelo: {e}")
        return None
        
pipeline_modelo = carregar_modelo()




st.set_page_config(layout='centered', page_title="Previsao de notas")

st.title("🤖 Previsor de desempenho academico")
st.markdown("""
    Essa ferramenta usa inteligencia artificial para prever o status final (Aprovado ou Reprovado) de um aluno com base em seu desempenho parcial
    
    **Preencha os dados do aluno abaixo para obter uma previsao: **
""")

#ETAPA3 FORMULARIO DE ENTRADA

if pipeline_modelo is not None:
#ultilizar um formulario para agrupar as entradas e o botap
    with st.form("Formulario_previsao"):
        st.subheader("Isira as notas e metricas do aluno")

        col1, col2, = st.columns(2)

        with col1:
            nota_p1 = st.slider("Nota da P1 (0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            media_trabalhos = st.slider("Media dos trabalhos (0 a 10)", min_value=0.0, max_value=10.0, value=10.0, step=0.5)
            reprovacoes_ant = st.number_input("Reprovações anteriores", min_value=0, max_value=10, value=5, step=1)
        with col2:
            nota_p2 = st.slider("Nota da P2 (0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            frequencia = st.slider("Frequencia (%)", min_value=0, max_value=100, value=75, step=5)
            acesso_mes = st.number_input("Media de acessos á plataforma (por mês)", min_value=0, max_value=100, value=10, step=1)

        submitted = st.form_submit_button("Realizar previsao")

    if submitted:


        features_name =[
            'Nota_P1',
            'Nota_P2',
            'Media_Trabalhos',
            "Frequencia",
            'Reprovacoes_Anteriores',
            'Acessos_Plataforma_Mes'
        ]

        #Criação de um dataframe a partir dos dados inseridos
        dados_alunos = pd.DataFrame(
            [[nota_p1, nota_p2, media_trabalhos, frequencia, reprovacoes_ant, acesso_mes]],
            columns=features_name
        )

        st.info("Processando dados e realizando a previsão...")

        try:
            #Realizar a previsão ([0] ou [1])
            previsao = pipeline_modelo.predict(dados_alunos)

            #Obter a probabilidade 
            probabilidade = pipeline_modelo.predict_proba(dados_alunos)

            prob_reprovados = probabilidade[0][0]
            prob_aprovados = probabilidade[0][1]
            resultado_texto = "APROVADO" if previsao[0] == 1 else "REPROVADO" 

                #EXIBIR OS RESULTADOS NA TELA

            st.subheader("Resultado da previsão")

            if previsao[0] == 1:
                st.success("Previsão: Aprovado!")
                st.markdown(f"""
                    Com base nos dados fornecidos, o modelo prevê que o aluno tem:
                    **{prob_aprovados*100:.2f}%** de chance de ser **aprovado**

                    *Chance de reprovação: {prob_reprovados*100:.2f}%*
                    """)
            else:
                st.error("Previsão: Reprovado (zona de risco)")
                st.markdown(f"""
                                
                    Com base nos dados fornecidos, o modelo prevê que o aluno tem:
                    **{prob_reprovados*100:.2f}%** de chance de ser **reprovado**

                    *Chance de aprovação: {prob_aprovados*100:.2f}%*
                """)
            nova_linha_dict = {
                'Nota_P1': nota_p1,
                'Nota_P2': nota_p2,
                'Media_Trabalho': media_trabalhos,
                'Frequencia': frequencia,
                'Repreovacoes_Anteriores': reprovacoes_ant,
                'Acessos_Plataforma_Mes': acesso_mes,
                'Previsao_Resultado': resultado_texto,
                'Prob_Aprovado': prob_aprovados,
                'Prob_Reprovados': prob_reprovados

            }

            nova_linha_df = pd.DataFrame([nova_linha_dict], columns=COLUNAS_HISTORICO)

            st.session_state.historico_previsoes = pd.concat(
                [st.session_state.historico_previsoes, nova_linha_df],
                ignore_index=True
            )

        except Exception as e:
            st.error(f"Erro ao fazer a previsao: {e}")
            st.error("Verifique se os nomes das colunas correspondem aos noms das colunas ultilizadas no treino")


    st.subheader("Historico de previsao realizadas na sessão: ")
    if st.session_state.historico_previsoes.empty:
        st.write("Nenhuma previsao foi realizada ainda")
    else:
        st.dataframe(st.session_state.historico_previsoes, use_container_width=True)

        if st.button("Limpar historico"):
            st.session_state.historico_previsoes = pd.DataFrame(columns = COLUNAS_HISTORICO)
        
            st.rerun()
        
else:
    st.warning("O aplicativo não pode fazer previsões porque o modelo nao foi carregado!")
