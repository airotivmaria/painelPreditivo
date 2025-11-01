import streamlit as st
import pandas as pd
import numpy as np
import os # Para checar se os arquivos existem

# --- Configuração da Página ---
# Define o título da aba do navegador e o layout
st.set_page_config(
    page_title="Painel de Priorização de Dívidas",
    layout="wide"
)

# --- FUNÇÃO DE CARREGAMENTO E CACHE ---
# @st.cache_data garante que o CSV seja lido apenas uma vez
@st.cache_data
def carregar_dados():
    """Carrega e limpa os dados de predição."""
    
    arquivo_predicao = 'devedores_PREDICAO_2025-06.csv'
    arquivo_grafico = 'prioridade_preditiva_distribuicao.png'

    # Verifica se os arquivos necessários existem
    if not os.path.exists(arquivo_predicao):
        st.error(f"Erro: O arquivo '{arquivo_predicao}' não foi encontrado.")
        st.info("Por favor, execute o 'script.py' primeiro para gerar as predições.")
        return None, None

    if not os.path.exists(arquivo_grafico):
        st.warning(f"Aviso: O arquivo de gráfico '{arquivo_grafico}' não foi encontrado.")
    
    # Carrega os dados de predição
    # (decimal=',' é crucial pois salvamos assim no script anterior)
    df = pd.read_csv(arquivo_predicao, sep=';', decimal=',')
    
    # --- Limpeza Adicional para Métricas ---
    # Precisamos converter o 'SALDO DEVEDOR' de texto para número
    # para podermos somar os valores.
    df['SALDO_DEVEDOR_NUMERICO'] = (
        df['SALDO DEVEDOR SEM HONORÁRIOS']
        .astype(str)
        .str.replace(r'[^\d,]+', '', regex=True) # Remove R$ e pontos
        .str.replace(',', '.', regex=False)       # Troca vírgula por ponto
        .pipe(pd.to_numeric, errors='coerce')     # Converte para número
        .fillna(0) # Preenche valores que falharam com 0
    )
    
    return df, arquivo_grafico

# --- Carrega os Dados ---
df_predicoes, img_grafico = carregar_dados()

# --- Título Principal ---
st.title("Painel Preditivo de Recuperação de Crédito")

# Se os dados não foram carregados, interrompe a execução
if df_predicoes is None:
    st.stop()

# --- BARRA LATERAL (FILTROS) ---
st.sidebar.header("Filtros da Cobrança")

# Cria um filtro para Prioridade Preditiva
prioridades = ['Todas'] + list(df_predicoes['PRIORIDADE_PREDITIVA'].unique())
filtro_prioridade = st.sidebar.selectbox(
    'Selecione a Prioridade:',
    options=prioridades,
    index=0 # Começa com 'Todas' selecionado
)

# Filtra o dataframe principal com base na seleção
if filtro_prioridade == 'Todas':
    df_filtrado = df_predicoes
else:
    df_filtrado = df_predicoes[df_predicoes['PRIORIDADE_PREDITIVA'] == filtro_prioridade]

# --- PAINEL PRINCIPAL (MÉTRICAS) ---
st.header(f"Análise de Prioridade: {filtro_prioridade}")
st.write(f"""
Abaixo estão os indicadores-chave para o grupo de devedores selecionado. 
Use o filtro ao lado para focar nas dívidas de prioridade 'Alta' 
(maior probabilidade de recuperação).
""")

# Métricas Dinâmicas (KPIs)
col1, col2, col3 = st.columns(3)

# KPI 1: Total de Dívidas
total_dividas = len(df_filtrado)
col1.metric("Total de Dívidas", f"{total_dividas:,.0f}".replace(",", "."))

# KPI 2: Valor Total a Cobrar
total_a_cobrar = df_filtrado['SALDO_DEVEDOR_NUMERICO'].sum()
col2.metric("Valor Total a Cobrar", f"R$ {total_a_cobrar:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

# KPI 3: Probabilidade Média
prob_media = df_filtrado['PROBABILIDADE_RECUPERACAO'].mean()
col3.metric("Prob. Média de Recuperação", f"{prob_media:.1%}")

# --- Gráfico e Dados Detalhados ---
st.markdown("---") # Linha divisória

# Divide a tela em duas colunas: Gráfico | Tabela
col_graf, col_tabela = st.columns([1, 2]) # Gráfico (1/3), Tabela (2/3)

with col_graf:
    st.subheader("Distribuição das Prioridades")
    if img_grafico:
        st.image(img_grafico, use_column_width=True, caption="Gráfico gerado pelo modelo de Machine Learning.")
    
with col_tabela:
    st.subheader("Devedores Selecionados")
    # Mostra a tabela de dados filtrada
    st.dataframe(
        df_filtrado, 
        use_container_width=True,
        # Oculta colunas que não precisamos ver no painel
        column_config={
            "SALDO_DEVEDOR_NUMERICO": None 
        },
        height=400
    )

st.sidebar.markdown("---")
st.sidebar.info("""
*Sobre este Painel:*
Este dashboard usa um modelo de Machine Learning (Random Forest) 
para prever a probabilidade de cada dívida ser recuperada.

- *script.py*: Treina o modelo e gera o CSV.
- *app.py*: Lê o CSV e gera este painel.
""")