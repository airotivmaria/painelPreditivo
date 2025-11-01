import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import re
import unicodedata # Para limpeza de acentos nos nomes das colunas
import os # Para garantir que os arquivos sejam encontrados

# --- Definição dos Arquivos ---
files = [
    "DEVEDORES_DIVIDA_ATIVA-2023-12.csv",
    "DEVEDORES_DIVIDA_ATIVA-2024-06.csv",
    "DEVEDORES_DIVIDA_ATIVA-2024-12.csv",
    "DEVEDORES_DIVIDA_ATIVA-2025-06.csv"
]

# --- FUNÇÃO 1: Limpeza dos Nomes das Colunas ---
def clean_col_names(df):
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = unicodedata.normalize('NFKD', col).encode('ascii', 'ignore').decode('utf-8')
        new_col = new_col.upper()
        new_col = new_col.replace(' ', '_')
        new_col = re.sub(r'[^A-Z0-9_]', '', new_col)
        new_cols.append(new_col)
    df.columns = new_cols
    return df

# --- FUNÇÃO 2: Pré-processamento e Engenharia de Features ---
def preprocess_data(file_name):
    """Carrega um CSV, limpa os dados e cria features."""
    
    # --- INÍCIO DA CORREÇÃO ---
    # Tenta ler o arquivo com UTF-8 (padrão moderno)
    try:
        df = pd.read_csv(file_name, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        # Se falhar (arquivos antigos), tenta ler com Latin1 (padrão Windows/legado)
        print(f"Aviso: UTF-8 falhou para {file_name}. Tentando 'latin1'.")
        df = pd.read_csv(file_name, sep=';', encoding='latin1')
    except Exception as e:
        # Outros erros (ex: arquivo não encontrado)
        print(f"Erro ao ler o arquivo {file_name}: {e}")
        return pd.DataFrame()
    # --- FIM DA CORREÇÃO ---

    # Bloco 'try' separado para o processamento das colunas
    try:
        # 1. Limpa os nomes das colunas usando a função acima
        df = clean_col_names(df)
        
        # 2. Limpeza do Saldo Devedor (convertendo para número)
        df['SALDO_DEVEDOR'] = df['SALDO_DEVEDOR_SEM_HONORARIOS'].astype(str).str.replace(r'[^\d,]+', '', regex=True)
        df['SALDO_DEVEDOR'] = df['SALDO_DEVEDOR'].str.replace(',', '.', regex=False)
        df['SALDO_DEVEDOR'] = pd.to_numeric(df['SALDO_DEVEDOR'], errors='coerce')

        # 3. Limpeza das Datas
        df['DATA_GERACAO'] = pd.to_datetime(df['DATA_DE_GERACAO'], format='%d/%m/%Y', errors='coerce')
        df['DATA_INSCRICAO'] = pd.to_datetime(df['DATA_DA_INSCRICAO'], format='%d/%m/%Y %H:%M', errors='coerce')
        if df['DATA_INSCRICAO'].isnull().any():
            df['DATA_INSCRICAO'] = df['DATA_INSCRICAO'].fillna(
                pd.to_datetime(df['DATA_DA_INSCRICAO'], format='%d/%m/%Y', errors='coerce')
            )

        # 4. Criação de um ID Único para cada dívida
        df['ID_UNICO_DIVIDA'] = df['CPFCNPJ_DEVEDOR'].astype(str) + '_' + df['SEQUENCIAL_DO_CREDITO'].astype(str)

        # 5. Engenharia de Feature: Tempo de Inadimplência
        df['TEMPO_INADIMPLENCIA_DIAS'] = (df['DATA_GERACAO'] - df['DATA_INSCRICAO']).dt.days

        # 6. Remoção de dados inválidos
        df.dropna(subset=['SALDO_DEVEDOR', 'DATA_INSCRICAO', 'DATA_GERACAO', 'ID_UNICO_DIVIDA', 'TEMPO_INADIMPLENCIA_DIAS'], inplace=True)
        
        # 7. Seleção das colunas relevantes
        cols_to_keep = [
            'ID_UNICO_DIVIDA', 'SALDO_DEVEDOR', 'TEMPO_INADIMPLENCIA_DIAS',
            'TIPO_DE_PESSOA', 'TIPO_DO_DEVEDOR', 'SITUACAO_DO_CREDITO', 'ORIGEM',
            'CPFCNPJ_DEVEDOR', 'NOME_DO_DEVEDOR', 'SALDO_DEVEDOR_SEM_HONORARIOS', 'DATA_DA_INSCRICAO'
        ]
        
        existing_cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        df = df[existing_cols_to_keep]
        
        return df
    
    except Exception as e:
        # Pega erros durante o processamento (ex: KeyError)
        print(f"Erro ao processar colunas do arquivo {file_name}: {e}")
        return pd.DataFrame()

# --- ETAPA 1: Carregar todos os dados ---
print("Iniciando. Carregando e pré-processando 4 arquivos...")
all_dfs = [preprocess_data(f) for f in files]
print(f"Arquivos carregados. {len(all_dfs[0])}, {len(all_dfs[1])}, {len(all_dfs[2])}, {len(all_dfs[3])} linhas válidas em cada.")

# --- ETAPA 2: Criar o Dataset de Treinamento (com Variável-Alvo) ---
print("Criando o dataset de treinamento (TARGET_PAGO_PROX_SEMESTRE)...")
training_sets = []
for i in range(len(all_dfs) - 1):
    if all_dfs[i].empty or all_dfs[i+1].empty:
        print(f"Aviso: Pulando o par {i}-{i+1} pois um dos dataframes está vazio.")
        continue
    
    df_base = all_dfs[i].copy()
    df_future = all_dfs[i+1]
    
    future_ids = set(df_future['ID_UNICO_DIVIDA'])
    
    df_base['TARGET_PAGO_PROX_SEMESTRE'] = df_base['ID_UNICO_DIVIDA'].apply(lambda x: 0 if x in future_ids else 1)
    
    training_sets.append(df_base)

if not training_sets:
    raise ValueError("Nenhum dado de treinamento foi gerado. Verifique se os arquivos CSV estão corretos e no mesmo diretório.")

df_train_full = pd.concat(training_sets, ignore_index=True)

print("\nDistribuição da variável-alvo (TARGET_PAGO_PROX_SEMESTRE):")
print(df_train_full['TARGET_PAGO_PROX_SEMESTRE'].value_counts(normalize=True))

# --- ETAPA 3: Definição do Modelo (Random Forest) ---
print("\nIniciando o treinamento do modelo Random Forest...")
features = [
    'SALDO_DEVEDOR', 
    'TEMPO_INADIMPLENCIA_DIAS',
    'TIPO_DE_PESSOA', 
    'TIPO_DO_DEVEDOR', 
    'SITUACAO_DO_CREDITO', 
    'ORIGEM'
]
target = 'TARGET_PAGO_PROX_SEMESTRE'

df_train_full = df_train_full.dropna(subset=[target] + features)
X = df_train_full[features]
y = df_train_full[target]

# --- ETAPA 4: Pré-processamento (Pipeline) ---
numeric_features = ['SALDO_DEVEDOR', 'TEMPO_INADIMPLENCIA_DIAS']
categorical_features = ['TIPO_DE_PESSOA', 'TIPO_DO_DEVEDOR', 'SITUACAO_DO_CREDITO', 'ORIGEM']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# --- ETAPA 5: Treinamento do Algoritmo ---
model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model.fit(X_train, y_train)

# --- ETAPA 6: Avaliação do Modelo ---
print("\n--- Relatório de Avaliação do Modelo (em dados de teste) ---")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Classe 0 (Não Pago)', 'Classe 1 (Pago)']))
print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# --- ETAPA 7: Previsão e Priorização (Usando o modelo treinado) ---
print("\nAplicando o modelo treinado nos dados mais recentes (2025-06)...")
df_predict = all_dfs[-1].copy() # Pega o último arquivo (2025-06)

# Esta verificação agora deve passar
if not df_predict.empty and all(col in df_predict.columns for col in features):
    X_new = df_predict[features]

    probabilities = model.predict_proba(X_new)[:, 1]
    df_predict['PROBABILIDADE_RECUPERACAO'] = probabilities

    # --- ETAPA 8: Classificação e Geração de Gráfico ---
    print("Gerando priorização baseada em probabilidade...")

    prob_quantiles = df_predict['PROBABILIDADE_RECUPERACAO'].quantile([0.33, 0.66])
    low_prob_threshold = prob_quantiles.iloc[0]
    medium_prob_threshold = prob_quantiles.iloc[1]

    def classify_priority_proba(score):
        if score >= medium_prob_threshold:
            return 'Alta'
        elif score >= low_prob_threshold:
            return 'Média'
        else:
            return 'Baixa'

    df_predict['PRIORIDADE_PREDITIVA'] = df_predict['PROBABILIDADE_RECUPERACAO'].apply(classify_priority_proba)
    
    priority_order = ['Baixa', 'Média', 'Alta']
    df_predict['PRIORIDADE_PREDITIVA'] = pd.Categorical(df_predict['PRIORIDADE_PREDITIVA'], categories=priority_order, ordered=True)
    priority_counts = df_predict['PRIORIDADE_PREDITIVA'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    priority_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Distribuição da Prioridade Preditiva (Random Forest)', fontsize=14)
    plt.xlabel('Prioridade de Recuperação (Probabilidade)', fontsize=12)
    plt.ylabel('Número de Dívidas', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, count in enumerate(priority_counts):
        plt.text(i, count + 10, str(count), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('prioridade_preditiva_distribuicao.png')
    print("Gráfico 'prioridade_preditiva_distribuicao.png' salvo.")

    # --- ETAPA 9: Salvar o Resultado Final ---
    output_file = 'devedores_PREDICAO_2025-06.csv'
    
    df_predict_output = df_predict[[
        'CPFCNPJ_DEVEDOR', 'NOME_DO_DEVEDOR', 'SALDO_DEVEDOR_SEM_HONORARIOS', 
        'DATA_DA_INSCRICAO', 'TEMPO_INADIMPLENCIA_DIAS', 
        'PROBABILIDADE_RECUPERACAO', 'PRIORIDADE_PREDITIVA'
    ]]
    
    df_predict_output.columns = [
        'CPF/CNPJ DEVEDOR', 'NOME DO DEVEDOR', 'SALDO DEVEDOR SEM HONORÁRIOS',
        'DATA DA INSCRIÇÃO', 'TEMPO_INADIMPLENCIA_DIAS',
        'PROBABILIDADE_RECUPERACAO', 'PRIORIDADE_PREDITIVA'
    ]
    
    df_predict_output = df_predict_output.sort_values(by='PROBABILIDADE_RECUPERACAO', ascending=False)
    
    df_predict_output.to_csv(output_file, index=False, sep=';', encoding='utf-8', decimal=',')

    print(f"\nConcluído! O arquivo com as predições foi salvo como: {output_file}")
    print("\nExemplo das dívidas com MAIOR probabilidade de recuperação:")
    print(df_predict_output.head().to_markdown(index=False, numalign='left', stralign='left'))

else:
    print("Erro: O dataframe de predição (2025-06) está vazio ou faltando colunas.")