import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

# 1. Configuração da Página do Simulador
st.set_page_config(page_title="Impacto Climático - Pirassununga", layout="wide")
st.title("🌊 Simulador de Impacto Climático no Abastecimento de Água")
st.subheader("Município: Pirassununga - SP | Modelagem Hidrológica Integrada via Machine Learning")

# Carregar os modelos treinados (Vazão e Turbidez) com tratamento de erros
try:
    model_rf = joblib.load('modelo_rf_vazao.pkl')
except FileNotFoundError:
    st.error("❌ Erro: O arquivo 'modelo_rf_vazao.pkl' não foi encontrado. Certifique-se de fazer o upload dele.")
    st.stop()

try:
    # Carrega o modelo campeão do XGBoost para Turbidez
    model_xgb = xgb.XGBRegressor()
    model_xgb.load_model('best_xgboost_model.json')
except Exception:
    try:
        # Alternativa caso você tenha salvo o XGBoost em formato .pkl
        model_xgb = joblib.load('best_xgboost_model.pkl')
    except Exception:
        st.error("❌ Erro: O arquivo do modelo de Turbidez ('best_xgboost_model.json' ou '.pkl') não foi encontrado.")
        st.stop()

# 2. Dados Históricos Reais (Consolidados via Google Colab)
dados_base = {
    'Mês_Num':,
    'Mês': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
    'Chuva_Base': [240.8, 163.1, 156.8, 61.8, 53.1, 32.5, 22.6, 23.9, 42.7, 120.7, 169.0, 186.3],
    'Temp_Base': [24.9, 25.3, 24.5, 23.0, 20.5, 19.6, 20.6, 22.6, 24.7, 25.2, 25.5, 25.5],
    'Fator_F': [1.04, 1.01, 1.02, 0.99, 0.97, 0.95, 0.96, 0.98, 1.00, 1.02, 1.03, 1.05]
}
df_base = pd.DataFrame(dados_base)

# 3. Painel Lateral Interativo (Controle Único por Temperatura)
st.sidebar.header("🎛️ Cenário de Aquecimento Global")
st.sidebar.markdown("Altere a temperatura para recalcular a pluviosidade (IPCC) e projetar os impactos ambientais:")

delta_temp = st.sidebar.slider("Variação da Temperatura Média (°C)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

# Gatilho Climático Regional (-7% de pluviosidade por grau de aquecimento conforme IPCC)
queda_chuva_por_grau = -0.07 
delta_chuva_percentual = delta_temp * queda_chuva_por_grau * 100

# 4. Processamento Hidrológico e Forçantes do IPCC
df_sim = df_base.copy()
df_sim['Temp_Sim'] = df_sim['Temp_Base'] + delta_temp
df_sim['Chuva_Sim'] = df_sim['Chuva_Base'] * (1 + delta_chuva_percentual / 100.0)

# Criar a memória de Chuva Anterior exigida pelo Random Forest de Vazão
df_sim['Chuva_Ant_Base'] = df_sim['Chuva_Base'].shift(1).fillna(df_sim['Chuva_Base'].mean())
df_sim['Chuva_Ant_Sim'] = df_sim['Chuva_Sim'].shift(1).fillna(df_sim['Chuva_Sim'].mean())

# Cálculo auxiliar da ETP local (Thornthwaite) para o gráfico de solo
I = np.sum((df_sim['Temp_Sim'] / 5.0) ** 1.514)
a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
df_sim['ETP'] = 16 * ((10 * df_sim['Temp_Sim'] / I) ** a) * df_sim['Fator_F']
df_sim['Bal'] = df_sim['Chuva_Sim'] - df_sim['ETP']
df_sim['EXC'] = df_sim['Bal'].apply(lambda x: x if x > 0 else 0)

# =====================================================================
# 5. EXECUÇÃO DO MODELO RANDOM FOREST (VAZÃO EM TEMPO REAL)
# =====================================================================
vazao_base = []
vazao_sim = []

for i in range(12):
    features_base = np.array([[df_base['Mês_Num'].iloc[i], df_base['Chuva_Base'].iloc[i], df_sim['Chuva_Ant_Base'].iloc[i], df_base['Temp_Base'].iloc[i]]])
    features_sim = np.array([[df_sim['Mês_Num'].iloc[i], df_sim['Chuva_Sim'].iloc[i], df_sim['Chuva_Ant_Sim'].iloc[i], df_sim['Temp_Sim'].iloc[i]]])
    
    v_base = float(model_rf.predict(features_base)[0])
    v_sim = float(model_rf.predict(features_sim)[0])
    
    vazao_base.append(v_base)
    vazao_sim.append(v_sim)

df_sim['Vazao_Base'] = vazao_base
df_sim['Vazao_Sim'] = vazao_sim

# =====================================================================
# 6. MODELAGEM DA MEMÓRIA DE CHUVA ACUMULADA (45 E 60 DIAS)
# =====================================================================
# Função cíclica para calcular acumulados considerando a transição de Dezembro para Janeiro
def calcular_acumulado_mensal(series_chuva, dias):
    acumulados = []
    valores = list(series_chuva.values)
    # Como a planilha é mensal, aproximamos 45 dias como 1.5 meses e 60 dias como 2 meses
    fator_meses = dias / 30.0
    
    for i in range(12):
        if fator_meses == 1.5:
            # 45 dias = Chuva do mês atual + metade do mês anterior
            val = valores[i] + (valores[i-1] * 0.5)
        elif fator_meses == 2.0:
            # 60 dias = Chuva do mês atual + mês anterior completo
            val = valores[i] + valores[i-1]
        else:
            val = valores[i]
        acumulados.append(val)
    return acumulados

# Aplica o cálculo de memória tanto na base quanto no cenário simulado
df_sim['chuva_45_dias_base'] = calcular_acumulado_mensal(df_sim['Chuva_Base'], 45)
df_sim['chuva_45_dias_sim'] = calcular_acumulado_mensal(df_sim['Chuva_Sim'], 45)

df_sim['chuva_60_dias_base'] = calcular_acumulado_mensal(df_sim['Chuva_Base'], 60)
df_sim['chuva_60_dias_sim'] = calcular_acumulado_mensal(df_sim['Chuva_Sim'], 60)

# =====================================================================
# 7. EXECUÇÃO DO MODELO XGBOOST (TURBIDEZ INTEGRADA)
# =====================================================================
# Mapeamento dinâmico automático com base nas variáveis selecionadas por Spearman no seu modelo
# O modelo XGBoost espera uma matriz com as colunas na ordem em que foi treinado. 
# Ajuste a ordem da lista abaixo se o seu ranking de Spearman tiver sido diferente:
recursos_modelo_turbidez = ['Precipitação', 'Vazão', 'Tmed', 'chuva_30_dias_acum', 'chuva_45_dias_acum', 'chuva_60_dias_acum']

turb_base = []
turb_sim = []

for i in range(12):
    # Dicionários temporários para mapear os dados de entrada na escala real
    map_base = {
        'Precipitação': df_sim['Chuva_Base'].iloc[i],
        'Vazão': df_sim['Vazao_Base'].iloc[i],
        'Tmed': df_sim['Temp_Base'].iloc[i],
        'chuva_30_dias_acum': df_sim['Chuva_Ant_Base'].iloc[i], # Chuva anterior de 30 dias
        'chuva_45_dias_acum': df_sim['chuva_45_dias_base'].iloc[i],
        'chuva_60_dias_acum': df_sim['chuva_60_dias_base'].iloc[i]
    }
    
    map_sim = {
        'Precipitação': df_sim['Chuva_Sim'].iloc[i],
        'Vazão': df_sim['Vazao_Sim'].iloc[i],
        'Tmed': df_sim['Temp_Sim'].iloc[i],
        'chuva_30_dias_acum': df_sim['Chuva_Ant_Sim'].iloc[i],
        'chuva_45_dias_acum': df_sim['chuva_45_dias_sim'].iloc[i],
        'chuva_60_dias_acum': df_sim['chuva_60_dias_sim'].iloc[i]
    }
    
    # Filtra apenas os preditores que seu modelo XGBoost final realmente utiliza
    features_base_turb = np.array([[map_base[col] for col in recursos_modelo_turbidez if col in map_base]])
    features_sim_turb = np.array([[map_sim[col] for col in recursos_modelo_turbidez if col in map_sim]])
    
    # Inferência preditiva da turbidez
    t_base = float(model_xgb.predict(features_base_turb)[0])
    t_sim = float(model_xgb.predict(features_sim_turb)[0])
    
    turb_base.append(max(0.1, t_base)) # Impede valores físicos impossíveis menores que zero
    turb_sim.append(max(0.1, t_sim))

df_sim['Turb_Base'] = turb_base
df_sim['Turb_Sim'] = turb_sim

# =====================================================================
# 8. CÁLCULO MÊS A MÊS DO AUMENTO DOS CUSTOS DE TRATAMENTO
# =====================================================================
# Parâmetros de engenharia sanitária definidos
TURB_MEDIA_HISTORICA = 46.0
FATOR_SENSIBILIDADE_CUSTO = 0.1162

custos_incremento_mensal = []

for i in range(12):
    t_atual = df_sim['Turb_Sim'].iloc[i]
    # Calcula a variação percentual em relação à média de referência de 46 NTU
    variacao_percentual_turb = ((t_atual - TURB_MEDIA_HISTORICA) / TURB_MEDIA_HISTORICA) * 100
    
    # Relação: Se aumentar 1% a turbidez, o custo aumenta em 0.1162%
    if variacao_percentual_turb > 0:
        aumento_custo = variacao_percentual_turb * FATOR_SENSIBILIDADE_CUSTO
    else:
        # Se a turbidez estiver abaixo da média, não há acréscimo no custo basal
        aumento_custo = 0.0
        
    custos_incremento_mensal.append(aumento_custo)

df_sim['Aumento_Custo_Pct'] = custos_incremento_mensal

# =====================================================================
# 9. APRESENTAÇÃO DOS INDICADORES NA TELA PRINCIPAL
# =====================================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡️ Aquecimento Média", f"{delta_temp} °C")
col2.metric("📉 Alteração Chuva (IPCC)", f"{delta_chuva_percentual:.1f} %")

queda_vazao_ago = ((df_sim['Vazao_Sim'].iloc[7] - df_sim['Vazao_Base'].iloc[7]) / df_sim['Vazao_Base'].iloc[7]) * 100
col3.metric("🚨 Vazão Fina (Agosto)", f"{queda_vazao_ago:.1f} %")

pico_custo_mensal = max(custos_incremento_mensal)
col4.metric("💰 Pico de Custo Químico", f"+{pico_custo_mensal:.2f} %")

# =====================================================================
# 10. CONSTRUÇÃO GRÁFICA DO PAINEL DE SIMULAÇÃO
# =====================================================================
st.markdown("### 📊 Comportamento Sazonal da Vazão e Excedente Hídrico")
fig1, ax1 = plt.subplots(figsize=(11, 3.8))
ax2 = ax1.twinx()
ax1.plot(df_sim['Mês'], df_sim['Vazao_Base'], 'g--', label='Vazão Histórica (m³/s)', alpha=0.7, linewidth=2)
ax1.plot(df_sim['Mês'], df_sim['Vazao_Sim'], 'g-', label='Vazão sob Cenário Climático (m³/s)', linewidth=3)
ax1.set_ylabel('Vazão do Rio (m³/s)', color='g', fontsize=11)
ax1.tick_params(axis='y', labelcolor='g')
ax2.bar(df_sim['Mês'], df_sim['EXC'], color='blue', alpha=0.12, label='Excedente Solo (mm)')
ax2.set_ylabel('Excedente Hídrico (mm)', color='b', fontsize=11)
ax2.tick_params(axis='y', labelcolor='b')
ax1.set_xlabel('Mês')
