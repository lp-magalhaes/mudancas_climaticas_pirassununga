import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

# 1. CONFIGURAÇÃO DA PÁGINA DO SIMULADOR
st.set_page_config(page_title="Impacto Climático - Pirassununga", layout="wide")
st.title("🌊 Simulador de Impacto Climático no Abastecimento de Água")
st.subheader("Município: Pirassununga - SP | Modelagem Hidrológica Integrada via Machine Learning")

# Carregar os modelos treinados (Vazão e Turbidez) com tratamento de erros robusto
try:
    model_rf = joblib.load('modelo_rf_vazao.pkl')
except FileNotFoundError:
    st.error("❌ Erro: O arquivo 'modelo_rf_vazao.pkl' não foi encontrado. Certifique-se de fazer o upload dele no repositório.")
    st.stop()

try:
    model_xgb = joblib.load('best_xgboost_model.pkl')
except FileNotFoundError:
    st.error("❌ Erro: O arquivo 'best_xgboost_model.pkl' não foi encontrado. Certifique-se de fazer o upload dele no repositório.")
    st.stop()

# 2. DADOS HISTÓRICOS REAIS (CONSOLIDADOS VIA GOOGLE COLAB)
dados_base = {
    'Mês_Num': list(range(1, 13)),
    'Mês': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
    'Chuva_Base': [240.8, 163.1, 156.8, 61.8, 53.1, 32.5, 22.6, 23.9, 42.7, 120.7, 169.0, 186.3],
    'Temp_Base': [24.9, 25.3, 24.5, 23.0, 20.5, 19.6, 20.6, 22.6, 24.7, 25.2, 25.5, 25.5],
    'Fator_F': [1.04, 1.01, 1.02, 0.99, 0.97, 0.95, 0.96, 0.98, 1.00, 1.02, 1.03, 1.05]
}
df_base = pd.DataFrame(dados_base)

# 3. PAINEL LATERAL INTERATIVO (CONTROLE ÚNICO POR TEMPERATURA)
st.sidebar.header("🎛️ Cenário de Aquecimento Global")
st.sidebar.markdown("Altere a temperatura para recalcular a pluviosidade (IPCC) e projetar os impactos ambientais:")

delta_temp = st.sidebar.slider("Variação da Temperatura Média (°C)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

# Gatilho Climático Regional (-7% de pluviosidade por grau de aquecimento conforme IPCC)
queda_chuva_por_grau = -0.07 
delta_chuva_percentual = delta_temp * queda_chuva_por_grau * 100

# 4. PROCESSAMENTO HIDROLÓGICO E FORÇANTES DO IPCC
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
    features_base = np.array([[df_base['Mês_Num'].iloc[i], df_base['Chuva_Base'].iloc[i], df_sim['Chuva_Ant_Base'].iloc[i], df_base['Temp_Base'].iloc[i]]], dtype=np.float64)
    features_sim = np.array([[df_sim['Mês_Num'].iloc[i], df_sim['Chuva_Sim'].iloc[i], df_sim['Chuva_Ant_Sim'].iloc[i], df_sim['Temp_Sim'].iloc[i]]], dtype=np.float64)
    
    # Executa a predição e extrai o valor numérico de dentro do array retornado
    v_base = float(model_rf.predict(features_base)[0])
    v_sim = float(model_rf.predict(features_sim)[0])
    
    vazao_base.append(v_base)
    vazao_sim.append(v_sim)

df_sim['Vazao_Base'] = vazao_base
df_sim['Vazao_Sim'] = vazao_sim

# =====================================================================
# 6. MODELAGEM DA MEMÓRIA DE CHUVA ACUMULADA (45 E 60 DIAS)
# =====================================================================
def calcular_acumulado_mensal(series_chuva, dias):
    acumulados = []
    valores = list(series_chuva.values)
    fator_meses = dias / 30.0
    
    for i in range(12):
        if fator_meses == 1.5:
            val = valores[i] + (valores[i-1] * 0.5)
        elif fator_meses == 2.0:
            val = valores[i] + valores[i-1]
        else:
            val = valores[i]
        acumulados.append(val)
    return acumulados

df_sim['chuva_45_dias_base'] = calcular_acumulado_mensal(df_sim['Chuva_Base'], 45)
df_sim['chuva_45_dias_sim'] = calcular_acumulado_mensal(df_sim['Chuva_Sim'], 45)

df_sim['chuva_60_dias_base'] = calcular_acumulado_mensal(df_sim['Chuva_Base'], 60)
df_sim['chuva_60_dias_sim'] = calcular_acumulado_mensal(df_sim['Chuva_Sim'], 60)

# =====================================================================
# 7. EXECUÇÃO DO MODELO XGBOOST (PREDIÇÃO EM MATRIZ COMPLETA DE UMA VEZ)
# =====================================================================
# Criação do DataFrame de entrada completo com os nomes exatos exigidos pelo scikit-learn do XGBoost
recursos_modelo_turbidez = ['Precipitação', 'Vazão', 'Tmed', 'chuva_30_dias_acum', 'chuva_45_dias_acum', 'chuva_60_dias_acum']

df_input_base_completo = pd.DataFrame({
    'Precipitação': df_sim['Chuva_Base'],
    'Vazão': df_sim['Vazao_Base'],
    'Tmed': df_sim['Temp_Base'],
    'chuva_30_dias_acum': df_sim['Chuva_Ant_Base'],
    'chuva_45_dias_acum': df_sim['chuva_45_dias_base'],
    'chuva_60_dias_acum': df_sim['chuva_60_dias_base']
})[recursos_modelo_turbidez].astype(np.float64)

df_input_sim_completo = pd.DataFrame({
    'Precipitação': df_sim['Chuva_Sim'],
    'Vazão': df_sim['Vazao_Sim'],
    'Tmed': df_sim['Temp_Sim'],
    'chuva_30_dias_acum': df_sim['Chuva_Ant_Sim'],
    'chuva_45_dias_acum': df_sim['chuva_45_dias_sim'],
    'chuva_60_dias_acum': df_sim['chuva_60_dias_sim']
})[recursos_modelo_turbidez].astype(np.float64)

# Predição em lote de todos os 12 meses de uma vez (Evita erros de dimensões e tipos de dados)
turb_base_pred = model_xgb.predict(df_input_base_completo)
turb_sim_pred = model_xgb.predict(df_input_sim_completo)

# Adiciona proteção física para garantir que a turbidez não fique abaixo de 0.1
df_sim['Turb_Base'] = [max(0.1, float(t)) for t in turb_base_pred]
df_sim['Turb_Sim'] = [max(0.1, float(t)) for t in turb_sim_pred]

# =====================================================================
# 8. CÁLCULO MÊS A MÊS DO AUMENTO DOS CUSTOS DE TRATAMENTO
# =====================================================================
TURB_MEDIA_HISTORICA = 46.0
FATOR_SENSIBILIDADE_CUSTO = 0.1162

custos_incremento_mensal = []

for i in range(12):
    t_atual = df_sim['Turb_Sim'].iloc[i]
    variacao_percentual_turb = ((t_atual - TURB_MEDIA_HISTORICA) / TURB_MEDIA_HISTORICA) * 100
    
    if variacao_percentual_turb > 0:
        aumento_custo = variacao_percentual_turb * FATOR_SENSIBILIDADE_CUSTO
    else:
        aumento_custo = 0.0
        
    custos_incremento_mensal.append(aumento_custo)

df_sim['Aumento_Custo_Pct'] = custos_incremento_mensal

# =====================================================================
# 9. APRESENTAÇÃO DOS INDICADORES NA TELA PRINCIPAL
# =====================================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡️ Aquecimento Médio", f"{delta_temp} °C")
col2.metric("📉 Alteração Chuva (IPCC)", f"{delta_chuva_percentual:.1f} %")

# Correção no fatiamento para Agosto (Índice 7)
v_sim_ago = df_sim['Vazao_Sim'].iloc[7]
v_base_ago = df_sim['Vazao_Base'].iloc[7]
queda_vazao_ago = ((v_sim_ago - v_base_ago) / v_base_ago) * 100
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
fig1.legend(loc="upper right", bbox_to_anchor=(0.85, 0.88))
ax1.grid(True, alpha=0.2)
st.pyplot(fig1)

# PAINEL DUPLO: Turbidez e Impacto Econômico
st.markdown("### 📈 Diagnóstico de Qualidade da Água e Impacto Financeiro")
col_graph1, col_grid2 = st.columns(2)

with col_graph1:
    st.markdown("#### Turbidez Projetada via XGBoost")
    fig2, ax_t = plt.subplots(figsize=(6, 4))
    ax_t.plot(df_sim['Mês'], df_sim['Turb_Base'], color='#7f7f7f', linestyle=':', marker='o', label='Turbidez Histórica Média')
    ax_t.plot(df_sim['Mês'], df_sim['Turb_Sim'], color='#d62728', linestyle='-', marker='s', linewidth=2.5, label='Turbidez Simulada Cenário')
    ax_t.axhline(y=TURB_MEDIA_HISTORICA, color='black', linestyle='--', alpha=0.5, label='Baseline (46 NTU)')
    ax_t.set_ylabel('Turbidez da Água Bruta (NTU)')
    ax_t.set_xlabel('Mês')
    ax_t.legend(fontsize=9, loc='upper right')
    ax_t.grid(True, alpha=0.2)
    st.pyplot(fig2)

with col_grid2:
    st.markdown("#### Acréscimo nos Custos de Tratamento Químico")
    fig3, ax_c = plt.subplots(figsize=(6, 4))
    ax_c.bar(df_sim['Mês'], df_sim['Aumento_Custo_Pct'], color='#ff7f0e', alpha=0.8, edgecolor='orange', label='Aumento do Custo (%)')
    ax_c.set_ylabel('Aumento Percentual do Custo (%)')
    ax_c.set_xlabel('Mês')
