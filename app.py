import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import statistics

# 1. CONFIGURAÇÃO DA PÁGINA DO SIMULADOR
st.set_page_config(page_title="Impacto Climático - Pirassununga", layout="wide")
st.title("🌊 Simulador de Impacto Climático no Abastecimento de Água")
st.subheader("Município: Pirassununga - SP | Modelagem Hidrológica Integrada via Machine Learning")

# Carregar os modelos treinados com tratamento de erros simples
try:
    model_rf = joblib.load('modelo_rf_vazao.pkl')
except FileNotFoundError:
    st.error("❌ Erro: O arquivo 'modelo_rf_vazao.pkl' não foi encontrado no repositório.")
    st.stop()

try:
    # Substituído o XGBoost pelo Random Forest de Turbidez
    model_rf_turb = joblib.load('modelo_rf_turbidez.pkl')
except FileNotFoundError:
    st.error("❌ Erro: O arquivo 'modelo_rf_turbidez.pkl' não foi encontrado no repositório.")
    st.stop()

# 2. DADOS HISTÓRICOS REAIS E VALORES DE TURBIDEZ BASE INTERPOLADOS MÊS A MÊS
dados_base = {
    'Mês_Num': list(range(1, 13)),
    'Mês': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
    'Chuva_Base': [240.8, 163.1, 156.8, 61.8, 53.1, 32.5, 22.6, 23.9, 42.7, 120.7, 169.0, 186.3],
    'Temp_Base': [24.9, 25.3, 24.5, 23.0, 20.5, 19.6, 20.6, 22.6, 24.7, 25.2, 25.5, 25.5],
    'Turb_Base': [63.43, 63.43, 42.88, 22.33, 17.33, 12.86, 11.04, 9.21, 44.90, 80.58, 34.46, 67.75],
    'Fator_F': [1.04, 1.01, 1.02, 0.99, 0.97, 0.95, 0.96, 0.98, 1.00, 1.02, 1.03, 1.05]
}
df_base = pd.DataFrame(dados_base)

# 3. PAINEL LATERAL INTERATIVO (CONTROLE ÚNICO POR TEMPERATURA)
st.sidebar.header("🎛️ Cenário de Aquecimento Global")
delta_temp = st.sidebar.slider("Variação da Temperatura Média (°C)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

# Gatilho Climático Regional (-7% de pluviosidade por grau de aquecimento)
queda_chuva_por_grau = -0.07 
delta_chuva_percentual = delta_temp * queda_chuva_por_grau * 100

# 4. PROCESSAMENTO HIDROLÓGICO DO CENÁRIO
df_sim = df_base.copy()
df_sim['Temp_Sim'] = df_sim['Temp_Base'] + delta_temp
df_sim['Chuva_Sim'] = df_sim['Chuva_Base'] * (1 + delta_chuva_percentual / 100.0)

# Memória de Chuva Anterior exigida pelo Random Forest de Vazão
df_sim['Chuva_Ant_Base'] = df_sim['Chuva_Base'].shift(1).fillna(df_sim['Chuva_Base'].mean())
df_sim['Chuva_Ant_Sim'] = df_sim['Chuva_Sim'].shift(1).fillna(df_sim['Chuva_Sim'].mean())

# Cálculo auxiliar do Balanço de Solo (Gráfico de Excedente)
I = np.sum((df_sim['Temp_Sim'] / 5.0) ** 1.514)
a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
df_sim['ETP'] = 16 * ((10 * df_sim['Temp_Sim'] / I) ** a) * df_sim['Fator_F']
df_sim['Bal'] = df_sim['Chuva_Sim'] - df_sim['ETP']
df_sim['EXC'] = df_sim['Bal'].apply(lambda x: x if x > 0 else 0)

# =====================================================================
# 5. EXECUÇÃO DO MODELO RANDOM FOREST (PREDIÇÃO DA VAZÃO)
# =====================================================================
vazao_base = []
vazao_sim = []

for i in range(12):
    features_base = np.array([[df_base['Mês_Num'].iloc[i], df_base['Chuva_Base'].iloc[i], df_sim['Chuva_Ant_Base'].iloc[i], df_base['Temp_Base'].iloc[i]]], dtype=np.float64)
    features_sim = np.array([[df_sim['Mês_Num'].iloc[i], df_sim['Chuva_Sim'].iloc[i], df_sim['Chuva_Ant_Sim'].iloc[i], df_sim['Temp_Sim'].iloc[i]]], dtype=np.float64)
    
    # CORREÇÃO DEFINITIVA: Desempacotamento de array adicionando explicitamente o [0]
    vazao_base.append(float(model_rf.predict(features_base)[0]))
    vazao_sim.append(float(model_rf.predict(features_sim)[0]))

df_sim['Vazao_Base'] = vazao_base
df_sim['Vazao_Sim'] = vazao_sim

# =====================================================================
# 6. CÁLCULO DIRETO DAS CHUVAS ACUMULADAS (45 E 60 DIAS)
# =====================================================================
chuva_45_base, chuva_45_sim = [], []
chuva_60_base, chuva_60_sim = [], []

c_base = list(df_sim['Chuva_Base'].values)
c_sim = list(df_sim['Chuva_Sim'].values)

for i in range(12):
   # Índices com tratamento cíclico para o ano (0 a 11)
    ant = (i - 1) % 12       # Se i=0 (Jan), ant=11 (Dez)
    retr = (i - 2) % 12      # Se i=0 (Jan), retr=10 (Nov) | Se i=1 (Fev), retr=11 (Dez)
    
    # Regra de 45 dias: Mês atual + Mês anterior + metade do mês retrasado
    chuva_45_base.append(c_base[ant] + (c_base[retr] * 0.5))
    chuva_45_sim.append(c_sim[ant] + (c_sim[retr] * 0.5))
    
    # Regra de 60 dias: Mês atual + Soma dos dois meses anteriores
    chuva_60_base.append(c_base[ant] + c_base[retr])
    chuva_60_sim.append(c_sim[ant] + c_sim[retr])

df_sim['Chuva_45_Base'] = chuva_45_base
df_sim['Chuva_45_Sim'] = chuva_45_sim
df_sim['Chuva_60_Base'] = chuva_60_base
df_sim['Chuva_60_Sim'] = chuva_60_sim

# =====================================================================
# 7. EXECUÇÃO DO MODELO RANDOM FOREST (PREDIÇÃO DA TURBIDEZ)
# =====================================================================
turb_sim = []
for i in range(12):
    # REGRA REQUERIDA: Força o cenário a assumir a base caso a temperatura esteja zerada
    if delta_temp == 0.0:
        turb_sim.append(df_sim['Turb_Base'].iloc[i])
    else:
        # CORREÇÃO: Utilizando as mesmas 4 variáveis que o modelo de vazão espera
        features_sim_turb = np.array([[
            df_sim['Mês_Num'].iloc[i], 
            df_sim['Chuva_Sim'].iloc[i], 
            df_sim['Chuva_Ant_Sim'].iloc[i], 
            df_sim['Temp_Sim'].iloc[i]
        ]], dtype=np.float64)
        
        # Desempacotamento de array adicionando o [0] para evitar erros de tipo
        t_sim = float(model_rf_turb.predict(features_sim_turb)[0])
        turb_sim.append(max(0.1, t_sim))

df_sim['Turb_Sim'] = turb_sim


# =====================================================================
# 8. CÁLCULO DO CUSTO DO TRATAMENTO DE ÁGUA MÊS A MÊS
# =====================================================================
FATOR_SENSIBILIDADE_CUSTO = 0.1162
TURBIDEZ_MEDIA = 46

custos_incremento_mensal = []
for i in range(12):
    t_atual = df_sim['Turb_Sim'].iloc[i]
    
    # Nova lógica: Subtração da turbidez atual pela média (46)
    diferenca_turb = t_atual - TURBIDEZ_MEDIA
    
    # O custo aumenta se a turbidez atual for maior que a média
    if diferenca_turb > 0:
        aumento_custo = diferenca_turb * FATOR_SENSIBILIDADE_CUSTO
    else:
        aumento_custo = 0.0
        
    custos_incremento_mensal.append(aumento_custo)

# Atribuindo os novos valores de custo ao DataFrame
df_sim['Aumento_Custo_Pct'] = custos_incremento_mensal

# =====================================================================
# 9. EXIBIÇÃO DOS INDICADORES DE TOPO
# =====================================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡️ Aquecimento Médio", f"{delta_temp} °C")
col2.metric("📉 Alteração Chuva (IPCC)", f"{delta_chuva_percentual:.1f} %")

v_sim_ago = float(df_sim['Vazao_Sim'].iloc[10])
v_base_ago = float(df_sim['Vazao_Base'].iloc[10])
queda_vazao_ago = ((v_sim_ago - v_base_ago) / v_base_ago) * 100
col3.metric("🚨 Variação da menor vazão", f"{queda_vazao_ago:.1f} %")

pico_custo_mensal = statistics.mean(custos_incremento_mensal)
col4.metric("💰 Variação média custo", f"+{pico_custo_mensal:.2f} %")

# =====================================================================
# 10. CONSTRUÇÃO DOS GRÁFICOS (VAZÃO, TURBIDEZ E CUSTO COM TRAVA LÓGICA)
# =====================================================================
st.markdown("### 📊 Comportamento Sazonal da Vazão via Machine Learning (Random Forest)")

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax2 = ax1.twinx()

ax1.plot(df_sim['Mês'], df_sim['Vazao_Base'], 'g--', label='Vazão Histórica Estimada (m³/s)', alpha=0.7, linewidth=2)
ax1.plot(df_sim['Mês'], df_sim['Vazao_Sim'], 'g-', label='Vazão sob Cenário Climático RF (m³/s)', linewidth=3)
ax1.set_ylabel('Vazão do Rio (m³/s)', color='g', fontsize=12)
ax1.tick_params(axis='y', labelcolor='g')
ax1.set_ylim(0, max(df_sim['Vazao_Base'].max(), df_sim['Vazao_Sim'].max()) * 1.3)

ax2.bar(df_sim['Mês'], df_sim['EXC'], color='blue', alpha=0.15, label='Excedente Hídrico Solo (mm)')
ax2.set_ylabel('Excedente Hídrico (mm)', color='b', fontsize=12)
ax2.tick_params(axis='y', labelcolor='b')

ax1.set_xlabel('Mês', fontsize=12)
fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.88))
ax1.grid(True, alpha=0.3)
st.pyplot(fig)

st.markdown("### 📈 Diagnóstico de Qualidade da Água e Impacto Financeiro")
col_graph1, col_grid2 = st.columns(2)

with col_graph1:
    # Ajustado título para referenciar o modelo Random Forest
    st.markdown("#### Valor da Turbidez do Rio via Random Forest")
    fig2, ax_t = plt.subplots(figsize=(6, 4))
    
    ax_t.axhline(y=46.0, color='black', linestyle='--', alpha=0.5, label='Referência Média (46 NTU)')
    
    # A linha vermelha desaparece da tela caso o controle de aquecimento esteja em zero
    if delta_temp != 0.0:
        ax_t.plot(df_sim['Mês'], df_sim['Turb_Sim'], color='#d62728', linestyle='-', marker='s', linewidth=2.5, label='Turbidez Simulada Cenário')
        
    ax_t.set_ylabel('Turbidez Bruta (NTU)')
    ax_t.set_xlabel('Mês')
    ax_t.legend(fontsize=9, loc='upper right')
    ax_t.grid(True, alpha=0.2)
    st.pyplot(fig2)

with col_grid2:
    st.markdown("#### Custo do Tratamento Químico Mês a Mês")
    fig3, ax_c = plt.subplots(figsize=(6, 4))
    ax_c.bar(df_sim['Mês'], df_sim['Aumento_Custo_Pct'], color='#ff7f0e', alpha=0.8, edgecolor='orange', label='Aumento do Custo (%)')
    ax_c.set_ylabel('Aumento no Custo de Tratamento (%)')
    ax_c.set_xlabel('Mês')
    ax_c.legend(fontsize=9, loc='upper right')
    ax_c.grid(True, alpha=0.2)
    st.pyplot(fig3)

# 11. TABELA DE MATRIZ DE DADOS COMPLETA
st.markdown("### 📝 Matriz de Variáveis Hidrológicas e Econômicas")
df_exibicao = df_sim[['Mês', 'Chuva_Sim', 'Temp_Sim', 'Vazao_Sim', 'Turb_Base', 'Turb_Sim', 'Aumento_Custo_Pct']].copy()
