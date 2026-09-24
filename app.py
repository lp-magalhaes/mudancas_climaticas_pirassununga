import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# 1. Configuração da Página do Simulador
st.set_page_config(page_title="Impacto Climático - Pirassununga", layout="wide")
st.title("🌊 Simulador de Impacto Climático no Abastecimento de Água")
st.subheader("Município: Pirassununga - SP | Predição Não Linear via Random Forest Regressor")

# Carregar o modelo Random Forest treinado (Tratamento de erro caso o arquivo não esteja no GitHub)
try:
    model_rf = joblib.load('modelo_rf_vazao.pkl')
except FileNotFoundError:
    st.error("❌ Erro: O arquivo 'modelo_rf_vazao.pkl' não foi encontrado no repositório do GitHub. Certifique-se de fazer o upload dele.")
    st.stop()

# 2. Dados Históricos Reais (Consolidados via Google Colab)
dados_base = {
    'Mês_Num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Mês': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
    'Chuva_Base': [240.8, 163.1, 156.8, 61.8, 53.1, 32.5, 22.6, 23.9, 42.7, 120.7, 169.0, 186.3],
    'Temp_Base': [24.9, 25.3, 24.5, 23.0, 20.5, 19.6, 20.6, 22.6, 24.7, 25.2, 25.5, 25.5],
    'Fator_F': [1.04, 1.01, 1.02, 0.99, 0.97, 0.95, 0.96, 0.98, 1.00, 1.02, 1.03, 1.05]
}
df_base = pd.DataFrame(dados_base)

# 3. Painel Lateral Interativo (Controle Único por Temperatura)
st.sidebar.header("🎛️ Cenário de Aquecimento Global")
st.sidebar.markdown("Altere a temperatura para recalcular a pluviosidade (IPCC) e projetar a vazão por Machine Learning:")

delta_temp = st.sidebar.slider("Aumento da Temperatura Média (°C)", min_value=0.0, max_value=5.0, value=2.0, step=0.5)

# Gatilho Climático Regional (-7% de pluviosidade por grau de aquecimento conforme IPCC)
queda_chuva_por_grau = -0.07 
delta_chuva_percentual = delta_temp * queda_chuva_por_grau * 100

# 4. Processamento Hidrológico e Forçantes do IPCC
df_sim = df_base.copy()
df_sim['Temp_Sim'] = df_sim['Temp_Base'] + delta_temp
df_sim['Chuva_Sim'] = df_sim['Chuva_Base'] * (1 + delta_chuva_percentual / 100.0)

# Criar a memória de Chuva Anterior exigida pelo Random Forest
df_sim['Chuva_Ant_Base'] = df_sim['Chuva_Base'].shift(1).fillna(df_sim['Chuva_Base'].mean())
df_sim['Chuva_Ant_Sim'] = df_sim['Chuva_Sim'].shift(1).fillna(df_sim['Chuva_Sim'].mean())

# Cálculo auxiliar da ETP local (Thornthwaite) para o gráfico de solo
I = np.sum((df_sim['Temp_Sim'] / 5.0) ** 1.514)
a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
df_sim['ETP'] = 16 * ((10 * df_sim['Temp_Sim'] / I) ** a) * df_sim['Fator_F']
df_sim['Bal'] = df_sim['Chuva_Sim'] - df_sim['ETP']
df_sim['EXC'] = df_sim['Bal'].apply(lambda x: x if x > 0 else 0)

# =====================================================================
# 5. EXECUÇÃO DO MODELO RANDOM FOREST EM TEMPO REAL
# =====================================================================
vazao_base = []
vazao_sim = []

for i in range(12):
    # Formato das Features idêntico ao de treinamento: ['Mes', 'Precipitação', 'Chuva_Anterior', 'Tmed']
    features_base = np.array([[df_base['Mês_Num'].iloc[i], df_base['Chuva_Base'].iloc[i], df_sim['Chuva_Ant_Base'].iloc[i], df_base['Temp_Base'].iloc[i]]])
    features_sim = np.array([[df_sim['Mês_Num'].iloc[i], df_sim['Chuva_Sim'].iloc[i], df_sim['Chuva_Ant_Sim'].iloc[i], df_sim['Temp_Sim'].iloc[i]]])
    
    # O modelo faz a inferência não linear com base no pkl carregado
    v_base = float(model_rf.predict(features_base)[0])
    v_sim = float(model_rf.predict(features_sim)[0])
    
    vazao_base.append(v_base)
    vazao_sim.append(v_sim)

df_sim['Vazao_Base'] = vazao_base
df_sim['Vazao_Sim'] = vazao_sim

# Cálculo do impacto percentual na vazão do mês mais seco (Agosto)
queda_vazao_ago = ((df_sim['Vazao_Sim'].iloc[7] - df_sim['Vazao_Base'].iloc[7]) / df_sim['Vazao_Base'].iloc[7]) * 100

# 6. Apresentação dos Indicadores na Tela Principal
col1, col2, col3 = st.columns(3)
col1.metric("🌡️ Aquecimento Simulado", f"+{delta_temp} °C")
col2.metric("📉 Queda na Chuva (IPCC)", f"{delta_chuva_percentual:.1f} %")
col3.metric("🚨 Mudança na Vazão Seca (Ago)", f"{queda_vazao_ago:.1f} %")

# 7. Construção Gráfica
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

# 8. Tabela de Dados Brutos Comparativos
st.markdown("### 📝 Matriz Comparativa de Dados Mensais")
df_exibicao = df_sim[['Mês', 'Chuva_Base', 'Chuva_Sim', 'Temp_Sim', 'EXC', 'Vazao_Base', 'Vazao_Sim']].copy()
df_exibicao.columns = ['Mês', 'Chuva Base (mm)', 'Chuva Reduzida (mm)', 'Temp. Simulada (°C)', 'Excedente Solo (mm)', 'Vazão Base (m³/s)', 'Vazão Simulada (m³/s)']
st.dataframe(df_exibicao.round(2), use_container_width=True)
