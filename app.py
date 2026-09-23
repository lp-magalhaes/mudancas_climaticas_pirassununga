import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração da Página do Simulador
st.set_page_config(page_title="Impacto Climático - Pirassununga", layout="wide")
st.title("🌊 Simulador de Impacto Climático no Abastecimento de Água")
st.subheader("Município: Pirassununga - SP | Modelo de Regressão Multivariada (Betas)")

# 2. Dados Históricos Reais (Consolidados via Google Colab)
dados_base = {
    'Mês': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
    'Chuva_Base': [223.2, 247.2, 251.8, 197.0, 138.2, 135.4, 112.8, 90.8, 169.6, 234.3, 201.3, 220.4],
    'Temp_Base': [24.9, 25.3, 24.5, 23.0, 20.5, 19.6, 20.6, 22.6, 24.7, 25.2, 25.5, 25.5],
    'Fator_F': [1.04, 1.01, 1.02, 0.99, 0.97, 0.95, 0.96, 0.98, 1.00, 1.02, 1.03, 1.05] # Fotoperíodo Pirassununga
}
df_base = pd.DataFrame(dados_base)

# 3. Painel Lateral Interativo (Controle Único por Temperatura)
st.sidebar.header(" Cenários de Mudança Climática")
st.sidebar.markdown("Altere a temperatura para desencadear as respostas automáticas do sistema:")

delta_temp = st.sidebar.slider("Aumento da Temperatura Média (°C)", min_value=0.0, max_value=5.0, value=2.0, step=0.5)

# Gatilho Climático Regional (-7% de pluviosidade por grau de aquecimento)
queda_chuva_por_grau = -0.07 
delta_chuva_percentual = delta_temp * queda_chuva_por_grau * 100

# Parâmetro físico fixo da capacidade do solo
cad = 100.0  

# 4. Processamento Hidrológico do Cenário Simulado
df_sim = df_base.copy()
df_sim['Temp_Sim'] = df_sim['Temp_Base'] + delta_temp
df_sim['Chuva_Sim'] = df_sim['Chuva_Base'] * (1 + delta_chuva_percentual / 100.0)

# Cálculo da Evapotranspiração Potencial (Thornthwaite)
I = np.sum((df_sim['Temp_Sim'] / 5.0) ** 1.514)
a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
df_sim['ETP'] = 16 * ((10 * df_sim['Temp_Sim'] / I) ** a) * df_sim['Fator_F']

# Cálculo do Excedente Hídrico do solo (EXC)
df_sim['Bal'] = df_sim['Chuva_Sim'] - df_sim['ETP']
df_sim['EXC'] = df_sim['Bal'].apply(lambda x: x if x > 0 else 0)

# 5. Modelo Retornado aos Betas Originais da sua Calibração
beta_0 = 3.7026  # Vazão de base do rio (sem chuva)
beta_1 = 0.0103  # Impacto da chuva imediata do mês
beta_2 = 0.0035  # Impacto do lençol freático (chuva acumulada)

vazao_base = []
vazao_sim = []

for i in range(12):
    # Identificação do mês atual e do mês anterior (efeito memória)
    p_atual_b = df_base['Chuva_Base'].iloc[i]
    p_ant_b = df_base['Chuva_Base'].iloc[i-1] if i > 0 else df_base['Chuva_Base'].iloc[-1]
    
    p_atual_s = df_sim['Chuva_Sim'].iloc[i]
    p_ant_s = df_sim['Chuva_Sim'].iloc[i-1] if i > 0 else df_sim['Chuva_Sim'].iloc[-1]
    
    # Aplicação matemática exata da equação multivariada
    # Usamos o max(2.0, ...) como restrição física para o rio nunca secar abaixo da cota crítica operacional
    v_base = max(2.0, beta_0 + (beta_1 * p_atual_b) + (beta_2 * p_ant_b))
    v_sim = max(2.0, beta_0 + (beta_1 * p_atual_s) + (beta_2 * p_ant_s))
    
    vazao_base.append(v_base)
    vazao_sim.append(v_sim)

df_sim['Vazao_Base'] = vazao_base
df_sim['Vazao_Sim'] = vazao_sim

# Cálculo do impacto percentual na vazão do mês mais seco (Agosto)
queda_vazao_ago = ((df_sim['Vazao_Sim'].iloc[7] - df_sim['Vazao_Base'].iloc[7]) / df_sim['Vazao_Base'].iloc[7]) * 100

# 6. Apresentação dos Indicadores na Tela Principal
col1, col2, col3 = st.columns(3)
col1.metric("🌡️ Aquecimento Simulado", f"+{delta_temp} °C")
col2.metric("📉 Queda Automática na Chuva", f"{delta_chuva_percentual:.1f} %")
col3.metric("🚨 Mudança na Vazão Seca (Ago)", f"{queda_vazao_ago:.1f} %")

# 7. Construção Gráfica
st.markdown("### 📊 Resposta da Vazão do Rio via Equação Multivariada de Betas")

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax2 = ax1.twinx()

# Linhas de Vazão (Eixo Esquerdo)
ax1.plot(df_sim['Mês'], df_sim['Vazao_Base'], 'g--', label='Vazão de Referência Histórica (m³/s)', alpha=0.7, linewidth=2)
ax1.plot(df_sim['Mês'], df_sim['Vazao_Sim'], 'g-', label='Vazão Cenário com Betas (m³/s)', linewidth=3)
ax1.set_ylabel('Vazão do Rio (m³/s)', color='g', fontsize=12)
ax1.tick_params(axis='y', labelcolor='g')
ax1.set_ylim(0, max(df_sim['Vazao_Base'].max(), df_sim['Vazao_Sim'].max()) * 1.3)

# Barras de Excedente do solo (Eixo Direito)
ax2.bar(df_sim['Mês'], df_sim['EXC'], color='blue', alpha=0.15, label='Excedente Hídrico (mm)')
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
