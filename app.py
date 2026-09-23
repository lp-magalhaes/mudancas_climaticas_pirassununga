import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração da Página do Simulador
st.set_page_config(page_title="Impacto Climático - Pirassununga", layout="wide")
st.title("🌊 Simulador de Impacto Climático Unificado")
st.subheader("Município: Pirassununga - SP | Modelo de Causa Única (Temperatura)")

# 2. Dados Históricos Reais (Consolidados via Google Colab)
dados_base = {
    'Mês': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
    'Chuva_Base': [223.2, 247.2, 251.8, 197.0, 138.2, 135.4, 112.8, 90.8, 169.6, 234.3, 201.3, 220.4],
    'Temp_Base': [24.9, 25.3, 24.5, 23.0, 20.5, 19.6, 20.6, 22.6, 24.7, 25.2, 25.5, 25.5],
    'Fator_F': [1.04, 1.01, 1.02, 0.99, 0.97, 0.95, 0.96, 0.98, 1.00, 1.02, 1.03, 1.05]
}
df_base = pd.DataFrame(dados_base)

# 3. Painel Lateral - CONTROLE ÚNICO DE TEMPERATURA
st.sidebar.header("🌡️ Cenário de Aquecimento Global")
st.sidebar.markdown("Arraste o slider para ver o efeito cascata em todo o ciclo hídrico:")

delta_temp = st.sidebar.slider("Aumento da Temperatura Média (°C)", min_value=0.0, max_value=5.0, value=2.0, step=0.5)

# 4. GATILHO CLIMÁTICO: Temperatura controlando a Pluviosidade (Sensibilidade de -7% de chuva por °C)
# Exemplo: +2°C vai gerar uma queda automática de -14% na chuva
queda_chuva_por_grau = -0.07 
delta_chuva_percentual = delta_temp * queda_chuva_por_grau * 100

# 5. Processamento Hidrológico Automático
df_sim = df_base.copy()
df_sim['Temp_Sim'] = df_sim['Temp_Base'] + delta_temp
# A nova chuva é calculada automaticamente baseada na temperatura escolhida
df_sim['Chuva_Sim'] = df_sim['Chuva_Base'] * (1 + (delta_chuva_percentual / 100.0))

# Cálculo da Evapotranspiração Potencial (Thornthwaite)
I = np.sum((df_sim['Temp_Sim'] / 5.0) ** 1.514)
a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
df_sim['ETP'] = 16 * ((10 * df_sim['Temp_Sim'] / I) ** a) * df_sim['Fator_F']

# Cálculo do Excedente Hídrico do solo (EXC)
df_sim['Bal'] = df_sim['Chuva_Sim'] - df_sim['ETP']
df_sim['EXC'] = df_sim['Bal'].apply(lambda x: x if x > 0 else 0)

# 6. Modelo de Vazão do Rio Respondendo à Nova Chuva Automatizada
beta_0 = 7.7386  # Vazão de base real obtida no Colab
fator_impacto_chuva = 0.015

vazao_base = []
vazao_sim = []

for i in range(12):
    p_base_mes = df_base['Chuva_Base'].iloc[i]
    p_sim_mes = df_sim['Chuva_Sim'].iloc[i]
    
    v_base = max(1.5, beta_0 + (p_base_mes * fator_impacto_chuva))
    v_sim = max(1.5, beta_0 + (p_sim_mes * fator_impacto_chuva))
    
    vazao_base.append(v_base)
    vazao_sim.append(v_sim)

df_sim['Vazao_Base'] = vazao_base
df_sim['Vazao_Sim'] = vazao_sim

# Cálculo do impacto na vazão crítica (Agosto)
queda_vazao_ago = ((df_sim['Vazao_Sim'].iloc[7] - df_sim['Vazao_Base'].iloc[7]) / df_sim['Vazao_Base'].iloc[7]) * 100

# 7. Apresentação dos Indicadores Dinâmicos na Tela
col1, col2, col3 = st.columns(3)
col1.metric("🌡️ Aquecimento Simulado", f"+{delta_temp} °C")
col2.metric("📉 Queda Automática na Chuva", f"{delta_chuva_percentual:.1f} %")
col3.metric("🚨 Redução na Vazão Seca (Ago)", f"{queda_vazao_ago:.1f} %")

# 8. Gráfico Interativo
st.markdown("### 📊 Efeito Cascata: Vazão do Rio e Excedente do Solo sob Aquecimento")

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax2 = ax1.twinx()

ax1.plot(df_sim['Mês'], df_sim['Vazao_Base'], 'g--', label='Vazão Histórica (m³/s)', alpha=0.7, linewidth=2)
ax1.plot(df_sim['Mês'], df_sim['Vazao_Sim'], 'g-', label='Vazão sob Novo Clima (m³/s)', linewidth=3)
ax1.set_ylabel('Vazão do Rio (m³/s)', color='g', fontsize=12)
ax1.tick_params(axis='y', labelcolor='g')
ax1.set_ylim(0, max(df_sim['Vazao_Base'].max(), df_sim['Vazao_Sim'].max()) * 1.2)

ax2.bar(df_sim['Mês'], df_sim['EXC'], color='blue', alpha=0.15, label='Excedente Hídrico (mm)')
ax2.set_ylabel('Excedente Hídrico (mm)', color='b', fontsize=12)
ax2.tick_params(axis='y', labelcolor='b')

ax1.set_xlabel('Mês', fontsize=12)
fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.88))
ax1.grid(True, alpha=0.3)

st.pyplot(fig)

# 9. Tabela de Dados Comparativos
st.markdown("### 📝 Dados Projetados por Cenário Térmico")
df_exibicao = df_sim[['Mês', 'Chuva_Base', 'Chuva_Sim', 'Temp_Base', 'Temp_Sim', 'EXC', 'Vazao_Base', 'Vazao_Sim']].copy()
df_exibicao.columns = ['Mês', 'Chuva Base (mm)', 'Chuva Reduzida (mm)', 'Temp. Base (°C)', 'Temp. Simulada (°C)', 'Excedente Solo (mm)', 'Vazão Base (m³/s)', 'Vazão Simulada (m³/s)']
st.dataframe(df_exibicao.round(2), use_container_width=True)
