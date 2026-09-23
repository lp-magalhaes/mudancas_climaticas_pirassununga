import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração da Página do Simulador
st.set_page_config(page_title="Impacto Climático - Pirassununga", layout="wide")
st.title("🌊 Simulador de Impacto Climático no Abastecimento de Água")
st.subheader("Município: Pirassununga - SP (Análise Histórica: 2006 - 2025)")

# 2. Dados Históricos Reais (Consolidados via Google Colab)
# Cole aqui os valores exatos gerados na lista 'Temp_Base' do seu Colab caso queira atualizar as temperaturas
dados_base = {
    'Mês': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
    'Chuva_Base': [223.2, 247.2, 251.8, 197.0, 138.2, 135.4, 112.8, 90.8, 169.6, 234.3, 201.3, 220.4],
    'Temp_Base': [24.9, 25.3, 24.5, 23.0, 20.6, 19.7, 20.6, 22.5, 24.7, 25.2, 25.4, 25.5],
    'Fator_F': [1.04, 1.01, 1.02, 0.99, 0.97, 0.95, 0.96, 0.98, 1.00, 1.02, 1.03, 1.05] # Fotoperíodo Pirassununga
}
df_base = pd.DataFrame(dados_base)

# 3. Painel Lateral Interativo (Sliders de Controle)
st.sidebar.header("🎛️ Cenários de Mudança Climática")
st.sidebar.markdown("Modifique os parâmetros para projetar o impacto no abastecimento:")

delta_temp = st.sidebar.slider("Variação da Temperatura (°C)", min_value=0.0, max_value=5.0, value=2.0, step=0.5)
delta_chuva = st.sidebar.slider("Variação da Pluviosidade (%)", min_value=-50, max_value=20, value=-15, step=5)
cad = st.sidebar.number_input("Capacidade de Água no Solo (CAD em mm)", min_value=50, max_value=200, value=100)

# 4. Processamento Hidrológico do Cenário Simulado
df_sim = df_base.copy()
df_sim['Temp_Sim'] = df_sim['Temp_Base'] + delta_temp
df_sim['Chuva_Sim'] = df_sim['Chuva_Base'] * (1 + delta_chuva / 100.0)

# Cálculo da Evapotranspiração Potencial (Thornthwaite & Mather)
I = np.sum((df_sim['Temp_Sim'] / 5.0) ** 1.514)
a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
df_sim['ETP'] = 16 * ((10 * df_sim['Temp_Sim'] / I) ** a) * df_sim['Fator_F']

# Cálculo simplificado do Excedente Hídrico do solo (EXC)
df_sim['Bal'] = df_sim['Chuva_Sim'] - df_sim['ETP']
df_sim['EXC'] = df_sim['Bal'].apply(lambda x: x if x > 0 else 0)

# 5. Modelo Ajustado de Vazão do Rio (Correção Teórica dos Betas Negativos)
beta_0 = 7.7386  # Vazão de base real observada nos seus dados
# Convertemos os betas estatísticos para coeficientes de impacto positivo proporcional (sensibilidade hídrica)
fator_impacto_chuva = 0.015 

vazao_base = []
vazao_sim = []

for i in range(12):
    p_base_mes = df_base['Chuva_Base'].iloc[i]
    p_sim_mes = df_sim['Chuva_Sim'].iloc[i]
    
    # A vazão responde proporcionalmente à flutuação da chuva em relação à base histórica
    v_base = max(1.5, beta_0 + (p_base_mes * fator_impacto_chuva))
    v_sim = max(1.5, beta_0 + (p_sim_mes * fator_impacto_chuva))
    
    vazao_base.append(v_base)
    vazao_sim.append(v_sim)

df_sim['Vazao_Base'] = vazao_base
df_sim['Vazao_Sim'] = vazao_sim

# Cálculo do impacto percentual na vazão crítica (Agosto)
queda_vazao_ago = ((df_sim['Vazao_Sim'].iloc[7] - df_sim['Vazao_Base'].iloc[7]) / df_sim['Vazao_Base'].iloc[7]) * 100

# 6. Apresentação dos Indicadores na Tela Principal
col1, col2, col3 = st.columns(3)
col1.metric("🌡️ Nova Temp. Média (Verão)", f"{df_sim['Temp_Sim'].max():.1f} °C", f"+{delta_temp} °C")
col2.metric("🌧️ Alteração na Pluviosidade", f"{delta_chuva} %")
col3.metric("📉 Impacto na Vazão Seca (Ago)", f"{queda_vazao_ago:.1f} %")

# 7. Construção Gráfica
st.markdown("### 📊 Comportamento da Vazão do Rio de Captação vs. Excedente Hídrico")

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax2 = ax1.twinx()

# Plotagem das linhas de vazão (Eixo Esquerdo)
ax1.plot(df_sim['Mês'], df_sim['Vazao_Base'], 'g--', label='Vazão Histórica (m³/s)', alpha=0.7, linewidth=2)
ax1.plot(df_sim['Mês'], df_sim['Vazao_Sim'], 'g-', label='Vazão Cenário Projetado (m³/s)', linewidth=3)
ax1.set_ylabel('Vazao do Rio (m³/s)', color='g', fontsize=12)
ax1.tick_params(axis='y', labelcolor='g')
ax1.set_ylim(0, max(df_sim['Vazao_Base'].max(), df_sim['Vazao_Sim'].max()) * 1.2)

# Plotagem do Excedente Hídrico como barras (Eixo Direito)
ax2.bar(df_sim['Mês'], df_sim['EXC'], color='blue', alpha=0.15, label='Excedente Hídrico no Solo (mm)')
ax2.set_ylabel('Excedente Hídrico (mm)', color='b', fontsize=12)
ax2.tick_params(axis='y', labelcolor='b')

ax1.set_xlabel('Mês', fontsize=12)
fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.88))
ax1.grid(True, alpha=0.3)

st.pyplot(fig)

# 8. Tabela de Dados Brutos Comparativos
st.markdown("### 📝 Dados Detalhados por Cenário (Valores Mensais Médios)")
df_exibicao = df_sim[['Mês', 'Chuva_Base', 'Chuva_Sim', 'Temp_Base', 'Temp_Sim', 'EXC', 'Vazao_Base', 'Vazao_Sim']].copy()
df_exibicao.columns = ['Mês', 'Chuva Base (mm)', 'Chuva Simulada (mm)', 'Temp. Base (°C)', 'Temp. Simulada (°C)', 'Excedente Solo (mm)', 'Vazão Base (m³/s)', 'Vazão Simulada (m³/s)']
st.dataframe(df_exibicao.round(2), use_container_width=True)
