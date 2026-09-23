import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração da Página do Simulador
st.set_page_config(page_title="Impacto Climático - Pirassununga", layout="wide")
st.title("🌊 Simulador de Impacto Climático no Abastecimento de Água")
st.subheader("Município: Pirassununga - SP | Regressão por Machine Learning (Bootstrap) + IPCC")

# 2. Dados Históricos Reais (Consolidados via Google Colab)
dados_base = {
    'Mês': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
    'Chuva_Base': [221.9, 247.0, 255.5, 199.0, 135.2, 137.2, 110.2, 94.6, 168.8, 229.5, 206.3, 216.9],
    'Temp_Base': [24.9, 25.3, 24.5, 23.0, 20.6, 19.7, 20.6, 22.5, 24.7, 25.2, 25.4, 25.5],
    'Fator_F': [1.04, 1.01, 1.02, 0.99, 0.97, 0.95, 0.96, 0.98, 1.00, 1.02, 1.03, 1.05] # Fotoperíodo Pirassununga
}
df_base = pd.DataFrame(dados_base)

# 3. Painel Lateral Interativo (Controle Único por Temperatura)
st.sidebar.header("🎛️ Cenário de Aquecimento Global")
st.sidebar.markdown("Altere a temperatura para ver a resposta automática do modelo de Machine Learning:")

delta_temp = st.sidebar.slider("Aumento da Temperatura Média (°C)", min_value=0.0, max_value=5.0, value=2.0, step=0.5)

# Gatilho Climático Regional (-7% de pluviosidade por grau de aquecimento conforme IPCC)
queda_chuva_por_grau = -0.07 
delta_chuva_percentual = delta_temp * queda_chuva_por_grau * 100

# Parâmetro físico fixo do solo para o modelo de Thornthwaite
cad = 100.0  

# 4. Processamento Hidrológico do Cenário Simulado
df_sim = df_base.copy()
df_sim['Temp_Sim'] = df_sim['Temp_Base'] + delta_temp
# A nova chuva simulada (usada como entrada no Machine Learning) é gerada aqui:
df_sim['Chuva_Sim'] = df_sim['Chuva_Base'] * (1 + delta_chuva_percentual / 100.0)

# Cálculo da Evapotranspiração Potencial (Thornthwaite) para o solo de Pirassununga
I = np.sum((df_sim['Temp_Sim'] / 5.0) ** 1.514)
a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
df_sim['ETP'] = 16 * ((10 * df_sim['Temp_Sim'] / I) ** a) * df_sim['Fator_F']

# Cálculo do Excedente Hídrico do solo (EXC)
df_sim['Bal'] = df_sim['Chuva_Sim'] - df_sim['ETP']
df_sim['EXC'] = df_sim['Bal'].apply(lambda x: x if x > 0 else 0)

# 5. Modelo de Regressão por Machine Learning (Bootstrap) sintonizado com IPCC
# ATENÇÃO: Substitua os valores abaixo pelos números exatos gerados no seu Google Colab
beta_0_boot = 7.7386  # Intercepto estável médio do Bootstrap
beta_1_boot = 0.0150  # Inclinação estável média do Bootstrap

vazao_base = []
vazao_sim = []

for i in range(12):
    p_base_mes = df_base['Chuva_Base'].iloc[i]
    p_sim_mes = df_sim['Chuva_Sim'].iloc[i] # Lâmina simulada recalculada pelo gatilho do IPCC
    
    # Aplicação da equação de ML obtida no Bootstrap
    # Usamos o max(1.5, ...) como restrição física para o rio nunca zerar vazão na tela
    v_base = max(1.5, beta_0_boot + (beta_1_boot * p_base_mes))
    v_sim = max(1.5, beta_0_boot + (beta_1_boot * p_sim_mes))
    
    vazao_base.append(v_base)
    vazao_sim.append(v_sim)

df_sim['Vazao_Base'] = vazao_base
df_sim['Vazao_Sim'] = vazao_sim

# Cálculo do impacto percentual na vazão do mês mais seco (Agosto)
queda_vazao_ago = ((df_sim['Vazao_Sim'].iloc[7] - df_sim['Vazao_Base'].iloc[7]) / df_sim['Vazao_Base'].iloc[7]) * 100

# 6. Apresentação dos Indicadores na Tela Principal
col1, col2, col3 = st.columns(3)
col1.metric("🌡️ Aquecimento Simulado", f"+{delta_temp} °C")
col2.metric("📉 Queda Automática na Chuva (IPCC)", f"{delta_chuva_percentual:.1f} %")
col3.metric("🚨 Impacto na Vazão Seca (Ago)", f"{queda_vazao_ago:.1f} %")

# 7. Construção Gráfica
st.markdown("### 📊 Resposta da Vazão do Rio via Equação de Machine Learning com Bootstrap")

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax2 = ax1.twinx()

# Linhas de Vazão (Eixo Esquerdo)
ax1.plot(df_sim['Mês'], df_sim['Vazao_Base'], 'g--', label='Vazão Histórica de Referência (m³/s)', alpha=0.7, linewidth=2)
ax1.plot(df_sim['Mês'], df_sim['Vazao_Sim'], 'g-', label='Vazão Projetada pelo Modelo (m³/s)', linewidth=3)
ax1.set_ylabel('Vazão do Rio (m³/s)', color='g', fontsize=12)
ax1.tick_params(axis='y', labelcolor='g')
ax1.set_ylim(0, max(df_sim['Vazao_Base'].max(), df_sim['Vazao_Sim'].max()) * 1.3)

# Barras de Excedente do solo (Eixo Direito)
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
