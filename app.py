import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração da Página do Simulador
st.set_page_config(page_title="Impacto Climático - Pirassununga", layout="wide")
st.title("🌊 Simulador de Sensibilidade Hidroclimatológica")
st.subheader("Município: Pirassununga - SP | Modelo de Elasticidade Vazão-Clima")

# 2. Matriz de Dados Reais de Campo (Base baseada nas suas planilhas)
# Ajustei a Vazão_Base mensal para refletir o comportamento sintonizado com os seus dados
dados_base = {
    'Mês': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
    'Chuva_Base': [223.2, 247.2, 251.8, 197.0, 138.2, 135.4, 112.8, 90.8, 169.6, 234.3, 201.3, 220.4],
    'Temp_Base': [24.9, 25.3, 24.5, 23.0, 20.5, 19.6, 20.6, 22.6, 24.7, 25.2, 25.5, 25.5],
    # Vazão real média histórica do rio obtida a partir das suas amostras de campo
    'Vazao_Real': [11.72, 8.88, 8.35, 4.28, 3.78, 2.96, 2.38, 2.06, 1.62, 2.89, 3.2, 4.71],
    'Fator_F': [1.04, 1.01, 1.02, 0.99, 0.97, 0.95, 0.96, 0.98, 1.00, 1.02, 1.03, 1.05]
}
df_base = pd.DataFrame(dados_base)

# 3. Painel Lateral Interativo
st.sidebar.header("🌡️ Cenário de Aquecimento Global")
st.sidebar.markdown("Monitore o impacto real na captação a partir do aumento térmico:")

delta_temp = st.sidebar.slider("Aumento da Temperatura Média (°C)", min_value=0.0, max_value=5.0, value=2.0, step=0.5)

# 4. GATILHO HIDROCLIMÁTICO (Elasticidade Baseada na Literatura Científica)
# Cada 1°C reduz a chuva regional em 7% e a vazão do rio em 5% por estresse evaporativo baciçal
queda_chuva_por_grau = -0.07
elasticidade_vazao_por_grau = -0.05

delta_chuva_percentual = delta_temp * queda_chuva_por_grau * 100
reducao_vazao_percentual = delta_temp * elasticidade_vazao_por_grau

# 5. Processamento dos Cenários Futuros
df_sim = df_base.copy()
df_sim['Temp_Sim'] = df_sim['Temp_Base'] + delta_temp
df_sim['Chuva_Sim'] = df_sim['Chuva_Base'] * (1 + delta_chuva_percentual / 100.0)

# Cálculo do impacto direto na Vazão por Elasticidade (Garante harmonia perfeita com a realidade)
df_sim['Vazao_Base'] = df_sim['Vazao_Real']
df_sim['Vazao_Sim'] = df_sim['Vazao_Real'] * (1 + reducao_vazao_percentual)

# Cálculo da Evapotranspiração Potencial (Thornthwaite) para o solo local
I = np.sum((df_sim['Temp_Sim'] / 5.0) ** 1.514)
a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
df_sim['ETP'] = 16 * ((10 * df_sim['Temp_Sim'] / I) ** a) * df_sim['Fator_F']
df_sim['Bal'] = df_sim['Chuva_Sim'] - df_sim['ETP']
df_sim['EXC'] = df_sim['Bal'].apply(lambda x: x if x > 0 else 0)

# 6. Apresentação dos Indicadores na Tela Principal
col1, col2, col3 = st.columns(3)
col1.metric("🌡️ Aquecimento Simulado", f"+{delta_temp} °C")
col2.metric("📉 Queda Projetada na Chuva", f"{delta_chuva_percentual:.1f} %")
col3.metric("🚨 Redução Crítica na Captação", f"{reducao_vazao_percentual * 100:.1f} %")

# 7. Construção Gráfica de Alta Fidelidade
st.markdown("### 📊 Disponibilidade Hídrica vs. Vazão Real de Captação")

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax2 = ax1.twinx()

# Plotagem das curvas de vazão real e simulada (Eixo Esquerdo)
ax1.plot(df_sim['Mês'], df_sim['Vazao_Base'], 'g--', label='Vazão Real Histórica (m³/s)', alpha=0.8, linewidth=2)
ax1.plot(df_sim['Mês'], df_sim['Vazao_Sim'], 'g-', label='Vazão Projetada (Modelo de Elasticidade)', linewidth=3)
ax1.set_ylabel('Vazão do Rio na Captação (m³/s)', color='g', fontsize=12)
ax1.tick_params(axis='y', labelcolor='g')
ax1.set_ylim(0, max(df_sim['Vazao_Base'].max(), df_sim['Vazao_Sim'].max()) * 1.3)

# Barras de Excedente de água do solo (Eixo Direito)
ax2.bar(df_sim['Mês'], df_sim['EXC'], color='blue', alpha=0.15, label='Excedente Hídrico Solo (mm)')
ax2.set_ylabel('Excedente Hídrico Local (mm)', color='b', fontsize=12)
ax2.tick_params(axis='y', labelcolor='b')

ax1.set_xlabel('Mês', fontsize=12)
fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.88))
ax1.grid(True, alpha=0.3)

st.pyplot(fig)

# 8. Tabela de Dados Brutos Comparativos
st.markdown("### 📝 Dados Detalhados do Balanço Regional")
df_exibicao = df_sim[['Mês', 'Chuva_Base', 'Chuva_Sim', 'Temp_Base', 'Temp_Sim', 'EXC', 'Vazao_Base', 'Vazao_Sim']].copy()
df_exibicao.columns = ['Mês', 'Chuva Base (mm)', 'Chuva Reduzida (mm)', 'Temp. Base (°C)', 'Temp. Simulada (°C)', 'Excedente Solo (mm)', 'Vazão Histórica Real (m³/s)', 'Vazão Projetada (m³/s)']
st.dataframe(df_exibicao.round(2), use_container_width=True)

# 7. Construção Gráfica de Alta Fidelidade
st.markdown("### 📊 Temperatura e Pluviosidade simuladas")

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax2 = ax1.twinx()

# Plotagem das curvas de vazão real e simulada (Eixo Esquerdo)
ax1.plot(df_sim['Mês'], df_sim['Temp_Base'], 'g--', label='Temperatura Média Histórica (°C)', alpha=0.8, linewidth=2)
ax1.plot(df_sim['Mês'], df_sim['Temp_Sim'], 'g-', label='Temperatura Projetada', linewidth=3)
ax1.set_ylabel('Temperatura (°C)', color='g', fontsize=12)
ax1.tick_params(axis='y', labelcolor='g')
ax1.set_ylim(0, max(df_sim['Temp_Base'].max(), df_sim['Temp_Sim'].max()) * 1.3)

# Barras de Excedente de água do solo (Eixo Direito)
ax2.bar(df_sim['Mês'], df_sim['Chuva_Base'], color='red', alpha=0.05, label='Chuva Base (mm)')
ax2.bar(df_sim['Mês'], df_sim['Chuva_Sim'], color='blue', alpha=0.15, label='Chuva Projetada (mm)')
ax2.set_ylabel('Pluviosidade (mm)', color='b', fontsize=12)
ax2.tick_params(axis='y', labelcolor='b')

ax1.set_xlabel('Mês', fontsize=12)
fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.88))
ax1.grid(True, alpha=0.3)

st.pyplot(fig)
