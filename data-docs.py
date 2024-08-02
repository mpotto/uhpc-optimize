import numpy as np
import pandas as pd
import streamlit as st

pd.set_option("display.precision", 2)

st.write(
    """
        Os dados carregados devem:
        * Especificar o diâmetro das partículas em mm.
        * Referir-se ao percentual volumétrico passante e não ao percentual retido.
        * Ter como valor mínimo de diâmetro o menor valor tal que nenhuma partícula é passante neste valor, mas há partículas passantes no primeiro valor de diâmetro *maior* que o mínimo.
        * Ter como valor máximo de diâmetro o maior valor tal que toda partícula é passante neste valor, mas há partículas não passantes no primeiro valor de diâmetro *menor* que o máximo.

        As duas últimas regras são necessárias para que os valores de $d_{\\textrm{min}}$ e $d_{\\textrm{max}}$ sejam corretamente inferidos a partir dos dados. 
        A tabela abaixo exemplifica a especificação correta dos dados:
    """
)

df = pd.read_excel("data/data.xlsx").astype(np.float32)
st.dataframe(df.style.format(precision=2))
