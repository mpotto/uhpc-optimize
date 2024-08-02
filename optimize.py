import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.optimize import Bounds, minimize, LinearConstraint

from utils import anderson_andreasen_psd, empirical_psd, objective, plot_components

pd.set_option("display.precision", 2)

# Initialize session state
if "data" not in st.session_state:
    st.session_state["data"] = None

if "constraints" not in st.session_state:
    st.session_state["variables"] = None

if "optimization" not in st.session_state:
    st.session_state["optimization"] = None

st.markdown("### Dados")

with st.form("data-form"):

    data_file = st.file_uploader(
        "Insira os dados granulométricos.", type=[".csv", ".xlsx"]
    )
    use_example_file = st.checkbox(
        "Usar dados de exemplo pré-carregados.",
        False,
    )

    submitted = st.form_submit_button("Submeter")
    if submitted:
        if use_example_file:
            data_file = "data/data.xlsx"

        df = pd.read_excel(data_file).astype(np.float32)

        st.session_state["data"] = df

if st.session_state["data"] is None:
    st.stop()

# Preparing data
df = st.session_state["data"]

diameter = df.iloc[:, 0].to_numpy(np.float32)[::-1] * 1e-3
d_min = diameter.min()
d_max = diameter.max()

cumulative_finer = df.iloc[:, 1:].to_numpy(np.float32)[::-1] * 1e-2

n_materials = cumulative_finer.shape[1]

names = list(df.columns)  # Names are inferred from spreadsheet.

with st.expander("Verifique os dados."):
    st.dataframe(df.style.format(precision=2))

# Curve preview
with st.expander("Visualize as distribuições granulométricas (passante)."):
    fig = plot_components(diameter, cumulative_finer, names, n_materials)
    st.pyplot(fig)


st.markdown("### Configuração")

q = np.float32(st.text_input("Módulo de distribuição ($q$):", value=0.23))

variables = st.multiselect(
    "Selecione as variáveis que gostaria de restringir:", names[1:]
)
n_constraints = len(variables)

lower_bounds = np.zeros(n_materials)
upper_bounds = np.ones(n_materials)

if n_constraints != 0:
    columns = st.columns(n_constraints)
    for i, col in enumerate(columns):
        col.write(f"{variables[i]}")
        constraint_type = col.selectbox(
            "Tipo de restrição.", ["Igualdade", "Desigualdade"], key=variables[i]
        )
        index = names.index(variables[i]) - 1

        if constraint_type == "Igualdade":
            val = col.number_input(
                "Valor",
                min_value=0.0,
                max_value=1.0,
                step=0.005,
                key=variables[i] + "o",
            )

            lower_bounds[index] = val
            upper_bounds[index] = val

        elif constraint_type == "Desigualdade":
            slider = col.slider(
                "Selecione o intervalo",
                0.0,
                1.0,
                (0.25, 0.75),
                key=variables[i] + "a",
            )
            lower_bounds[index] = slider[0]
            upper_bounds[index] = slider[1]

bounds = Bounds(lower_bounds, upper_bounds)

st.markdown("### Otimização")

with st.form("optimize"):

    A = np.ones((1, n_materials))
    sum_to_one = LinearConstraint(A, np.ones([1]), np.ones([1]))

    initial_guess = np.ones(n_materials) / n_materials
    submitted = st.form_submit_button("Submeter")

    if submitted:
        result = minimize(
            objective,
            initial_guess,
            args=(diameter, cumulative_finer, d_min, d_max, q),
            method="trust-constr",
            constraints=[sum_to_one],
            bounds=bounds,
        )

        y_target = anderson_andreasen_psd(diameter, q, d_min, d_max)

        st.markdown("**Visualização do Ajuste**")
        fig, ax = plt.subplots(1, 1)
        ax.step(
            diameter,
            empirical_psd(result.x, cumulative_finer),
            color="crimson",
            label="Ajuste",
        )
        ax.plot(diameter, y_target, "-", color="black", label="Curva ótima")

        ax.legend(bbox_to_anchor=(1.01, 0.99), frameon=False)
        ax.set_xlabel("Diâmetro (m)")
        ax.set_ylabel("Passante (V/V)")
        ax.set_xscale("log")
        st.pyplot(fig)

        st.markdown("""---""")
        st.markdown("**Proporções volumétricas ótimas**")
        proportions = pd.DataFrame(np.abs(result.x).reshape(-1, 1).transpose())
        proportions.columns = names[1:]
        st.dataframe(proportions.style.format("{:.2%}"), hide_index=True)
