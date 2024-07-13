import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import cvxpy as cp

pd.set_option("display.precision", 2)

st.title("Otimização da mistura de concreto")

st.markdown("## Dados")

uploaded_file = st.file_uploader("Carregue o arquivo .csv", type=[".csv", ".xlsx"])
use_example_file = st.checkbox(
    "Usar dados de exemplo pré-carregados.",
    True,
)

if use_example_file:
    uploaded_file = "data.xlsx"

df = pd.read_excel(
    uploaded_file,
).astype(np.float32)

diam = df.iloc[:, 0].to_numpy(np.float32)[::-1] * 1e-6
comp = df.iloc[:, 1:].to_numpy(np.float32)[::-1] * 1e-2
names = list(df.columns)


def plot_components(diam, comp, names):
    n_comp = comp.shape[1]
    fig, ax = plt.subplots(1, 1)
    for i in range(n_comp):
        ax.plot(diam, comp[:, i], "o-", ms=3, label=names[i + 1])
    ax.set_xscale("log")
    ax.legend(bbox_to_anchor=(1.0, -0.1), ncol=n_comp // 2, frameon=False)
    return fig


with st.expander("Pré-visualize os dados"):
    st.dataframe(df.head().style.format(precision=2))

st.markdown("## Otimização")


def psd(d, q=0.3, d_min=3.4e-7, d_max=1e-3):
    p = (d**q - d_min**q) / (d_max**q - d_min**q)
    return p


st.markdown("### Configure as restrições")

q = np.float32(st.text_input("Coeficiente q", value=0.23))

variables = st.multiselect("Selecione as variáveis para restringir.", names[1:], names[1])

n_constraints = len(variables)

constraint_types = []
constraint_values = []

columns = st.columns(n_constraints)
for i, col in enumerate(columns):
    col.write(f"**{variables[i]}**")
    constraint_type = col.selectbox(
        "Tipo de restrição.", ["Igualdade", "Desigualdade"], key=variables[i]
    )

    if constraint_type == "Igualdade":
        val = col.number_input(
            "Valor",
            min_value=0.0,
            max_value=1.0,
            step=0.005,
            key=variables[i] + "o",
        )
        constraint_types.append("eq")
        constraint_values.append([val])

    elif constraint_type == "Desigualdade":
        slider = col.slider(
            "Selecione o intervalo",
            0.0,
            1.0,
            (0.25, 0.75),
            key=variables[i] + "a",
        )

        constraint_types.append("ineq")
        constraint_values.append(slider)

st.markdown("### Execute a otimização")

y = psd(diam, q=q)

diam_test = np.logspace(np.log10(diam.min()), np.log10(diam.max()), 10**3)
y_test = psd(diam_test, q=q)

with st.form("optimize"):

    n_comp = comp.shape[1]
    vol_prop = cp.Variable(n_comp)
    cost = cp.sum_squares(y - comp @ vol_prop) / y.size

    A = np.ones((1, n_comp))

    # Configure constraints using cols
    constraints = [A @ vol_prop == 1, vol_prop >= 0]
    for i, var in enumerate(variables):
        index = names.index(var) - 1

        if constraint_types[i] == "eq":
            val = constraint_values[i]
            constraints.append(vol_prop[index] == val)
        elif constraint_types[i] == "ineq":
            values = constraint_values[i]
            constraints.append(vol_prop[index] >= values[0])
            constraints.append(vol_prop[index] <= values[1])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(verbose=False)

    submitted = st.form_submit_button("Submeter")

    if submitted:
        fig, ax = plt.subplots(1, 1)
        ax.plot(diam_test, y_test, "-", lw=1, color="k", label="Curva ótima")
        ax.plot(diam, comp @ vol_prop.value, "--", lw=2, color="crimson", label="Solução")
        ax.legend(bbox_to_anchor=(1.01, 0.99), frameon=False)
        ax.set_xlabel("Diâmetro (m)")
        ax.set_ylabel("Passante")
        ax.set_xscale("log")

        st.pyplot(fig)

        proportions = pd.DataFrame(np.abs(vol_prop.value)).transpose()
        proportions.columns = names[1:]
        st.dataframe(proportions.style.format("{:.2%}"))
