import numpy as np
import matplotlib.pyplot as plt


def anderson_andreasen_psd(diameter, q=0.3, d_min=3.4e-7, d_max=1e-3):
    psd = (diameter**q - d_min**q) / (d_max**q - d_min**q)
    return psd


def empirical_psd(vol_proportion, cumulative_finer):
    psd = np.cumsum(cumulative_finer @ vol_proportion) / np.sum(
        cumulative_finer @ vol_proportion
    )
    return psd


def objective(
    vol_proportion,
    diameter,
    cumulative_finer,
    d_min,
    d_max,
    q,
):

    y_target = anderson_andreasen_psd(diameter, q, d_min, d_max)
    objective = np.mean(
        np.square(y_target - empirical_psd(vol_proportion, cumulative_finer))
    )
    return objective


def plot_components(diameter, cumulative_finer, names, n_materials):
    fig, ax = plt.subplots(1, 1)
    for i in range(n_materials):
        ax.plot(diameter, cumulative_finer[:, i], "o-", ms=3, label=names[i + 1])
    ax.set_xscale("log")
    ax.set_xlabel(r"Diâmetro (m)")
    ax.set_ylabel("Passante (V/V)")
    ax.legend(bbox_to_anchor=(1.0, -0.15), ncol=n_materials // 2, frameon=False)
    return fig
