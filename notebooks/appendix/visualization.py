import matplotlib.pyplot as plt
import numpy as np


def pole_visualization(poles, step=-1, ax=None):
    batch_size, series_size, num_poles = poles.shape  # pylint: disable=unused-variable
    step = min(series_size - 1, step)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()
    for i in range(num_poles):
        x = np.real(poles)[:, step, i]
        y = np.imag(poles)[:, step, i]
        ax.scatter(x, y, label=f"$p_{i+1}$", alpha=0.2)
    leg = ax.legend(ncol=num_poles // 2)

    for lh in leg.legendHandles:  # pyright: ignore
        lh.set_alpha(1)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title("Poles")
    xmin, xmax = ax.get_xlim()
    xmin = min(-10.5, xmin)
    xmax = max(1.5, xmax)
    ax.set_xlim([xmin, xmax])  # pyright: ignore
    ymin, ymax = ax.get_ylim()
    ymin = min(-20, ymin, -ymax)
    ymax = -ymin
    ax.set_ylim([ymin, ymax])  # pyright: ignore
    ax.grid(True)
    fig.tight_layout()
    return fig


def coeff_visualization(coeffs, degree=1, step=-1, ax=None):
    batch_size, series_size, num_poles, max_degree = coeffs.shape  # pylint: disable=unused-variable
    degree = min(max_degree, degree)
    step = min(series_size - 1, step)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()
    for i in range(num_poles):
        x = np.real(coeffs)[:, step, i, degree - 1]
        y = np.imag(coeffs)[:, step, i, degree - 1]
        ax.scatter(x, y, label="$c_{" + f"{i+1},{degree}" + "}$", alpha=0.2)
    leg = ax.legend(ncol=num_poles // 2)

    for lh in leg.legendHandles:  # pyright: ignore
        lh.set_alpha(1)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title("Coefficients")
    xmin, xmax = ax.get_xlim()
    xmin = min(-5, xmin)
    xmax = max(5, xmax)
    ax.set_xlim([xmin, xmax])  # pyright: ignore
    ax.set_ylim([xmin, xmax])  # pyright: ignore
    ax.grid(True)
    fig.tight_layout()
    return fig


def fn_visualization(t, f, f_rec, mask, idx, window=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    f_rec_r, f_rec_i = f_rec

    step = int(np.sum(mask[idx]) - 1)
    ax.plot(
        t[idx, : step + 1],
        f[idx, : step + 1],
        label="$x(\mathbf{t})$",  # noqa # pylint: disable=E,W # pyright: ignore
        marker="o",
    )
    if window is not None:
        ax.plot(
            t[idx, step + 1 - window : step + 1],
            f_rec_r[idx, step],
            label="$\mathrm{Re}(\hat{x}(\mathbf{t}))$",  # noqa # pylint: disable=E,W # pyright: ignore
            marker="x",
        )
        ax.plot(
            t[idx, step + 1 - window : step + 1],
            f_rec_i[idx, step],
            label="$\mathrm{Im}(\hat{x}(\mathbf{t}))$",  # noqa # pylint: disable=E,W # pyright: ignore
            marker="x",
        )
    else:
        ax.plot(t[idx], f_rec_r[idx, step], label="Re(f_rec)", marker="x")
        ax.plot(t[idx], f_rec_i[idx, step], label="Im(f_rec)", marker="x")
    ymin, ymax = plt.ylim()  # pylint: disable=unused-variable
    # ax.vlines(t[idx,step],ymin,ymax,color='red', linestyles='dashed')
    ax.set_xlabel("$t$")
    ax.legend()
    fig.tight_layout()
    return fig
