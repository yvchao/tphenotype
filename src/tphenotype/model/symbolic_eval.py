import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def round_expr(expr, num_digits):
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sp.Number)})


def build_expression(poles, coeffs):
    # poles: n
    # coeffs: n x d
    n, d = coeffs.shape
    s = sp.symbols("s")
    components = []
    for i in range(n):
        for j in range(d):
            expr = coeffs[i, j] / sp.Pow(s - poles[i], j + 1)
            components.append(expr)
    return components, s


def get_transfer_function(poles, coeffs, num_digits=2):
    components, s = build_expression(poles, coeffs)
    F = sum(components)
    F = round_expr(F, num_digits)
    return F, s


def symbol_ILT(poles, coeffs):
    # poles: n
    # coeffs: n x d
    n, d = coeffs.shape
    t = sp.symbols("t", real=True)
    components = []
    for i in range(n):
        for j in range(d):
            expr = (
                coeffs[i, j]
                * sp.Pow(t, j)
                * sp.exp(poles[i] * t)
                * sp.Heaviside(t)
                / np.math.factorial(j)  # pyright: ignore
            )
            components.append(expr)
    return components, t


def get_function(poles, coeffs, num_digits=2, return_complex=True):
    components, t = symbol_ILT(poles, coeffs)
    f = 0
    for expr in components:
        f += expr

    if return_complex:
        f = round_expr(f, num_digits)
        return f, t
    else:
        f_re = sp.re(f)
        f_im = sp.im(f)
        f_re = round_expr(f_re, num_digits)
        f_im = round_expr(f_im, num_digits)
        return (f_re, f_im), t


def plot_with_matplotlib(sp_plt, ax, label=None):
    for curve in sp_plt:
        data = curve.get_points()
        ax.plot(data[0], data[1], label=label)
    return ax


def plot(f, t, range_=(0, 1), ax=None, return_ax=False):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    plot_ = sp.plot(sp.re(f), (t, *range_), show=False)
    ax = plot_with_matplotlib(plot_, ax, label="Re(f)")

    plot_ = sp.plot(sp.im(f), (t, *range_), show=False)
    ax = plot_with_matplotlib(plot_, ax, label="Im(f)")
    ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    fig.tight_layout()

    if return_ax:
        return fig, ax
    else:
        return fig
