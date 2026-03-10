"""
Plotting for tickling/RF scan analysis.
Uses stitched left+right segment fit curves for display (not merged model on full x).
"""
import numpy as np
import matplotlib.pyplot as plt

from fitting_functions import gaussian_sum
from data_io import load_data, load_configuration


def _plot_fit_curve(ax, x, y_data, fit_result, mode):
    """Add scatter and fit curve. Uses stitched left+right segment fits when present."""
    ax.scatter(x, y_data, label="data")
    best_fit = fit_result[mode].get("best")
    if best_fit is None or not best_fit.get("ok", False):
        ax.set_ylabel(f"{mode} count / loading count")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        return False
    best_l = fit_result[mode].get("best_left")
    best_r = fit_result[mode].get("best_right")
    split_x = fit_result.get("split_x")
    xfit = np.linspace(np.min(x), np.max(x), 1000)
    if (best_l is not None or best_r is not None) and split_x is not None:
        yfit = np.zeros_like(xfit, dtype=float)
        if best_l and best_l.get("ok") and best_l.get("popt") is not None:
            mask = xfit <= split_x
            yfit[mask] = gaussian_sum(xfit[mask], *best_l["popt"])
        if best_r and best_r.get("ok") and best_r.get("popt") is not None:
            mask = xfit > split_x
            yfit[mask] = gaussian_sum(xfit[mask], *best_r["popt"])
        n_peaks = best_fit["n_peaks"]
    else:
        popt = best_fit["popt"]
        n_peaks = best_fit["n_peaks"]
        yfit = gaussian_sum(xfit, *popt)
    ax.plot(xfit, yfit, label=f"fit {n_peaks} peaks")
    ax.set_ylabel(f"{mode} count / loading count")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    return True


def plot_fine_scan_from_arrays(x, y, fit_result, out_name, rf_amp_label, title_label="averaged"):
    """Plot data and fit from pre-loaded arrays."""
    modes = ["lost", "trapped"]
    y_dict = {"lost": y["ratio_lost"], "trapped": y["ratio_signal"]}
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    axes = {"lost": axs[0], "trapped": axs[1]}
    for mode in modes:
        ax = axes[mode]
        best_fit = fit_result[mode].get("best")
        if best_fit is None or not best_fit.get("ok", False):
            ax.scatter(x, y_dict[mode], label="data")
            ax.set_title(f"{title_label}: RF={rf_amp_label} ({mode}) - [NO FIT]")
        else:
            ax.set_title(f"{title_label}: RF={rf_amp_label} ({mode})")
        _plot_fit_curve(ax, x, y_dict[mode], fit_result, mode)
    axes["trapped"].set_xlabel("Tickle Frequency (MHz)")
    path = f"{out_name}.png" if not str(out_name).lower().endswith(".png") else out_name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {path}")


def plot_fine_scan(timestamp, fit_result, out_name, data_root=None):
    """Plot single-scan fit."""
    x, ys = load_data(timestamp, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
    conf = load_configuration(timestamp, conf_names=["RF_amplitude"], data_root=data_root)
    rf_val = conf[0] if conf and conf[0] is not None else float("nan")
    rf_label = f"{rf_val:.2f} dBm" if np.isfinite(rf_val) else "?"
    y = {"lost": ys["ratio_lost"], "trapped": ys["ratio_signal"]}
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    axes = {"lost": axs[0], "trapped": axs[1]}
    for mode in ("lost", "trapped"):
        ax = axes[mode]
        best_fit = fit_result[mode].get("best")
        if best_fit is None or not best_fit.get("ok", False):
            ax.scatter(x, y[mode], label="data")
            ax.set_title(f"{timestamp}: RF={rf_label} ({mode}) - [NO FIT]")
        else:
            ax.set_title(f"{timestamp}: RF={rf_label} ({mode})")
        _plot_fit_curve(ax, x, y[mode], fit_result, mode)
    axes["trapped"].set_xlabel("Tickle Frequency (MHz)")
    path = f"{out_name}.png" if not str(out_name).lower().endswith(".png") else out_name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {path}")
