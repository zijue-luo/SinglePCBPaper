import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1) Data Loading and Processing
# ===================================================================
data = pd.read_csv("20251210_144546_DOE_results_table.csv")
data["run_index"] = np.arange(len(data))
#data.sort_values(by="wait_time", ascending=True, inplace=True)

# Significant time drift was observed in the inital segments of data
# Filtering range was estimated externally
data = data[data["run_index"] > 35]

result = (
    data
    .groupby("wait_time")
    .agg(
        ratio_mean = ("ratio_signal", "mean"),
        ratio_std = ("ratio_signal", "std"),
        trapped_mean = ("trapped_signal", "mean"),
        trapped_std = ("trapped_signal", "std"),
        n = ("ratio_signal", "count")
    )
    .reset_index()
)

result["ratio_uncertainty"] = result["ratio_std"] / np.sqrt(result["n"])
result["trapped_uncertainty"] = result["trapped_std"] / np.sqrt(result["n"])
result = result[result["wait_time"] != 0]

#print(result)
"""
mask = result["ratio_std"] > result["ratio_mean"]
bad_points = result[mask]
print(bad_points)
bad_point_x = bad_points["wait_time"].to_numpy()
suspect_data = data[data["wait_time"].isin(bad_point_x)].copy()
suspect_data = suspect_data.sort_values(by="wait_time")
print(suspect_data)
"""

result["wait_time"] = result["wait_time"] / 1000  # Convert unit from us to ms

"""
def flag_outliers_group(group, col="ratio_signal", thresh=3.5):
    v = group[col]
    med = v.median()
    mad = (v - med).abs().median()
    if mad == 0:
        return pd.Series(False, index=group.index)
    z = 0.6745 * (v - med) / mad
    return z.abs() > thresh

mask_outlier = (
    data
    .groupby("wait_time", group_keys=False)
    .apply(flag_outliers_group)
)
data["is_outlier"] = mask_outlier
outliers_by_time = data[data["is_outlier"]].sort_values("run_index")
print(outliers_by_time[[
    "run_index",
    "wait_time",
    "ratio_signal",
    "trapped_signal",
    "loading_signal",
    "time_cost"
]])
"""

# 2) Curve Fitting and Lifetime Extracting
# ===================================================================
def exp_with_C(t, A, tau, C):
    return A * np.exp(-t / tau) + C

def exp_no_C(t, A, tau):
    return A * np.exp(-t / tau)

fit_model = exp_with_C
fit_limit = (0, 5.05)

# Prepare data for fitting
fitting_result = result[result["wait_time"] < fit_limit[1]].copy()
t = fitting_result["wait_time"].to_numpy()
y_ratio = fitting_result["ratio_mean"].to_numpy()
yerr_ratio = fitting_result["ratio_uncertainty"].to_numpy()
y_count = fitting_result["trapped_mean"].to_numpy()
yerr_count = fitting_result["trapped_uncertainty"].to_numpy()

# Initial Guess
if fit_model is exp_with_C:
    p0_ratio  = [y_ratio.max(), 1.0, y_ratio.min()]
    p0_count  = [y_count.max(), 1.0, y_count.min()]
elif fit_model is exp_no_C:
    p0_ratio  = [y_ratio.max(), 1.0]
    p0_count  = [y_count.max(), 1.0]
else:
    raise

# Fitting: ratio
popt_ratio, pcov_ratio = curve_fit(
    fit_model,
    t, y_ratio,
    sigma=yerr_ratio,
    absolute_sigma=True,
    p0=p0_ratio,
    maxfev=10000,
)
perr_ratio = np.sqrt(np.diag(pcov_ratio))

# Fitting: count
popt_count, pcov_count = curve_fit(
    fit_model,
    t, y_count,
    sigma=yerr_count,
    absolute_sigma=True,
    p0=p0_count,
    maxfev=10000,
)
perr_count = np.sqrt(np.diag(pcov_count))

# print result
def print_fit_result(title, popt, perr):
    print(f"\n=== {title} ===")
    if len(popt) == 3:
        A, tau, C = popt
        dA, dtau, dC = perr
        print(f"A   = {A: .4g} ± {dA:.4g}")
        print(f"tau = {tau: .4g} ms ± {dtau:.4g} ms")
        print(f"C   = {C: .4g} ± {dC:.4g}")
    elif len(popt) == 2:
        A, tau = popt
        dA, dtau = perr
        print(f"A   = {A: .4g} ± {dA:.4g}")
        print(f"tau = {tau: .4g} ms ± {dtau:.4g} ms")
    else:
        for i, (v, dv) in enumerate(zip(popt, perr)):
            print(f"p[{i}] = {v:.4g} ± {dv:.4g}")

print_fit_result("Fit to ratio_mean", popt_ratio, perr_ratio)
print_fit_result("Fit to trapped_mean", popt_count, perr_count)

# 2) Plot Result
# ===================================================================
def create_plot(x, y, yerr, yname, model=None, popt=None, fit_label=None, x_fit_range=(0, 5.05), title_suffix=""):

    plt.figure(figsize=(10, 8))

    plt.errorbar(
        x, y, yerr=yerr,       # data
        fmt="x",               # marker style
        capsize=3,             # lenth of dash on error bar
        linestyle="none",      # plot scatter
        label="Data"
    )

    if (model is not None) and (popt is not None):
        x_fit = np.linspace(x_fit_range[0], x_fit_range[1], 400)
        y_fit = model(x_fit, *popt)
        plt.plot(
            x_fit, y_fit,
            linestyle="--",
            label=fit_label if fit_label is not None else "Fit"
        )

    plt.xlabel("Wait time (ms)")
    plt.ylabel(yname)
    plt.title("Electron Lifetime Measurement" + title_suffix)
    plt.xlim(x_fit_range)
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()

def make_fit_label(prefix, popt, perr):
    if len(popt) == 3:
        A, tau, C = popt
        dA, dtau, dC = perr
        return (f"{prefix}: A={A:.3g}±{dA:.2g}, "
                f"τ={tau:.3g}±{dtau:.2g} ms, "
                f"C={C:.2g}±{dC:.1g}")
    elif len(popt) == 2:
        A, tau = popt
        dA, dtau = perr
        return (f"{prefix}: A={A:.3g}±{dA:.2g}, "
                f"τ={tau:.3g}±{dtau:.2g} ms")
    else:
        return prefix

label_ratio  = make_fit_label("Fit", popt_ratio, perr_ratio)
label_count  = make_fit_label("Fit", popt_count, perr_count)

create_plot(
    result["wait_time"],
    result["ratio_mean"],
    result["ratio_uncertainty"],
    "Trapped Ratio",
    model=fit_model,
    popt=popt_ratio,
    fit_label=label_ratio,
    x_fit_range=fit_limit,
    title_suffix=" (Ratio)"
)
create_plot(
    result["wait_time"],
    result["trapped_mean"],
    result["trapped_uncertainty"],
    "Trapped Count",
    model=fit_model,
    popt=popt_count,
    fit_label=label_count,
    x_fit_range=fit_limit,
    title_suffix=" (Count)"
)
