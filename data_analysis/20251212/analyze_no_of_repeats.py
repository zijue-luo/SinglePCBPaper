import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

# =========================
# Config
# =========================
CSV_PATH = "no_of_repeats_merged.csv"

N_COL = "no_of_repeats"
T_COL = "time_accum"     # seconds in your table

# 你想批量分析的列：
# - kind="ratio"      : y 本身是概率/比例（0~1），可直接做 Poisson/Binomial 对比
# - kind="count_rate" : y 是 count，需要先除以 N 转成 rate 才适合做 1/N + Poisson 对比
METRICS = [
    ("ratio_signal", "ratio"),
    ("ratio_lost", "ratio"),
    ("trapped_signal", "count_rate"),
    ("lost_signal", "count_rate"),
    ("loading_signal", "count_rate"),  # 若它真是“计数”，也可以；否则就别放
]


# =========================
# Helpers
# =========================
def _prepare_base_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {N_COL, T_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out = out.dropna(subset=[N_COL, T_COL])
    out = out[out[N_COL] > 0].copy()

    out["t_hr"] = out[T_COL] / 3600.0
    out["t_hr_c"] = out["t_hr"] - out["t_hr"].mean()  # center for interpretability
    out["invN"] = 1.0 / out[N_COL].astype(float)
    return out


def analyze_metric(df: pd.DataFrame, y_col: str, kind: str):
    """
    输出：
    - 时间效应：k、k_se、95% CI、partial R^2、k^2 Var(t) 估计的方差占比
    - 残差噪声：Var(resid|N) ≈ a*(1/N)+b，并与 Poisson 的 a_poisson 对比（如果适用）
    - 画图：时间效应图 + Var vs 1/N 图
    """
    if y_col not in df.columns:
        raise ValueError(f"Column not found: {y_col}")

    work = df[[N_COL, "t_hr", "t_hr_c", "invN", y_col]].dropna().copy()

    # 选择分析变量 y
    if kind == "ratio":
        # ratio 应该在 0~1（如果你允许 >1 就放宽或删掉这一行）
        work = work[(work[y_col] >= 0) & (work[y_col] <= 1)].copy()
        work["y"] = work[y_col].astype(float)

    elif kind == "count_rate":
        # count -> rate = count / N，转成“每 shot 的概率/率”
        work = work[work[y_col] >= 0].copy()
        work["y"] = work[y_col].astype(float) / work[N_COL].astype(float)
        # rate 理论上也应在 0~1（如果你的计数定义不是 Binomial，就自行调整）
        work = work[(work["y"] >= 0) & (work["y"] <= 1)].copy()

    else:
        raise ValueError(f"Unknown kind: {kind}")

    if len(work) < 20:
        raise ValueError(f"Too few rows after filtering for {y_col} ({len(work)})")

    # -------------------------
    # A) Time effect: ANCOVA
    #   m0: y ~ C(N)
    #   m1: y ~ C(N) + t
    # -------------------------
    m0 = smf.ols(f"y ~ C({N_COL})", data=work).fit()
    m1 = smf.ols(f"y ~ C({N_COL}) + t_hr_c", data=work).fit()

    SSE0 = float(np.sum(m0.resid ** 2))
    SSE1 = float(np.sum(m1.resid ** 2))
    partial_R2_time = 1.0 - SSE1 / SSE0

    k = float(m1.params["t_hr_c"])      # rate per hour
    k_se = float(m1.bse["t_hr_c"])
    k_ci95 = (k - 1.96 * k_se, k + 1.96 * k_se)

    y_mean = float(work["y"].mean())
    rel_drift_pct_per_hr = (k / y_mean) * 100.0 if y_mean != 0 else np.nan

    var_t = float(work["t_hr_c"].var(ddof=1))
    var_time_component = (k ** 2) * var_t
    var_total = float(work["y"].var(ddof=1))
    frac_var_time = var_time_component / var_total if var_total > 0 else np.nan

    # -------------------------
    # B) Residual variance vs 1/N
    #   resid = y - (alpha_N + k*t)
    #   Var(resid|N) ≈ a*(1/N) + b
    # -------------------------
    work["resid"] = m1.resid

    summary = (
        work.groupby(N_COL, sort=True)
            .agg(
                N=(N_COL, "first"),
                invN=("invN", "first"),
                n_points=("resid", "size"),
                resid_var=("resid", lambda x: float(np.var(x, ddof=1))),
                p_mean=("y", "mean"),
            )
            .reset_index(drop=True)
            .sort_values("N")
            .reset_index(drop=True)
    )

    X = sm.add_constant(summary["invN"].to_numpy())
    yv = summary["resid_var"].to_numpy()
    var_model = sm.OLS(yv, X).fit()

    b = float(var_model.params[0])
    a_emp = float(var_model.params[1])
    b_se = float(var_model.bse[0])
    a_se = float(var_model.bse[1])

    # Poisson/Binomial slope a_poisson ≈ E[p(1-p)]（在“rate/ratio”空间里成立）
    summary["a_poisson_est"] = summary["p_mean"] * (1.0 - summary["p_mean"])
    a_poisson = float(summary["a_poisson_est"].mean())

    a_extra = a_emp - a_poisson
    poisson_frac = a_poisson / a_emp if a_emp != 0 else np.nan
    extra_frac = a_extra / a_emp if a_emp != 0 else np.nan

    # -------------------------
    # Print
    # -------------------------
    print(f"\n==================== {y_col} ({kind}) ====================")
    print("Time effect (ANCOVA): y ~ C(N) + k*t")
    print(f"k = {k:.6g} ± {k_se:.2g} (per hour, 1σ), 95% CI [{k_ci95[0]:.6g}, {k_ci95[1]:.6g}]")
    print(f"mean(y) = {y_mean:.6g}  -> relative drift ≈ {rel_drift_pct_per_hr:.3g}%/hour")
    print(f"partial R^2 (time | group) = {partial_R2_time:.4f}")
    print(f"Estimated Var(time component) = k^2 Var(t) = {var_time_component:.6g}")
    print(f"Estimated fraction of variance from time trend ≈ {frac_var_time:.3f}")

    print("\nResidual noise after accounting for group + time:")
    print("Var(resid|N) ≈ a*(1/N) + b")
    print(f"a_emp = {a_emp:.6g} ± {a_se:.2g}")
    print(f"b     = {b:.6g} ± {b_se:.2g}")
    print(f"a_poisson ≈ mean(p(1-p)) = {a_poisson:.6g}")
    print(f"a_extra = a_emp - a_poisson = {a_extra:.6g}")
    print(f"Poisson fraction within 1/N term ≈ {poisson_frac:.3f}")
    print(f"Extra (non-Poisson but ~1/N) fraction ≈ {extra_frac:.3f}")

    # -------------------------
    # Plots
    # -------------------------
    # (1) time effect visualization: remove group mean, keep time
    work["y_centered_by_group"] = work["y"] - work.groupby(N_COL)["y"].transform("mean")

    x_fit = np.linspace(work["t_hr_c"].min(), work["t_hr_c"].max(), 200)
    y_fit = k * x_fit  # centered, so intercept ~ 0

    plt.figure(figsize=(10, 6))
    plt.scatter(work["t_hr_c"], work["y_centered_by_group"], marker=".", alpha=0.35,
                label="y - mean(within N)")
    plt.plot(x_fit, y_fit, linestyle="--",
             label=f"time trend: k={k:.3g}/hr, partial R²={partial_R2_time:.3g}")
    plt.xlabel("time (hours, centered)")
    plt.ylabel(f"{y_col} (as rate/ratio) - mean(within N)")
    plt.title(f"Time effect: {y_col}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # (2) Var(resid) vs 1/N with Poisson comparison
    x = summary["invN"].to_numpy()
    y = summary["resid_var"].to_numpy()

    x_line = np.linspace(0, x.max() * 1.05, 200)
    y_line = a_emp * x_line + b
    y_poiss = a_poisson * x_line

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker="x", label="Measured Var(resid) per N")
    plt.plot(x_line, y_line, linestyle="--", label=f"Fit: a*(1/N)+b (a={a_emp:.3g}, b={b:.3g})")
    plt.plot(x_line, y_poiss, linestyle=":", label=f"Poisson-only: a_P*(1/N) (a_P={a_poisson:.3g})")
    plt.xlabel("1 / N  (N = no_of_repeats)")
    plt.ylabel("Var(residual) after accounting for group + time")
    plt.title(f"1/N noise decomposition: {y_col}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 返回用于汇总写表
    return {
        "metric": y_col,
        "kind": kind,
        "k_per_hr": k,
        "k_se": k_se,
        "k_ci95_low": k_ci95[0],
        "k_ci95_high": k_ci95[1],
        "partial_R2_time": partial_R2_time,
        "frac_var_time_est": frac_var_time,
        "a_emp": a_emp,
        "a_emp_se": a_se,
        "b": b,
        "b_se": b_se,
        "a_poisson": a_poisson,
        "a_extra": a_extra,
        "poisson_frac_in_1overN": poisson_frac,
        "extra_frac_in_1overN": extra_frac,
    }


def main():
    df = pd.read_csv(CSV_PATH)
    df = _prepare_base_df(df)

    results = []
    for y_col, kind in METRICS:
        res = analyze_metric(df, y_col, kind)
        results.append(res)

    out = pd.DataFrame(results)
    out.to_csv("multi_metric_time_and_noise_report.csv", index=False)
    print("\nSaved: multi_metric_time_and_noise_report.csv")
    print(out)


if __name__ == "__main__":
    main()
