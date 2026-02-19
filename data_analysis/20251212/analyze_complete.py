import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ============================================================
# 0) CONFIG
# ============================================================
CSV_PATH = "no_of_repeats_merged.csv"

# Core columns
N_COL = "no_of_repeats"
T_COL = "time_accum"          # seconds
LOADING_COL = "loading_signal"
TRAPPED_COL = "trapped_signal"
LOST_COL = "lost_signal"
RATIO_COL = "ratio_signal"
RATIO_LOST_COL = "ratio_lost"

# Output directory
OUTDIR = "guidance_outputs"

# Choose reference N values for decision table
N_REFS = [2000, 4000, 8000, 16000]

# Ratio diagnostics: number of loading bins
N_LOADING_BINS = 6

# Regression-normalized observable:
# Use T_norm = T - beta*L (beta learned from T ~ L + controls)
# Then define a normalized rate y = T_norm / N to compare across N
USE_REGRESSION_NORMALIZED = True


# ============================================================
# 1) Utilities
# ============================================================
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def check_required_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def add_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add time in hours, centered time, invN, and rates used for comparison."""
    out = df.copy()
    out = out.dropna(subset=[N_COL, T_COL]).copy()
    out = out[out[N_COL] > 0].copy()

    out["t_hr"] = out[T_COL] / 3600.0
    out["t_hr_c"] = out["t_hr"] - out["t_hr"].mean()
    out["invN"] = 1.0 / out[N_COL].astype(float)

    # Convert counts -> per-shot rates for comparability
    for col in [LOADING_COL, TRAPPED_COL, LOST_COL]:
        if col in out.columns:
            out[f"{col}_rate"] = out[col].astype(float) / out[N_COL].astype(float)

    return out

def safe_var(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    return float(np.var(x, ddof=1))

def safe_mean(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))

def fit_ancova_time(df: pd.DataFrame, y: str) -> dict:
    """
    ANCOVA time effect controlling group offsets by N:
        m0: y ~ C(N)
        m1: y ~ C(N) + t
    Return k, k_se, partial R^2, and a "time variance fraction" estimate.
    """
    work = df[[N_COL, "t_hr_c", y]].dropna().copy()
    m0 = smf.ols(f"{y} ~ C({N_COL})", data=work).fit()
    m1 = smf.ols(f"{y} ~ C({N_COL}) + t_hr_c", data=work).fit()

    SSE0 = float(np.sum(m0.resid ** 2))
    SSE1 = float(np.sum(m1.resid ** 2))
    partial_R2 = 1.0 - SSE1 / SSE0 if SSE0 > 0 else np.nan

    k = float(m1.params.get("t_hr_c", np.nan))
    k_se = float(m1.bse.get("t_hr_c", np.nan))
    k_ci95 = (k - 1.96 * k_se, k + 1.96 * k_se) if np.isfinite(k_se) else (np.nan, np.nan)

    var_t = float(work["t_hr_c"].var(ddof=1))
    var_time_component = (k ** 2) * var_t
    var_total = float(work[y].var(ddof=1))
    frac_var_time = var_time_component / var_total if var_total > 0 else np.nan

    return {
        "k_per_hr": k,
        "k_se": k_se,
        "k_ci95_low": k_ci95[0],
        "k_ci95_high": k_ci95[1],
        "partial_R2_time": partial_R2,
        "frac_var_time_est": frac_var_time,
        "m1": m1,
        "resid": m1.resid,
        "work": work,
    }

def fit_var_vs_invN(df: pd.DataFrame, resid: pd.Series, y_mean_byN: pd.Series) -> dict:
    """
    Fit: Var(resid | N) ≈ a*(1/N) + b
    Poisson slope estimate for rate/ratio: a_poisson ≈ mean(p*(1-p)) using y_mean_byN.
    """
    tmp = df[[N_COL, "invN"]].copy()
    tmp["resid"] = resid.values

    g = tmp.groupby(N_COL, sort=True)
    summary = g.agg(
        N=(N_COL, "first"),
        invN=("invN", "first"),
        n_points=("resid", "size"),
        resid_var=("resid", safe_var),
    ).reset_index(drop=True)

    # attach mean p per N (for Poisson estimate)
    # y_mean_byN should be indexed by N
    summary["p_mean"] = summary["N"].map(y_mean_byN.to_dict())

    # OLS fit resid_var ~ const + invN
    X = sm.add_constant(summary["invN"].to_numpy())
    yv = summary["resid_var"].to_numpy()
    model = sm.OLS(yv, X, missing="drop").fit()

    b = float(model.params[0])
    a = float(model.params[1])
    b_se = float(model.bse[0])
    a_se = float(model.bse[1])

    # Poisson/Binomial expectation in "rate/ratio space"
    summary["a_poisson_est"] = summary["p_mean"] * (1.0 - summary["p_mean"])
    a_poisson = float(np.nanmean(summary["a_poisson_est"].to_numpy()))

    a_extra = a - a_poisson
    poisson_frac = a_poisson / a if a != 0 else np.nan
    extra_frac = a_extra / a if a != 0 else np.nan

    return {
        "a_emp": a,
        "a_emp_se": a_se,
        "b": b,
        "b_se": b_se,
        "a_poisson": a_poisson,
        "a_extra": a_extra,
        "poisson_frac_in_1overN": poisson_frac,
        "extra_frac_in_1overN": extra_frac,
        "var_model": model,
        "var_summary_byN": summary,
    }

def rsd_from_ab(mu: float, a: float, b: float, N: float) -> float:
    """Relative std dev (CV): sqrt(a/N + b)/mu"""
    if (mu is None) or (mu == 0) or (not np.isfinite(mu)):
        return np.nan
    v = a / N + b
    if v < 0:
        return np.nan
    return float(np.sqrt(v) / mu)

def improvement_if_doubleN(mu: float, a: float, b: float, N: float) -> float:
    """
    Return fractional improvement in std when doubling N:
        improvement = 1 - sigma(2N)/sigma(N)
    """
    if not np.isfinite(mu) or mu == 0:
        return np.nan
    v1 = a / N + b
    v2 = a / (2 * N) + b
    if v1 <= 0 or v2 <= 0:
        return np.nan
    s1 = np.sqrt(v1)
    s2 = np.sqrt(v2)
    return float(1.0 - (s2 / s1))

def make_n_tradeoff_plot(decision_rows: list[dict], outpath: str) -> None:
    """Plot RSD(N) curves from (mu, a, b) for each metric."""
    Ns = np.array([500, 1000, 2000, 4000, 8000, 16000, 32000, 64000], dtype=float)

    plt.figure(figsize=(11, 7))
    for row in decision_rows:
        mu, a, b = row["mu"], row["a_emp"], row["b"]
        name = row["metric"]
        rsd = [rsd_from_ab(mu, a, b, N) for N in Ns]
        plt.plot(Ns, rsd, marker="x", label=name)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N (no_of_repeats)")
    plt.ylabel("RSD(N) = std/mean (dimensionless)")
    plt.title("Cross-metric comparable noise vs N: RSD(N) from Var ≈ a/N + b")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_time_effect(df: pd.DataFrame, y: str, fit: dict, outpath: str, title: str) -> None:
    """
    Plot y centered by group mean vs time, with fitted slope k.
    """
    work = df[[N_COL, "t_hr_c", y]].dropna().copy()
    # center by within-N mean
    work["y_centered"] = work[y] - work.groupby(N_COL)[y].transform("mean")

    k = fit["k_per_hr"]
    partial_R2 = fit["partial_R2_time"]

    x_fit = np.linspace(work["t_hr_c"].min(), work["t_hr_c"].max(), 200)
    y_fit = k * x_fit

    plt.figure(figsize=(11, 6))
    plt.scatter(work["t_hr_c"], work["y_centered"], s=10, alpha=0.35, label="y - mean(within N)")
    plt.plot(x_fit, y_fit, linestyle="--",
             label=f"time trend: k={k:.3g}/hr, partial R²={partial_R2:.3g}")
    plt.xlabel("time (hours, centered)")
    plt.ylabel(f"{y} (centered within N)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def ratio_loading_bins_diagnostics(df: pd.DataFrame, out_csv: str, out_fig: str) -> None:
    """
    Diagnose why ratio is non-Poisson:
    - bin by loading_signal (or loading_rate)
    - compute empirical Var(ratio) within bin
    - compute expected Var(ratio) ~ p(1-p)/L (per-run), average over bin
    - overdispersion phi = Var_emp / mean(Var_expected)
    """
    check_required_columns(df, [RATIO_COL, LOADING_COL])

    work = df[[RATIO_COL, LOADING_COL, N_COL, "t_hr"]].dropna().copy()
    work = work[(work[RATIO_COL] >= 0) & (work[RATIO_COL] <= 1)].copy()
    work = work[work[LOADING_COL] > 0].copy()

    # Bin by loading count (not rate) because delta approx uses L in denominator
    # Use quantile bins for robustness
    qs = np.linspace(0, 1, N_LOADING_BINS + 1)
    edges = work[LOADING_COL].quantile(qs).to_numpy()
    edges[0] = max(edges[0] - 1e-9, 0)
    edges[-1] = edges[-1] + 1e-9

    work["bin"] = pd.cut(work[LOADING_COL], bins=edges, include_lowest=True, duplicates="drop")

    rows = []
    for bname, g in work.groupby("bin", observed=True):
        p = g[RATIO_COL].to_numpy()
        L = g[LOADING_COL].to_numpy()

        var_emp = np.var(p, ddof=1) if len(p) >= 2 else np.nan
        # per-run delta-method expected var ~ p(1-p)/L
        var_pred_runs = p * (1.0 - p) / L
        var_pred = float(np.nanmean(var_pred_runs))

        phi = var_emp / var_pred if (np.isfinite(var_emp) and np.isfinite(var_pred) and var_pred > 0) else np.nan

        rows.append({
            "loading_bin": str(bname),
            "n_points": len(g),
            "loading_mean": float(np.mean(L)),
            "ratio_mean": float(np.mean(p)),
            "ratio_var_emp": float(var_emp) if np.isfinite(var_emp) else np.nan,
            "ratio_var_pred_mean": var_pred,
            "overdispersion_phi": float(phi) if np.isfinite(phi) else np.nan,
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    # Plot: phi vs loading_mean
    plt.figure(figsize=(10, 6))
    plt.scatter(out["loading_mean"], out["overdispersion_phi"], marker="x")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean loading_signal in bin (counts, log)")
    plt.ylabel("Overdispersion phi = Var_emp / Var_pred (log)")
    plt.title("Ratio non-Poisson diagnosis: overdispersion vs loading level")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()


# ============================================================
# 2) Mediation (Time -> Loading -> Trapped)
# ============================================================
def mediation_models(df: pd.DataFrame, out_csv: str) -> None:
    """
    Quantify: How much time effect in trapped is mediated by loading.
    Compare models (control C(N)):
      L ~ C(N) + t
      T ~ C(N) + t
      T ~ C(N) + L
      T ~ C(N) + L + t
    Output SSE and partial R^2 style deltas.
    """
    check_required_columns(df, [N_COL, "t_hr_c", LOADING_COL, TRAPPED_COL])

    work = df[[N_COL, "t_hr_c", LOADING_COL, TRAPPED_COL]].dropna().copy()

    # loading model
    mL = smf.ols(f"{LOADING_COL} ~ C({N_COL}) + t_hr_c", data=work).fit()

    # trapped models
    mT0 = smf.ols(f"{TRAPPED_COL} ~ C({N_COL})", data=work).fit()
    mTt = smf.ols(f"{TRAPPED_COL} ~ C({N_COL}) + t_hr_c", data=work).fit()
    mTL = smf.ols(f"{TRAPPED_COL} ~ C({N_COL}) + {LOADING_COL}", data=work).fit()
    mTLt = smf.ols(f"{TRAPPED_COL} ~ C({N_COL}) + {LOADING_COL} + t_hr_c", data=work).fit()

    def SSE(m): return float(np.sum(m.resid ** 2))

    SSE_T0 = SSE(mT0)
    SSE_Tt = SSE(mTt)
    SSE_TL = SSE(mTL)
    SSE_TLt = SSE(mTLt)

    # How much does time help without loading?
    R2_time_only = 1.0 - SSE_Tt / SSE_T0 if SSE_T0 > 0 else np.nan
    # How much does loading help beyond group offsets?
    R2_loading = 1.0 - SSE_TL / SSE_T0 if SSE_T0 > 0 else np.nan
    # How much does time still help after loading?
    R2_time_given_loading = 1.0 - SSE_TLt / SSE_TL if SSE_TL > 0 else np.nan

    # Time coefficient change (mediation intuition)
    kt = float(mTt.params.get("t_hr_c", np.nan))
    kt_afterL = float(mTLt.params.get("t_hr_c", np.nan))
    kL = float(mL.params.get("t_hr_c", np.nan))

    out = pd.DataFrame([{
        "k_loading_per_hr": kL,
        "k_trapped_per_hr_without_loading": kt,
        "k_trapped_per_hr_with_loading": kt_afterL,
        "SSE_T0": SSE_T0,
        "SSE_Tt": SSE_Tt,
        "SSE_TL": SSE_TL,
        "SSE_TLt": SSE_TLt,
        "partial_R2_time_in_trapped": R2_time_only,
        "partial_R2_loading_in_trapped": R2_loading,
        "partial_R2_time_given_loading": R2_time_given_loading,
        "interpretation_hint": (
            "If partial_R2_time_given_loading is small and k_trapped shrinks after adding loading, "
            "then trapped time drift is largely mediated by loading."
        )
    }])
    out.to_csv(out_csv, index=False)


# ============================================================
# 3) Main: decision table + plots + diagnostics
# ============================================================
def analyze_metric(df: pd.DataFrame, metric_name: str, kind: str) -> dict:
    """
    kind:
      - 'ratio' : already in [0,1]
      - 'count_rate' : treat raw count column; analyze on rate = count/N
    Returns a dictionary that includes time effect + a/N+b fit + summary.
    """
    if kind == "ratio":
        check_required_columns(df, [metric_name])
        work = df.copy()
        work = work[(work[metric_name] >= 0) & (work[metric_name] <= 1)].copy()
        work["y"] = work[metric_name].astype(float)

    elif kind == "count_rate":
        check_required_columns(df, [metric_name])
        work = df.copy()
        work = work[work[metric_name] >= 0].copy()
        work["y"] = work[metric_name].astype(float) / work[N_COL].astype(float)
        work = work[(work["y"] >= 0) & (work["y"] <= 1)].copy()

    else:
        raise ValueError(f"Unknown kind: {kind}")

    # Time effect
    te = fit_ancova_time(work, "y")

    # mean p per N for Poisson estimate
    y_mean_byN = work.groupby(N_COL)["y"].mean()

    # Var vs 1/N on residuals
    vv = fit_var_vs_invN(work, te["resid"], y_mean_byN)

    mu = float(work["y"].mean())
    return {
        "metric": metric_name,
        "kind": kind,
        "mu": mu,
        **{k: te[k] for k in ["k_per_hr", "k_se", "k_ci95_low", "k_ci95_high", "partial_R2_time", "frac_var_time_est"]},
        **{k: vv[k] for k in ["a_emp", "a_emp_se", "b", "b_se", "a_poisson", "a_extra",
                              "poisson_frac_in_1overN", "extra_frac_in_1overN"]},
        "time_effect_model": te["m1"],
        "var_summary_byN": vv["var_summary_byN"],
    }

def build_decision_table(metric_results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in metric_results:
        a, b, mu = r["a_emp"], r["b"], r["mu"]

        # characteristic N* = a/b (if b <= 0, not meaningful)
        N_star = (a / b) if (np.isfinite(a) and np.isfinite(b) and b > 0) else np.nan

        row = {
            "metric": r["metric"],
            "kind": r["kind"],
            "mu": mu,
            "k_per_hr": r["k_per_hr"],
            "k_ci95_low": r["k_ci95_low"],
            "k_ci95_high": r["k_ci95_high"],
            "relative_drift_pct_per_hr": (r["k_per_hr"] / mu * 100.0) if (mu != 0 and np.isfinite(mu)) else np.nan,
            "partial_R2_time": r["partial_R2_time"],
            "a_emp": a,
            "b": b,
            "a_poisson": r["a_poisson"],
            "overdispersion_a_over_ap": (a / r["a_poisson"]) if (r["a_poisson"] and r["a_poisson"] > 0) else np.nan,
            "poisson_frac_in_1overN": r["poisson_frac_in_1overN"],
            "N_star_a_over_b": N_star,
        }

        # add RSD at reference N, plus marginal improvement
        for N in N_REFS:
            row[f"RSD_at_N={N}"] = rsd_from_ab(mu, a, b, N)
            row[f"improve_std_if_doubleN_from_{N}"] = improvement_if_doubleN(mu, a, b, N)

        rows.append(row)

    out = pd.DataFrame(rows)
    # helpful ordering: prefer lower RSD at your main N_ref, and lower relative drift
    keyN = N_REFS[len(N_REFS)//2]  # e.g. 4000/8000 depending list length
    sort_cols = [f"RSD_at_N={keyN}", "relative_drift_pct_per_hr"]
    out = out.sort_values(by=sort_cols, ascending=[True, True], na_position="last").reset_index(drop=True)
    return out

def main():
    ensure_outdir(OUTDIR)

    df = pd.read_csv(CSV_PATH)
    # Required minimal columns for all tasks
    check_required_columns(df, [N_COL, T_COL])

    df = add_common_fields(df)

    # ---------- Metrics to analyze ----------
    metrics = [
        (RATIO_COL, "ratio"),
        (RATIO_LOST_COL, "ratio"),
        (TRAPPED_COL, "count_rate"),
        (LOST_COL, "count_rate"),
        (LOADING_COL, "count_rate"),
    ]
    # Keep only metrics that exist in the file
    metrics = [(m, k) for (m, k) in metrics if m in df.columns]

    metric_results = []
    for m, kind in metrics:
        metric_results.append(analyze_metric(df, m, kind))

    # ---------- Decision table ----------
    decision = build_decision_table(metric_results)
    decision_path = os.path.join(OUTDIR, "decision_table.csv")
    decision.to_csv(decision_path, index=False)

    # ---------- Time effect plots (centered within N) ----------
    for r in metric_results:
        m = r["metric"]
        kind = r["kind"]
        # reconstruct the "y" used in fitting
        if kind == "ratio":
            work = df[(df[m] >= 0) & (df[m] <= 1)].copy()
            work["y"] = work[m].astype(float)
        else:
            work = df[df[m] >= 0].copy()
            work["y"] = work[m].astype(float) / work[N_COL].astype(float)
            work = work[(work["y"] >= 0) & (work["y"] <= 1)].copy()

        te = fit_ancova_time(work, "y")
        fig_path = os.path.join(OUTDIR, f"time_effect_{m}.png")
        plot_time_effect(work, "y", te, fig_path, title=f"Time effect (centered within N): {m} ({kind})")

    # ---------- Mediation analysis ----------
    if (LOADING_COL in df.columns) and (TRAPPED_COL in df.columns):
        mediation_path = os.path.join(OUTDIR, "mediation_summary.csv")
        mediation_models(df, mediation_path)

    # ---------- Ratio diagnostics: overdispersion vs loading bins ----------
    if (RATIO_COL in df.columns) and (LOADING_COL in df.columns):
        ratio_diag_csv = os.path.join(OUTDIR, "ratio_loading_bins_diagnostics.csv")
        ratio_diag_fig = os.path.join(OUTDIR, "ratio_overdispersion_vs_loading.png")
        ratio_loading_bins_diagnostics(df, ratio_diag_csv, ratio_diag_fig)

    # ---------- N tradeoff curve plot (RSD(N)) ----------
    tradeoff_fig = os.path.join(OUTDIR, "N_tradeoff_curves.png")
    make_n_tradeoff_plot(
        decision_rows=[{
            "metric": row["metric"],
            "mu": row["mu"],
            "a_emp": row["a_emp"],
            "b": row["b"],
        } for _, row in decision.iterrows()],
        outpath=tradeoff_fig
    )

    print("\n=== DONE ===")
    print(f"Wrote: {decision_path}")
    if (LOADING_COL in df.columns) and (TRAPPED_COL in df.columns):
        print(f"Wrote: {os.path.join(OUTDIR, 'mediation_summary.csv')}")
    if (RATIO_COL in df.columns) and (LOADING_COL in df.columns):
        print(f"Wrote: {os.path.join(OUTDIR, 'ratio_loading_bins_diagnostics.csv')}")
        print(f"Wrote: {os.path.join(OUTDIR, 'ratio_overdispersion_vs_loading.png')}")
    print(f"Wrote: {tradeoff_fig}")
    print(f"All outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
