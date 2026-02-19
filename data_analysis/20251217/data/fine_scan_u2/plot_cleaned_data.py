"""
Plot cleaned averaged data (lost and trapped) with classified branches as scatter plots.
Reads from cleaned_averaged_lost.csv and cleaned_averaged_trapped.csv.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Config (edit here instead of using command-line options)
# ---------------------------------------------------------------------------
# DATA_DIR : str | Path | None
#     Directory containing cleaned_averaged_lost.csv and cleaned_averaged_trapped.csv.
#     None = use the directory of this script.
# SAVE_DIR : str | Path | None
#     Directory to save plot images. None = do not save; "." = current directory;
#     or set to a path like "plots" or "output/figures".
# SHOW_PLOTS : bool
#     If True, show plots in a window when the script is run.
#
# Error bar (line width) options
# ------------------------------
# ERRORBAR_SOURCE : "peak" | "group"
#     "peak" = use the selected peak's sigma (line width) as y-error.
#     "group" = use weighted average of peaks' sigma in each group as y-error
#     (weight = amp * sigma). Requires sigma_group column from data_cleaning.
# PLOT_WIDTH_MODE : "individual" | "averaged"
#     "individual" = each point gets its own error bar from the chosen source.
#     "averaged" = use one constant error bar for all points (simple mean of
#     the chosen width over the plotted data).
#
# Band fit initial guess per band (tune if fits do not converge)
# --------------------------------------------------------------
# Models: y = a*sqrt(b-U2) + c  and  y = a*sqrt(b-U2). b must be > max(U2).
# Dict: band name -> (a0, b0, c0) or (a0, b0). Use key None for default for any band not listed.
# For b0 use None to set b0 = max(U2) + 0.05 automatically.
#
DATA_DIR = None
SAVE_DIR = "."
SHOW_PLOTS = False
ERRORBAR_SOURCE = "group"
PLOT_WIDTH_MODE = "individual"

# Per-band initial guess: band -> (a0, b0, c0) for sqrt_offset; band -> (a0, b0) for sqrt_only.
# None key = default for bands not in the dict.
FIT_SQRT_OFFSET_INIT = {
    None: (50.0, None, 0.0),
    "band_38": (60.0, 0.0, 0.0),
    "band_61": (70.0, 0.0, 0.0),
    "band_80": (80.0, 0.0, 0.0),
    "band_102": (90.0, 0.0, 0.0),
    "band_120": (100.0, 0.0, 0.0),
    "band_144": (110.0, 0.0, 0.0),
}
FIT_SQRT_ONLY_INIT = {
    None: (50.0, None),
    "band_38": (60.0, 0.0),
    "band_61": (70.0, 0.0),
    "band_80": (80.0, 0.0),
    "band_102": (90.0, 0.0),
    "band_120": (100.0, 0.0),
    "band_144": (110.0, 0.0),
}


def _get_errorbar_array(
    subset: pd.DataFrame,
    errorbar_source: str,
    plot_width_mode: str,
) -> np.ndarray:
    """
    Return 1d array of y-error bar values (line width) for a subset.

    errorbar_source: "peak" -> sigma column; "group" -> sigma_group column.
    plot_width_mode: "individual" -> one value per row; "averaged" -> constant (mean).
    """
    if "sigma" not in subset.columns:
        return np.zeros(len(subset))
    err_col = "sigma_group" if errorbar_source == "group" else "sigma"
    if err_col not in subset.columns:
        err_col = "sigma"
    vals = np.asarray(subset[err_col], dtype=float)
    if plot_width_mode == "averaged":
        vals = np.full(len(subset), np.nanmean(vals))
    return np.atleast_1d(vals)


# ---------------------------------------------------------------------------
# Read CSV
# ---------------------------------------------------------------------------

def read_cleaned_lost(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Read cleaned averaged lost data from CSV.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to cleaned_averaged_lost.csv. If None, uses the file in the script directory.

    Returns
    -------
    pd.DataFrame
        Loaded data with columns including U2, mu, amp, sigma, band, width_class, etc.
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "cleaned_averaged_lost.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned lost CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def read_cleaned_trapped(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Read cleaned averaged trapped data from CSV.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to cleaned_averaged_trapped.csv. If None, uses the file in the script directory.

    Returns
    -------
    pd.DataFrame
        Loaded data with columns including U2, mu, amp, sigma, and band/line_id if present.
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "cleaned_averaged_trapped.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned trapped CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Branch classification for plotting
# ---------------------------------------------------------------------------

def _branch_column(df: pd.DataFrame) -> str:
    """Return the column name to use for branch classification (band or line_id)."""
    if "band" in df.columns:
        return "band"
    if "line_id" in df.columns:
        return "line_id"
    return ""


# Canonical band order (from data_cleaning.BAND_LABELS)
BAND_ORDER = ["band_38", "band_61", "band_80", "band_102", "band_120", "band_144"]


# ---------------------------------------------------------------------------
# Band fit: y = a*sqrt(b-U2) + c  and  y = a*sqrt(b-U2)
# ---------------------------------------------------------------------------

# Display names for fit plot titles
_MODEL_TITLE = {"sqrt_offset": "y = a√(b−U2) + c", "sqrt_only": "y = a√(b−U2)"}


def _format_fit_formula(popt: np.ndarray | None, model: str) -> str:
    """Format fitted parameters as formula string for legend."""
    if popt is None or len(popt) == 0:
        return ""
    if model == "sqrt_offset" and len(popt) >= 3:
        a, b, c = popt[0], popt[1], popt[2]
        return f"y = {a:.4g}√({b:.4g}−U2) + {c:.4g}"
    if model == "sqrt_only" and len(popt) >= 2:
        a, b = popt[0], popt[1]
        return f"y = {a:.4g}√({b:.4g}−U2)"
    return ""


def _model_sqrt_offset(U2: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """y = a * sqrt(b - U2) + c. Requires b > U2."""
    return a * np.sqrt(np.maximum(b - U2, 1e-12)) + c


def _model_sqrt_only(U2: np.ndarray, a: float, b: float) -> np.ndarray:
    """y = a * sqrt(b - U2). Requires b > U2."""
    return a * np.sqrt(np.maximum(b - U2, 1e-12))


def fit_band_model(
    U2: np.ndarray,
    freq: np.ndarray,
    sigma_y: np.ndarray | None,
    model: str,
    band: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Fit one band with the given model.

    model: "sqrt_offset" -> a*sqrt(b-U2)+c; "sqrt_only" -> a*sqrt(b-U2).
    band: band name for per-band initial guess lookup (FIT_SQRT_OFFSET_INIT[band], etc.).
    Returns (popt, pcov) or (None, None) on failure.
    """
    U2 = np.asarray(U2, dtype=float)
    freq = np.asarray(freq, dtype=float)
    if U2.size < 2 or freq.size != U2.size:
        return None, None
    u_max = float(np.max(U2))
    b_auto = u_max + 0.05
    b_min = u_max + 1e-6
    if sigma_y is not None:
        sigma_y = np.asarray(sigma_y, dtype=float)
        if sigma_y.size != U2.size or np.any(sigma_y <= 0):
            sigma_y = None
    if model == "sqrt_offset":
        init = FIT_SQRT_OFFSET_INIT.get(band, FIT_SQRT_OFFSET_INIT.get(None, (50.0, None, 0.0)))
        if not isinstance(init, (tuple, list)) or len(init) < 3:
            init = (50.0, None, 0.0)
        a0 = init[0] if init[0] is not None else 50.0
        b0 = init[1] if init[1] is not None else b_auto
        c0 = init[2] if len(init) > 2 and init[2] is not None else 0.0
        p0 = (float(a0), float(b0), float(c0))
        bounds = ([1e-6, b_min, -np.inf], [np.inf, np.inf, np.inf])
        try:
            popt, pcov = curve_fit(
                _model_sqrt_offset, U2, freq,
                p0=p0, bounds=bounds, maxfev=5000,
                sigma=sigma_y, absolute_sigma=(sigma_y is not None),
            )
            return popt, pcov
        except Exception:
            return None, None
    elif model == "sqrt_only":
        init = FIT_SQRT_ONLY_INIT.get(band, FIT_SQRT_ONLY_INIT.get(None, (50.0, None)))
        if not isinstance(init, (tuple, list)) or len(init) < 2:
            init = (50.0, None)
        a0 = init[0] if init[0] is not None else 50.0
        b0 = init[1] if init[1] is not None else b_auto
        p0 = (float(a0), float(b0))
        bounds = ([1e-6, b_min], [np.inf, np.inf])
        try:
            popt, pcov = curve_fit(
                _model_sqrt_only, U2, freq,
                p0=p0, bounds=bounds, maxfev=5000,
                sigma=sigma_y, absolute_sigma=(sigma_y is not None),
            )
            return popt, pcov
        except Exception:
            return None, None
    return None, None


def _get_band_fit_curves(
    df: pd.DataFrame,
    branch_col: str,
    x_col: str,
    y_col: str,
    err_col: str | None,
    model: str,
    bands: list[str],
    u2_line: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """
    For each band, fit the model and evaluate on u2_line.
    Returns dict band -> (fitted_curve_on_u2_line, popt or None).
    """
    out = {}
    for band in bands:
        sub = df[df[branch_col] == band]
        if sub.empty or len(sub) < 2:
            out[band] = (np.full_like(u2_line, np.nan), None)
            continue
        U2 = sub[x_col].values.astype(float)
        freq = sub[y_col].values.astype(float)
        sigma_y = sub[err_col].values.astype(float) if err_col and err_col in sub.columns else None
        if sigma_y is not None and np.any(sigma_y <= 0):
            sigma_y = None
        popt, _ = fit_band_model(U2, freq, sigma_y, model, band=band)
        if popt is None:
            out[band] = (np.full_like(u2_line, np.nan), None)
            continue
        if model == "sqrt_offset":
            curve = _model_sqrt_offset(u2_line, *popt)
        else:
            curve = _model_sqrt_only(u2_line, *popt)
        out[band] = (curve, popt)
    return out


def _band_y_span(
    df: pd.DataFrame,
    branch_col: str,
    bands: list[str],
    fit_curves: dict[str, tuple[np.ndarray, np.ndarray | None]],
    u_line: np.ndarray,
    x_col: str,
    y_col: str,
    margin_frac: float = 0.05,
) -> float:
    """Return a common y-axis span so all individual band plots use the same span. Uses max of per-band (data + curve) ranges."""
    u_min_g, u_max_g = float(np.min(u_line)), float(np.max(u_line))
    span_u = u_max_g - u_min_g or 1.0
    margin = max(margin_frac * span_u, 0.005)
    max_span = 0.0
    for band in bands:
        sub = df[df[branch_col] == band]
        if sub.empty:
            continue
        u_lo = float(sub[x_col].min()) - margin
        u_hi = float(sub[x_col].max()) + margin
        mask = (u_line >= u_lo) & (u_line <= u_hi)
        curve, _ = fit_curves.get(band, (np.full_like(u_line, np.nan), None))
        curve_band = curve[mask] if np.any(mask) else np.array([np.nan])
        y_curve_lo = np.nanmin(curve_band) if np.any(np.isfinite(curve_band)) else None
        y_curve_hi = np.nanmax(curve_band) if np.any(np.isfinite(curve_band)) else None
        y_data_lo = float(sub[y_col].min())
        y_data_hi = float(sub[y_col].max())
        y_lo = y_data_lo if y_curve_lo is None else min(y_data_lo, y_curve_lo)
        y_hi = y_data_hi if y_curve_hi is None else max(y_data_hi, y_curve_hi)
        s = y_hi - y_lo
        if s > 0 and np.isfinite(s):
            max_span = max(max_span, s)
    if max_span <= 0:
        return 10.0
    return max_span * 1.06


def plot_band_with_fit(
    df: pd.DataFrame,
    band: str,
    branch_col: str,
    model: str,
    x_col: str = "U2",
    y_col: str = "mu",
    errorbar_source: str | None = None,
    plot_width_mode: str | None = None,
    title: str | None = None,
    mode_label: str = "",
    popt: np.ndarray | None = None,
    ylim_span: float | None = None,
) -> plt.Figure:
    """One band: data with error bars and fitted curve. Single figure. popt: optional pre-fitted params. ylim_span: fix y-axis span (same for all individual plots)."""
    subset = df[df[branch_col] == band]
    if subset.empty or len(subset) < 2:
        fig, ax = plt.subplots()
        ax.set_title(title or f"{mode_label} {band} — {_MODEL_TITLE.get(model, model)}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        return fig
    err_src = errorbar_source if errorbar_source is not None else ERRORBAR_SOURCE
    width_mode = plot_width_mode if plot_width_mode is not None else PLOT_WIDTH_MODE
    err_col = "sigma_group" if err_src == "group" else "sigma"
    if err_col not in subset.columns:
        err_col = "sigma"
    err = _get_errorbar_array(subset, err_src, width_mode)
    U2 = subset[x_col].values.astype(float)
    freq = subset[y_col].values.astype(float)
    if popt is None:
        sigma_y = subset[err_col].values.astype(float) if err_col in subset.columns else None
        popt, _ = fit_band_model(U2, freq, sigma_y, model, band=band)
    fig, ax = plt.subplots()
    ax.errorbar(subset[x_col], subset[y_col], yerr=err, fmt="o", capsize=2, alpha=0.7, ecolor="k", capthick=0.5)
    u_min, u_max = float(U2.min()), float(U2.max())
    u_margin = max(0.02 * (u_max - u_min), 0.005)
    u_smooth = np.linspace(u_min - u_margin, u_max + u_margin, 200)
    if popt is not None:
        if model == "sqrt_offset":
            y_smooth = _model_sqrt_offset(u_smooth, *popt)
        else:
            y_smooth = _model_sqrt_only(u_smooth, *popt)
        formula = _format_fit_formula(popt, model)
        ax.plot(u_smooth, y_smooth, "-", color="C1", label=formula if formula else "fit")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{mode_label} {band} — {_MODEL_TITLE.get(model, model)}")
    ax.legend(loc="best")
    if ylim_span is not None and ylim_span > 0:
        y_data_lo = float(subset[y_col].min())
        y_data_hi = float(subset[y_col].max())
        if popt is not None and np.any(np.isfinite(y_smooth)):
            y_data_lo = min(y_data_lo, float(np.nanmin(y_smooth)))
            y_data_hi = max(y_data_hi, float(np.nanmax(y_smooth)))
        center = (y_data_lo + y_data_hi) / 2.0
        ax.set_ylim(center - ylim_span / 2.0, center + ylim_span / 2.0)
    fig.tight_layout()
    return fig


def plot_all_bands_with_fits(
    df: pd.DataFrame,
    branch_col: str,
    model: str,
    x_col: str = "U2",
    y_col: str = "mu",
    errorbar_source: str | None = None,
    plot_width_mode: str | None = None,
    title: str | None = None,
    mode_label: str = "",
    fit_curves: dict[str, tuple[np.ndarray, np.ndarray | None]] | None = None,
) -> plt.Figure:
    """All bands: all points (colored by band) and all fitted curves. Single figure. fit_curves: optional pre-fitted."""
    if branch_col not in df.columns or df.empty:
        fig, ax = plt.subplots()
        ax.set_title(title or f"{mode_label} all bands — {_MODEL_TITLE.get(model, model)}")
        return fig
    bands = [b for b in BAND_ORDER if (df[branch_col] == b).any()]
    bands += [b for b in df[branch_col].dropna().unique() if b not in bands]
    if not bands:
        fig, ax = plt.subplots()
        ax.set_title(title or f"{mode_label} all bands — {_MODEL_TITLE.get(model, model)}")
        return fig
    err_src = errorbar_source if errorbar_source is not None else ERRORBAR_SOURCE
    width_mode = plot_width_mode if plot_width_mode is not None else PLOT_WIDTH_MODE
    err_col = "sigma_group" if err_src == "group" else "sigma"
    if err_col not in df.columns:
        err_col = "sigma"
    u_all = df[x_col].values.astype(float)
    u_min, u_max = float(np.min(u_all)), float(np.max(u_all))
    u_margin = max(0.02 * (u_max - u_min), 0.005)
    u_line = np.linspace(u_min - u_margin, u_max + u_margin, 300)
    if fit_curves is None:
        fit_curves = _get_band_fit_curves(df, branch_col, x_col, y_col, err_col, model, bands, u_line)
    fig, ax = plt.subplots()
    cmap = plt.colormaps["tab10"]
    for i, band in enumerate(bands):
        sub = df[df[branch_col] == band]
        if sub.empty:
            continue
        err = _get_errorbar_array(sub, err_src, width_mode)
        color = cmap(i % 10)
        curve, popt = fit_curves.get(band, (np.full_like(u_line, np.nan), None))
        formula = _format_fit_formula(popt, model) if popt is not None else ""
        leg_label = f"{band}: {formula}" if formula else band
        ax.errorbar(sub[x_col], sub[y_col], yerr=err, fmt="o", capsize=2, alpha=0.7, label=leg_label, color=color, ecolor="k", capthick=0.5)
        if np.any(np.isfinite(curve)):
            ax.plot(u_line, curve, "-", color=color, label=None)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{mode_label} all bands — {_MODEL_TITLE.get(model, model)}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def generate_band_fit_plots(
    df: pd.DataFrame,
    mode_label: str,
    save_dir: str | Path,
    branch_col: str = "band",
    x_col: str = "U2",
    y_col: str = "mu",
    dpi: int = 150,
    bands: list[str] | None = None,
) -> list[Path]:
    """
    For each band and for "all", plot data with error bars and fitted curves.
    Saves separate images for model1 (a*sqrt(U2-b)+c) and model2 (a*sqrt(U2-b)).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if branch_col not in df.columns or df.empty:
        return []
    if bands is None:
        bands = [b for b in BAND_ORDER if (df[branch_col] == b).any()]
        bands += [b for b in df[branch_col].dropna().unique() if b not in bands]
    err_col = "sigma_group" if ERRORBAR_SOURCE == "group" else "sigma"
    if err_col not in df.columns:
        err_col = "sigma"
    u_all = df[x_col].values.astype(float)
    u_min, u_max = float(np.min(u_all)), float(np.max(u_all))
    u_margin = max(0.02 * (u_max - u_min), 0.005)
    u_line = np.linspace(u_min - u_margin, u_max + u_margin, 300)

    fit_rows: list[dict] = []
    saved = []
    for model_name, model_key in [("model1_sqrt_offset", "sqrt_offset"), ("model2_sqrt_only", "sqrt_only")]:
        fit_curves = _get_band_fit_curves(df, branch_col, x_col, y_col, err_col, model_key, bands, u_line)
        for band in bands:
            curve, popt = fit_curves.get(band, (np.full_like(u_line, np.nan), None))
            if popt is not None:
                row = {"mode": mode_label, "model": model_key, "band": band, "a": popt[0], "b": popt[1]}
                if model_key == "sqrt_offset" and len(popt) >= 3:
                    row["c"] = popt[2]
                else:
                    row["c"] = np.nan
                fit_rows.append(row)
        ylim_span = _band_y_span(df, branch_col, bands, fit_curves, u_line, x_col, y_col)
        for band in bands:
            if not (df[branch_col] == band).any():
                continue
            _, popt = fit_curves.get(band, (None, None))
            fig = plot_band_with_fit(df, band, branch_col, model_key, x_col=x_col, y_col=y_col, mode_label=mode_label, popt=popt, ylim_span=ylim_span)
            path = save_dir / f"fit_{model_name}_{mode_label}_{band}.png"
            path = path.with_name(path.name.replace(" ", "").replace("—", ""))
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
        fig = plot_all_bands_with_fits(df, branch_col, model_key, x_col=x_col, y_col=y_col, mode_label=mode_label, fit_curves=fit_curves)
        path = save_dir / f"fit_{model_name}_{mode_label}_all.png"
        path = path.with_name(path.name.replace(" ", "").replace("—", ""))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    if fit_rows:
        fit_df = pd.DataFrame(fit_rows)
        csv_path = save_dir / f"band_fit_{mode_label}.csv"
        fit_df.to_csv(csv_path, index=False)
        saved.append(csv_path)
    return saved


def _get_branch_order(branch_col: str, df: pd.DataFrame) -> list:
    """Return ordered list of unique branch values for consistent colors."""
    if not branch_col:
        return []
    uniq = df[branch_col].dropna().unique().tolist()
    ordered = [b for b in BAND_ORDER if b in uniq]
    ordered += [b for b in uniq if b not in BAND_ORDER]
    return ordered


# ---------------------------------------------------------------------------
# Scatter plots by branch
# ---------------------------------------------------------------------------

def plot_lost_branches(
    df: pd.DataFrame,
    x_col: str = "U2",
    y_col: str = "mu",
    branch_col: str | None = None,
    ax: plt.Axes | None = None,
    legend: bool = True,
    title: str = "Cleaned averaged lost — by branch",
    errorbar_source: str | None = None,
    plot_width_mode: str | None = None,
) -> plt.Figure:
    """
    Scatter plot of cleaned lost data, with points colored by branch (frequency band).
    Optional y-error bars from line width (sigma or sigma_group).

    Parameters
    ----------
    errorbar_source : "peak" | "group", optional
        Use selected peak's sigma or group weighted-average sigma. Default from config.
    plot_width_mode : "individual" | "averaged", optional
        Per-point error bar or constant mean. Default from config.
    """
    if df.empty:
        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots()
        ax_.set_title(title)
        return ax_.figure

    err_src = errorbar_source if errorbar_source is not None else ERRORBAR_SOURCE
    width_mode = plot_width_mode if plot_width_mode is not None else PLOT_WIDTH_MODE
    branch_col = branch_col or _branch_column(df)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if not branch_col:
        err = _get_errorbar_array(df, err_src, width_mode)
        ax.errorbar(
            df[x_col], df[y_col], yerr=err, fmt="o", capsize=2, alpha=0.7,
            label="lost", color="C0", ecolor="k", capthick=0.5,
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        return fig

    branches = _get_branch_order(branch_col, df)
    cmap = plt.colormaps["tab10"]
    colors = {b: cmap(i % 10) for i, b in enumerate(branches)}

    for br in branches:
        subset = df[df[branch_col] == br]
        if subset.empty:
            continue
        err = _get_errorbar_array(subset, err_src, width_mode)
        ax.errorbar(
            subset[x_col],
            subset[y_col],
            yerr=err,
            fmt="o",
            capsize=2,
            alpha=0.7,
            label=str(br),
            color=colors[br],
            ecolor="k",
            capthick=0.5,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    if legend:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_trapped_branches(
    df: pd.DataFrame,
    x_col: str = "U2",
    y_col: str = "mu",
    branch_col: str | None = None,
    ax: plt.Axes | None = None,
    legend: bool = True,
    title: str = "Cleaned averaged trapped — by branch",
    errorbar_source: str | None = None,
    plot_width_mode: str | None = None,
) -> plt.Figure:
    """
    Scatter plot of cleaned trapped data, with points colored by branch.
    Optional y-error bars from line width (sigma or sigma_group).
    """
    if df.empty:
        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots()
        ax_.set_title(title)
        return ax_.figure

    err_src = errorbar_source if errorbar_source is not None else ERRORBAR_SOURCE
    width_mode = plot_width_mode if plot_width_mode is not None else PLOT_WIDTH_MODE
    branch_col = branch_col or _branch_column(df)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if not branch_col:
        err = _get_errorbar_array(df, err_src, width_mode)
        ax.errorbar(
            df[x_col], df[y_col], yerr=err, fmt="o", capsize=2, alpha=0.7,
            label="trapped", color="C0", ecolor="k", capthick=0.5,
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        return fig

    branches = _get_branch_order(branch_col, df)
    if not branches:
        branches = df[branch_col].dropna().unique().tolist()
    cmap = plt.colormaps["tab10"]
    colors = {b: cmap(i % 10) for i, b in enumerate(branches)}

    for br in branches:
        subset = df[df[branch_col] == br]
        if subset.empty:
            continue
        err = _get_errorbar_array(subset, err_src, width_mode)
        ax.errorbar(
            subset[x_col],
            subset[y_col],
            yerr=err,
            fmt="o",
            capsize=2,
            alpha=0.7,
            label=str(br),
            color=colors[br],
            ecolor="k",
            capthick=0.5,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    if legend:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Single-band plots (one figure per band)
# ---------------------------------------------------------------------------

def plot_lost_band(
    df: pd.DataFrame,
    band: str,
    x_col: str = "U2",
    y_col: str = "mu",
    branch_col: str = "band",
    ax: plt.Axes | None = None,
    title: str | None = None,
    errorbar_source: str | None = None,
    plot_width_mode: str | None = None,
) -> plt.Figure:
    """
    Scatter plot of cleaned lost data for a single band only. Optional y-error bars.
    """
    if not branch_col or branch_col not in df.columns:
        subset = df
    else:
        subset = df[df[branch_col] == band]
    if subset.empty:
        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots()
        ax_.set_title(title or f"Cleaned averaged lost — {band}")
        return ax_.figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    err_src = errorbar_source if errorbar_source is not None else ERRORBAR_SOURCE
    width_mode = plot_width_mode if plot_width_mode is not None else PLOT_WIDTH_MODE
    err = _get_errorbar_array(subset, err_src, width_mode)
    ax.errorbar(
        subset[x_col], subset[y_col], yerr=err, fmt="o", capsize=2, alpha=0.7,
        ecolor="k", capthick=0.5,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title if title is not None else f"Cleaned averaged lost — {band}")
    fig.tight_layout()
    return fig


def plot_trapped_band(
    df: pd.DataFrame,
    band: str,
    x_col: str = "U2",
    y_col: str = "mu",
    branch_col: str = "band",
    ax: plt.Axes | None = None,
    title: str | None = None,
    errorbar_source: str | None = None,
    plot_width_mode: str | None = None,
) -> plt.Figure:
    """
    Scatter plot of cleaned trapped data for a single band only. Optional y-error bars.
    """
    if not branch_col or branch_col not in df.columns:
        subset = df
    else:
        subset = df[df[branch_col] == band]
    if subset.empty:
        fig, ax_ = (ax.figure, ax) if ax is not None else plt.subplots()
        ax_.set_title(title or f"Cleaned averaged trapped — {band}")
        return ax_.figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    err_src = errorbar_source if errorbar_source is not None else ERRORBAR_SOURCE
    width_mode = plot_width_mode if plot_width_mode is not None else PLOT_WIDTH_MODE
    err = _get_errorbar_array(subset, err_src, width_mode)
    ax.errorbar(
        subset[x_col], subset[y_col], yerr=err, fmt="o", capsize=2, alpha=0.7,
        ecolor="k", capthick=0.5,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title if title is not None else f"Cleaned averaged trapped — {band}")
    fig.tight_layout()
    return fig


def generate_lost_band_plots(
    df: pd.DataFrame,
    save_dir: str | Path,
    branch_col: str = "band",
    x_col: str = "U2",
    y_col: str = "mu",
    dpi: int = 150,
    bands: list[str] | None = None,
) -> list[Path]:
    """
    Plot each band individually and save one image per band (lost data).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned lost data.
    save_dir : str or Path
        Directory to save PNG files.
    branch_col : str
        Column holding band labels (default "band").
    x_col, y_col : str
        Axis columns.
    dpi : int
        Resolution for saved figures.
    bands : list of str, optional
        Bands to plot. If None, uses bands present in df (in BAND_ORDER order).

    Returns
    -------
    list of Path
        Paths to saved files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if branch_col not in df.columns:
        return []
    if bands is None:
        bands = [b for b in BAND_ORDER if (df[branch_col] == b).any()]
        bands += [b for b in df[branch_col].dropna().unique() if b not in bands]
    saved = []
    for band in bands:
        if (df[branch_col] == band).any():
            fig = plot_lost_band(df, band, x_col=x_col, y_col=y_col, branch_col=branch_col)
            path = save_dir / f"cleaned_lost_{band}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    return saved


def generate_trapped_band_plots(
    df: pd.DataFrame,
    save_dir: str | Path,
    branch_col: str = "band",
    x_col: str = "U2",
    y_col: str = "mu",
    dpi: int = 150,
    bands: list[str] | None = None,
) -> list[Path]:
    """
    Plot each band individually and save one image per band (trapped data).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned trapped data.
    save_dir : str or Path
        Directory to save PNG files.
    branch_col : str
        Column holding band labels (default "band").
    x_col, y_col : str
        Axis columns.
    dpi : int
        Resolution for saved figures.
    bands : list of str, optional
        Bands to plot. If None, uses bands present in df (in BAND_ORDER order).

    Returns
    -------
    list of Path
        Paths to saved files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if branch_col not in df.columns:
        # No band column: save a single "all" plot if there is data
        if not df.empty:
            fig = plot_trapped_band(df, band="all", branch_col="", title="Cleaned averaged trapped")
            path = save_dir / "cleaned_trapped_all.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            return [path]
        return []
    if bands is None:
        bands = [b for b in BAND_ORDER if (df[branch_col] == b).any()]
        bands += [b for b in df[branch_col].dropna().unique() if b not in bands]
    saved = []
    for band in bands:
        if (df[branch_col] == band).any():
            fig = plot_trapped_band(df, band, x_col=x_col, y_col=y_col, branch_col=branch_col)
            path = save_dir / f"cleaned_trapped_{band}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# Main: generate individual plots
# ---------------------------------------------------------------------------

def main(
    data_dir: str | Path | None = None,
    save_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """
    Load cleaned lost and trapped CSVs and generate individual scatter plots by branch.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing cleaned_averaged_lost.csv and cleaned_averaged_trapped.csv.
        If None, uses the script directory.
    save_dir : str or Path, optional
        Directory to save plot images. If None, plots are not saved to disk.
    show : bool
        If True, call plt.show() to display plots (default False).
    """
    data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent
    lost_path = data_dir / "cleaned_averaged_lost.csv"
    trapped_path = data_dir / "cleaned_averaged_trapped.csv"

    df_lost = read_cleaned_lost(lost_path)
    df_trapped = read_cleaned_trapped(trapped_path)

    # Individual plot: lost (all branches)
    fig_lost = plot_lost_branches(df_lost)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_lost.savefig(save_dir / "cleaned_lost_by_branch.png", dpi=150, bbox_inches="tight")
        plt.close(fig_lost)
        generate_lost_band_plots(df_lost, save_dir)
        generate_band_fit_plots(df_lost, mode_label="lost", save_dir=save_dir)

    # Individual plot: trapped (all branches)
    fig_trapped = plot_trapped_branches(df_trapped)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_trapped.savefig(save_dir / "cleaned_trapped_by_branch.png", dpi=150, bbox_inches="tight")
        plt.close(fig_trapped)
        generate_trapped_band_plots(df_trapped, save_dir)
        generate_band_fit_plots(df_trapped, mode_label="trapped", save_dir=save_dir)

    if show:
        plt.show()


if __name__ == "__main__":
    main(data_dir=DATA_DIR, save_dir=SAVE_DIR, show=SHOW_PLOTS)
