"""
Recover split_x (segment cut point in MHz) and baseline/y_min/y_max from existing data.
Use when CSV was exported before these were saved, or to fill without re-running fitting.

- split_x: from peak params (averaged = largest gap in mus; per_scan = L/R boundary).
- baseline_*, y_min, y_max: from raw data only — load arrays, run split_experiment, compute (no fitting).
"""
import os
import numpy as np
import pandas as pd


def recover_split_x_averaged(row, mu_cols=None):
    """Recover split_x from merged peak format (amp1, mu1, sigma1, ...).
    Finds largest gap in sorted mus; split lies in that gap.
    Returns float or None if recovery fails.
    """
    if mu_cols is None:
        mu_cols = [f"mu{i}" for i in range(1, 19)]
    mus = []
    for c in mu_cols:
        if c in row and pd.notna(row[c]):
            mus.append(float(row[c]))
    if len(mus) < 2:
        return None
    mus = np.sort(mus)
    gaps = mus[1:] - mus[:-1]
    idx = np.argmax(gaps)
    return float((mus[idx] + mus[idx + 1]) / 2)


def recover_split_x_per_scan(row, n_peaks=9):
    """Recover split_x from per-scan format (mu1_L, mu1_R, ...).
    split_x lies between max(left mus) and min(right mus).
    Returns float or None if recovery fails.
    """
    left_mus = []
    right_mus = []
    for i in range(1, n_peaks + 1):
        for suffix, lst in [("_L", left_mus), ("_R", right_mus)]:
            c = f"mu{i}{suffix}"
            if c in row and pd.notna(row[c]):
                lst.append(float(row[c]))
    if not left_mus or not right_mus:
        return None
    max_left = max(left_mus)
    min_right = min(right_mus)
    if max_left >= min_right:
        return float((max_left + min_right) / 2)  # overlap: use midpoint
    return float((max_left + min_right) / 2)


BASELINE_COLS = [
    "baseline_left_mean", "baseline_left_std", "baseline_right_mean", "baseline_right_std",
    "y_min", "y_max",
]


def add_baseline_cols_if_missing(df):
    """Add baseline and y_min/y_max columns as NaN when missing (cannot recover from peak params)."""
    for c in BASELINE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def add_split_x_to_dataframe(df, format="averaged"):
    """Add split_x_MHz column to DataFrame, recovering from peak params if missing.
    format: "averaged" or "per_scan"
    Modifies df in place, returns df.
    """
    if "split_x_MHz" in df.columns and df["split_x_MHz"].notna().any():
        return df
    recover = recover_split_x_averaged if format == "averaged" else recover_split_x_per_scan
    values = []
    for _, row in df.iterrows():
        v = recover(row)
        values.append(v)
    df["split_x_MHz"] = values
    add_baseline_cols_if_missing(df)
    return df


# -----------------------------------------------------------------------------
# Recover baseline and y_min/y_max from raw data (no fitting)
# -----------------------------------------------------------------------------


def _compute_baseline_stats_from_xy(x, y_trapped, y_lost, split_res):
    """Compute per-mode baseline_left/right_mean/std and y_min/y_max from split result.
    Same logic as analysis.py; no fitting. Returns dict mode -> {baseline_left_mean, ...}.
    """
    import numpy as np
    modes = ["lost", "trapped"]
    rs = np.asarray(y_trapped)
    rl = np.asarray(y_lost)
    split_x = split_res.get("split_x")
    sidx = split_res.get("split_idx")
    n = len(x)
    if split_x is not None and sidx is not None:
        left_idx = np.arange(n) <= sidx
        right_idx = np.arange(n) > sidx
    else:
        mid = n // 2
        left_idx = np.arange(n) <= mid
        right_idx = np.arange(n) > mid

    def _mean_std(v):
        if len(v) >= 2:
            return float(np.mean(v)), float(np.std(v))
        if len(v) == 1:
            return float(np.mean(v)), np.nan
        return np.nan, np.nan

    baseline_stats = {}
    for mode, ym, bl_mask in [
        ("trapped", rs, split_res.get("baseline_mask_t")),
        ("lost", rl, split_res.get("baseline_mask_l")),
    ]:
        if bl_mask is None or len(bl_mask) != n:
            baseline_stats[mode] = {
                "baseline_left_mean": np.nan, "baseline_left_std": np.nan,
                "baseline_right_mean": np.nan, "baseline_right_std": np.nan,
                "y_min": float(np.min(ym)), "y_max": float(np.max(ym)),
            }
            continue
        bl = np.asarray(bl_mask, dtype=bool)
        vals_l = ym[bl & left_idx]
        vals_r = ym[bl & right_idx]
        ml, sl = _mean_std(vals_l)
        mr, sr = _mean_std(vals_r)
        baseline_stats[mode] = {
            "baseline_left_mean": ml, "baseline_left_std": sl,
            "baseline_right_mean": mr, "baseline_right_std": sr,
            "y_min": float(np.min(ym)), "y_max": float(np.max(ym)),
        }
    return baseline_stats


def recover_baseline_for_timestamp(timestamp, data_root=None):
    """Load raw data for one run, run split_experiment, return baseline stats (no fitting).
    Returns dict: {"lost": {...}, "trapped": {...}} with baseline_* and y_min, y_max.
    """
    from data_io import load_data, get_data_root
    data_root = get_data_root(data_root)
    x, ys = load_data(timestamp, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
    y_trapped = np.asarray(ys["ratio_signal"])
    y_lost = np.asarray(ys["ratio_lost"])
    from split_experiment import split_experiment
    split_res = split_experiment(x, y_trapped, y_lost)
    return _compute_baseline_stats_from_xy(x, y_trapped, y_lost, split_res)


def recover_baseline_for_averaged(ts_list, data_root=None):
    """Load and average raw data for a group, run split_experiment, return baseline stats (no fitting).
    Returns dict: {"lost": {...}, "trapped": {...}}.
    """
    from data_io import load_and_average_group, get_data_root
    data_root = get_data_root(data_root)
    if not ts_list:
        return {
            "lost": {"baseline_left_mean": np.nan, "baseline_left_std": np.nan,
                     "baseline_right_mean": np.nan, "baseline_right_std": np.nan, "y_min": np.nan, "y_max": np.nan},
            "trapped": {"baseline_left_mean": np.nan, "baseline_left_std": np.nan,
                        "baseline_right_mean": np.nan, "baseline_right_std": np.nan, "y_min": np.nan, "y_max": np.nan},
        }
    x, ys_avg = load_and_average_group(ts_list, data_root=data_root)
    y_trapped = np.asarray(ys_avg["ratio_signal"])
    y_lost = np.asarray(ys_avg["ratio_lost"])
    from split_experiment import split_experiment
    split_res = split_experiment(x, y_trapped, y_lost)
    return _compute_baseline_stats_from_xy(x, y_trapped, y_lost, split_res)


def add_baseline_recovery_to_dataframe(
    df, data_root=None, format="averaged", exclude_dates=None
):
    """Fill baseline_* and y_min, y_max by loading raw data and running split (no fitting).
    format: 'averaged' | 'per_scan'. Modifies df in place, returns df.
    For averaged, uses build_tickling_groups to map RF -> timestamps; exclude_dates applied.
    """
    from data_io import get_data_root
    from metadata import build_tickling_groups, filter_excluded_timestamps
    from config import EXCLUDE_DATES
    data_root = get_data_root(data_root)
    exclude_dates = exclude_dates if exclude_dates is not None else EXCLUDE_DATES
    add_baseline_cols_if_missing(df)

    rf_col = "RF_set_dBm" if "RF_set_dBm" in df.columns else "RF_amplitude"
    if rf_col not in df.columns:
        return df

    if format == "averaged":
        groups, _, _ = build_tickling_groups()
        rf_to_ts = {}
        for g in groups:
            ts_list = filter_excluded_timestamps(g["timestamps"], exclude_dates)
            if not ts_list:
                continue
            rf = g["RF_amplitude"]
            if rf is not None:
                rf_to_ts[float(rf)] = ts_list

        for rf_val, ts_list in rf_to_ts.items():
            try:
                stats = recover_baseline_for_averaged(ts_list, data_root=data_root)
            except Exception:
                continue
            for mode in ("lost", "trapped"):
                rf_vals = pd.to_numeric(df[rf_col], errors="coerce")
                mask = (df["mode"] == mode) & np.isclose(rf_vals, float(rf_val), rtol=0, atol=1e-6)
                if not mask.any():
                    continue
                for col in BASELINE_COLS:
                    val = stats.get(mode, {}).get(col)
                    if val is not None and (not isinstance(val, float) or not np.isnan(val)):
                        df.loc[mask, col] = val

    else:
        if "timestamp" not in df.columns:
            return df
        for ts in df["timestamp"].dropna().unique():
            if ts == "averaged" or not ts:
                continue
            try:
                stats = recover_baseline_for_timestamp(ts, data_root=data_root)
            except Exception:
                continue
            for mode in ("lost", "trapped"):
                mask = (df["timestamp"] == ts) & (df["mode"] == mode)
                if not mask.any():
                    continue
                for col in BASELINE_COLS:
                    val = stats.get(mode, {}).get(col)
                    if val is not None and (not isinstance(val, float) or not np.isnan(val)):
                        df.loc[mask, col] = val

    return df
