"""
Filter peaks within each group: keep only those that are composite local maxima
with sufficient normalized intensity and prominence.
"""
import numpy as np
from scipy.signal import find_peaks
from fitting_functions import gaussian_sum
from config import STEP_SIZE_MHZ


def _group_curve(x, peaks_subset, c0):
    """Evaluate c0 + sum of Gaussians for peaks in peaks_subset. peaks_subset = list of (amp, mu, sigma)."""
    if not peaks_subset:
        return np.full_like(x, c0)
    params = [c0] + [v for p in peaks_subset for v in p]
    return gaussian_sum(np.asarray(x, dtype=float), *params)


def filter_peaks_by_composite_local_max(
    row,
    groups_left,
    groups_right,
    intensity_thresh=0.02,
    prominence_thresh=0.02,
    n_grid=500,
    stepsize=None,
):
    """
    For each group, mark peaks as keep/drop based on:
    1. Is the peak a composite local maximum in the group's total fitting curve?
    2. Normalized intensity = amp / (y_max - y_min) >= intensity_thresh
    3. Normalized prominence >= prominence_thresh (waived for peak at group's global max)

    When two peaks are extremely close (|mu_i - mu_j| < raw step size), prefer higher intensity.
    The group's global maximum is always assigned to a peak; that peak bypasses prominence but not intensity.

    Returns a copy of the row with amp, mu, sigma set to NaN for dropped peaks.
    """
    import pandas as pd

    step_mhz = STEP_SIZE_MHZ if stepsize is None else float(stepsize)
    split_x = row["split_x_MHz"]
    mode = str(row.get("mode", "lost")).strip().lower()
    # lost: peaks (local max), trapped: dips (local min) -> use -y for trapped
    invert_for_peaks = mode == "trapped"
    c0 = row["c0"] if pd.notna(row.get("c0")) else 0.0
    y_min = row["y_min"]
    y_max = row["y_max"]
    y_range = float(y_max) - float(y_min)
    if y_range < 1e-12:
        y_range = 1.0

    out = row.copy()
    peaks_left, peaks_right = [], []
    col_left, col_right = [], []
    for i in range(1, 19):
        a, m, s = row.get(f"amp{i}"), row.get(f"mu{i}"), row.get(f"sigma{i}")
        if pd.isna(a) or pd.isna(m) or pd.isna(s):
            continue
        p = (float(a), float(m), float(s))
        if float(m) <= split_x:
            peaks_left.append(p)
            col_left.append(i)
        else:
            peaks_right.append(p)
            col_right.append(i)

    to_drop_left = set()
    to_drop_right = set()
    for grp in groups_left:
        if not grp:
            continue
        subset = [peaks_left[i] for i in grp]
        x_lo = min(p[1] - 5 * p[2] for p in subset)
        x_hi = max(p[1] + 5 * p[2] for p in subset)
        x_grid = np.linspace(x_lo, x_hi, n_grid)
        y_grid = _group_curve(x_grid, subset, c0)
        y_scan = -np.asarray(y_grid, dtype=float) if invert_for_peaks else np.asarray(y_grid, dtype=float)
        peaks_idx, props = find_peaks(y_scan, prominence=1e-12)
        prom = props["prominences"] if len(peaks_idx) > 0 else np.array([])
        # Group's "global best": max for lost (bumps), min for trapped (dips) -> prominence waiver
        if len(peaks_idx) > 0:
            vals = y_grid[peaks_idx]
            global_max_j = int(np.argmin(vals) if invert_for_peaks else np.argmax(vals))
        else:
            global_max_j = None
        # Each local max assigned to ONE peak. When two candidates are extremely close
        # (|mu_i-mu_j| < step), prefer higher intensity; else prefer closest to local max.
        assigned_left = set()
        global_max_gi_left = None
        if len(peaks_idx) > 0:
            for j in range(len(peaks_idx)):
                x_lm = x_grid[peaks_idx[j]]
                candidates = []
                for gi in grp:
                    amp_pk, mu_pk, sigma_pk = peaks_left[gi]
                    d = np.abs(x_lm - mu_pk)
                    thresh = max(0.5 * sigma_pk, 0.1)
                    if d <= thresh:
                        candidates.append((gi, d, np.abs(amp_pk), mu_pk))
                if not candidates:
                    continue
                # When any two candidates have |mu_i-mu_j| < step: prefer higher intensity
                if len(candidates) >= 2:
                    mus = [c[3] for c in candidates]
                    close_pair = any(
                        np.abs(mus[i] - mus[j]) < step_mhz
                        for i in range(len(candidates))
                        for j in range(i + 1, len(candidates))
                    )
                    if close_pair:
                        best_gi = max(candidates, key=lambda c: c[2])[0]
                    else:
                        best_gi = min(candidates, key=lambda c: c[1])[0]
                else:
                    best_gi = candidates[0][0]
                assigned_left.add(best_gi)
                if j == global_max_j:
                    global_max_gi_left = best_gi
        for gi in grp:
            if gi not in assigned_left:
                to_drop_left.add(gi)
                continue
            amp_pk, mu_pk, sigma_pk = peaks_left[gi]
            dist = np.abs(x_grid[peaks_idx] - mu_pk)
            best = np.argmin(dist)
            prom_norm = prom[best] / y_range
            intensity_norm = np.abs(amp_pk) / y_range
            if intensity_norm < intensity_thresh:
                to_drop_left.add(gi)
            elif gi != global_max_gi_left and prom_norm < prominence_thresh:
                to_drop_left.add(gi)

    for grp in groups_right:
        if not grp:
            continue
        subset = [peaks_right[i] for i in grp]
        x_lo = min(p[1] - 5 * p[2] for p in subset)
        x_hi = max(p[1] + 5 * p[2] for p in subset)
        x_grid = np.linspace(x_lo, x_hi, n_grid)
        y_grid = _group_curve(x_grid, subset, c0)
        y_scan = -np.asarray(y_grid, dtype=float) if invert_for_peaks else np.asarray(y_grid, dtype=float)
        peaks_idx, props = find_peaks(y_scan, prominence=1e-12)
        prom = props["prominences"] if len(peaks_idx) > 0 else np.array([])
        if len(peaks_idx) > 0:
            vals = y_grid[peaks_idx]
            global_max_j = int(np.argmin(vals) if invert_for_peaks else np.argmax(vals))
        else:
            global_max_j = None
        assigned_right = set()
        global_max_gi_right = None
        if len(peaks_idx) > 0:
            for j in range(len(peaks_idx)):
                x_lm = x_grid[peaks_idx[j]]
                candidates = []
                for gi in grp:
                    amp_pk, mu_pk, sigma_pk = peaks_right[gi]
                    d = np.abs(x_lm - mu_pk)
                    thresh = max(0.5 * sigma_pk, 0.1)
                    if d <= thresh:
                        candidates.append((gi, d, np.abs(amp_pk), mu_pk))
                if not candidates:
                    continue
                if len(candidates) >= 2:
                    mus = [c[3] for c in candidates]
                    close_pair = any(
                        np.abs(mus[i] - mus[j]) < step_mhz
                        for i in range(len(candidates))
                        for j in range(i + 1, len(candidates))
                    )
                    if close_pair:
                        best_gi = max(candidates, key=lambda c: c[2])[0]
                    else:
                        best_gi = min(candidates, key=lambda c: c[1])[0]
                else:
                    best_gi = candidates[0][0]
                assigned_right.add(best_gi)
                if j == global_max_j:
                    global_max_gi_right = best_gi
        for gi in grp:
            if gi not in assigned_right:
                to_drop_right.add(gi)
                continue
            amp_pk, mu_pk, sigma_pk = peaks_right[gi]
            dist = np.abs(x_grid[peaks_idx] - mu_pk)
            best = np.argmin(dist)
            prom_norm = prom[best] / y_range
            intensity_norm = np.abs(amp_pk) / y_range
            if intensity_norm < intensity_thresh:
                to_drop_right.add(gi)
            elif gi != global_max_gi_right and prom_norm < prominence_thresh:
                to_drop_right.add(gi)

    for gi in to_drop_left:
        col = col_left[gi]
        out[f"amp{col}"] = np.nan
        out[f"mu{col}"] = np.nan
        out[f"sigma{col}"] = np.nan
    for gi in to_drop_right:
        col = col_right[gi]
        out[f"amp{col}"] = np.nan
        out[f"mu{col}"] = np.nan
        out[f"sigma{col}"] = np.nan

    n_peaks_new = sum(1 for i in range(1, 19) if pd.notna(out.get(f"mu{i}")))
    if "n_peaks" in out:
        out["n_peaks"] = n_peaks_new

    return out
