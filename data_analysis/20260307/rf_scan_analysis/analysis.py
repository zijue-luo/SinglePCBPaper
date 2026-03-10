"""
Peak fitting analysis pipeline.
Splits spectrum at baseline, fits left/right segments separately, merges. Incremental peak adding
until R² threshold or AICc degrades. Uses split_experiment for init guesses when enabled.
"""
import numpy as np

from fitting_functions import (
    fit_n_peaks,
    _extract_mus_from_popt,
    _extract_sigmas_from_popt,
    gaussian_sum,
)
from config import (
    R2_THRESHOLD,
    STEP_SIZE_MHZ,
    SCAN_COUNT,
    MAX_N_PEAKS,
    N_JOBS,
    MAX_NFEV,
)
from data_io import load_data


def _fit_segment_incremental(
    x_seg, y_seg, mode, guesses_seg, stepsize, r2_gate, scan_count,
    max_n_peaks, n_jobs, max_nfev, return_history=False
):
    """Incremental fit on one segment. Add peaks until R² gate or AICc degrades. guesses_seg = [(amp, mu, sigma), ...]."""
    best = None
    incremental_history = [] if return_history else None
    if guesses_seg:
        n_peaks = min(len(guesses_seg), max_n_peaks)
        mu0 = [g[1] for g in guesses_seg[:n_peaks]]
        sig0 = [g[2] for g in guesses_seg[:n_peaks]]
    else:
        n_peaks = 1
        mu0 = [float(x_seg[np.argmax(y_seg)])] if mode == "lost" else [float(x_seg[np.argmin(y_seg)])]
        sig0 = None
    while n_peaks <= max_n_peaks:
        kwargs = dict(
            stepsize=stepsize,
            init_mus=mu0,
            scan_count=scan_count,
            n_jobs=n_jobs,
            init_sigmas=sig0,
            return_history=return_history,
        )
        if max_nfev is not None:
            kwargs["max_nfev"] = max_nfev
        result_n, history_n = fit_n_peaks(x_seg, y_seg, n_peaks, mode, **kwargs)
        if return_history:
            incremental_history.append({
                "n_peaks": int(n_peaks),
                "scan_count": int(scan_count),
                "seed_count": int(len(mu0)),
                "stage_best": result_n,
                "history": history_n,
            })
        if result_n is None or not result_n.get("ok") or result_n.get("popt") is None:
            break
        if best is None or result_n["aicc"] < best["aicc"]:
            best = result_n
        if result_n["r2"] > r2_gate:
            break
        if n_peaks >= 2 and result_n["aicc"] > best["aicc"]:
            break
        mu0 = _extract_mus_from_popt(result_n["popt"], n_peaks)
        sig0 = _extract_sigmas_from_popt(result_n["popt"], n_peaks)
        n_peaks += 1
    if return_history:
        return best, incremental_history
    return best


def _combine_segment_fits(x_full, y_full, best_left, best_right, mode):
    """Merge left and right segment fits into one. Recompute r2/aicc on full data."""
    if best_left is None and best_right is None:
        return None
    if best_left is None:
        return best_right
    if best_right is None:
        return best_left
    popt_l = best_left["popt"]
    popt_r = best_right["popt"]
    n_l = best_left["n_peaks"]
    n_r = best_right["n_peaks"]
    c0_merge = (popt_l[0] + popt_r[0]) / 2.0
    merged = [c0_merge] + list(popt_l[1 : 1 + 3 * n_l]) + list(popt_r[1 : 1 + 3 * n_r])
    n_tot = n_l + n_r
    yhat = gaussian_sum(x_full, *merged)
    r2 = 1.0 - np.sum((y_full - yhat) ** 2) / max(np.sum((y_full - np.mean(y_full)) ** 2), 1e-300)
    aicc = float("inf")
    k = 1 + 3 * n_tot
    rss = max(np.sum((y_full - yhat) ** 2), 1e-300)
    if len(y_full) > k + 1:
        aicc = len(y_full) * np.log(rss / len(y_full)) + 2 * k + (2 * k * (k + 1)) / (len(y_full) - k - 1)
    return {
        "n_peaks": n_tot,
        "popt": np.array(merged),
        "pcov": None,
        "r2": float(r2),
        "aicc": float(aicc),
        "ok": True,
    }


def analyze_fine_scan_from_arrays(
    x, y, stepsize=None, r2_gate=None, scan_count=None, max_n_peaks=None,
    n_jobs=None, max_nfev=None, use_find_peaks_init=True, return_fit_history=False
):
    """Split at baseline, fit left/right separately, combine."""
    stepsize = STEP_SIZE_MHZ if stepsize is None else stepsize
    r2_gate = R2_THRESHOLD if r2_gate is None else r2_gate
    scan_count = SCAN_COUNT if scan_count is None else scan_count
    max_n_peaks = MAX_N_PEAKS if max_n_peaks is None else max_n_peaks
    n_jobs = N_JOBS if n_jobs is None else n_jobs
    max_nfev = MAX_NFEV if max_nfev is None else max_nfev

    x = np.asarray(x, dtype=float)
    modes = ["lost", "trapped"]
    split_x = None
    guesses_by_mode = {}
    split_res = None
    if use_find_peaks_init:
        from split_experiment import split_experiment, initial_peaks_guess
        split_res = split_experiment(x, y["trapped"], y["lost"])
        split_x = split_res.get("split_x")
        g = initial_peaks_guess(x, y["trapped"], y["lost"])
        guesses_by_mode = {m: g[m] for m in modes if g.get(m)}

    if split_x is None:
        split_x = float(0.5 * (x[0] + x[-1]))  # fallback if split_experiment not used
    mask_l = x <= split_x
    mask_r = x > split_x

    # Compute left/right segment baseline stats and y min/max for data cleaning
    baseline_stats = {}
    if split_res is not None:
        sidx = split_res.get("split_idx")
        mask_t = split_res.get("baseline_mask_t")
        mask_lost = split_res.get("baseline_mask_l")
        rs, rl = np.asarray(y["trapped"]), np.asarray(y["lost"])
        n = len(x)
        left_idx = (np.arange(n) <= sidx) if sidx is not None else mask_l
        right_idx = (np.arange(n) > sidx) if sidx is not None else mask_r
        for mode, ym, bl_mask in [
            ("trapped", rs, mask_t),
            ("lost", rl, mask_lost),
        ]:
            if bl_mask is None or len(bl_mask) != n:
                baseline_stats[mode] = {
                    "baseline_left_mean": np.nan, "baseline_left_std": np.nan,
                    "baseline_right_mean": np.nan, "baseline_right_std": np.nan,
                    "y_min": float(np.min(ym)), "y_max": float(np.max(ym)),
                }
                continue
            bl = np.asarray(bl_mask, dtype=bool)
            left_bl = bl & left_idx
            right_bl = bl & right_idx
            vals_l = ym[left_bl]
            vals_r = ym[right_bl]
            def _mean_std(v):
                return (float(np.mean(v)), float(np.std(v))) if len(v) >= 2 else (float(np.mean(v)), np.nan) if len(v) == 1 else (np.nan, np.nan)
            ml, sl = _mean_std(vals_l)
            mr, sr = _mean_std(vals_r)
            baseline_stats[mode] = {
                "baseline_left_mean": ml, "baseline_left_std": sl,
                "baseline_right_mean": mr, "baseline_right_std": sr,
                "y_min": float(np.min(ym)), "y_max": float(np.max(ym)),
            }
    else:
        for mode in modes:
            ym = np.asarray(y[mode])
            baseline_stats[mode] = {
                "baseline_left_mean": np.nan, "baseline_left_std": np.nan,
                "baseline_right_mean": np.nan, "baseline_right_std": np.nan,
                "y_min": float(np.min(ym)), "y_max": float(np.max(ym)),
            }
    x_l = x[mask_l]
    x_r = x[mask_r]
    has_left = len(x_l) >= 10
    has_right = len(x_r) >= 10

    # Allocate scan_count by segment length so total remains scan_count
    n_l, n_r = len(x_l), len(x_r)
    if has_left and has_right and (n_l + n_r) > 0:
        scan_left = max(1, int(scan_count * n_l / (n_l + n_r)))
        scan_right = scan_count - scan_left
        if scan_right < 1:
            scan_right, scan_left = 1, scan_count - 1
    else:
        scan_left = scan_count if has_left else 0
        scan_right = scan_count if has_right else 0

    result = {}
    for mode in modes:
        ym = np.asarray(y[mode], dtype=float)
        y_l = ym[mask_l]
        y_r = ym[mask_r]
        guesses = guesses_by_mode.get(mode, [])
        g_left = [p for p in guesses if p[1] <= split_x]
        g_right = [p for p in guesses if p[1] > split_x]

        if has_left:
            if return_fit_history:
                best_l, hist_l = _fit_segment_incremental(
                    x_l, y_l, mode, g_left, stepsize, r2_gate, scan_left,
                    max_n_peaks, n_jobs, max_nfev, return_history=True
                )
            else:
                best_l = _fit_segment_incremental(
                    x_l, y_l, mode, g_left, stepsize, r2_gate, scan_left,
                    max_n_peaks, n_jobs, max_nfev, return_history=False
                )
                hist_l = []
        else:
            best_l, hist_l = None, []
        if has_right:
            if return_fit_history:
                best_r, hist_r = _fit_segment_incremental(
                    x_r, y_r, mode, g_right, stepsize, r2_gate, scan_right,
                    max_n_peaks, n_jobs, max_nfev, return_history=True
                )
            else:
                best_r = _fit_segment_incremental(
                    x_r, y_r, mode, g_right, stepsize, r2_gate, scan_right,
                    max_n_peaks, n_jobs, max_nfev, return_history=False
                )
                hist_r = []
        else:
            best_r, hist_r = None, []

        best = _combine_segment_fits(x, ym, best_l, best_r, mode)
        mode_history = []
        if return_fit_history:
            mode_history = [
                {
                    "segment": "left",
                    "split_x": float(split_x),
                    "x_min": float(np.min(x_l)) if len(x_l) else None,
                    "x_max": float(np.max(x_l)) if len(x_l) else None,
                    "n_points": int(len(x_l)),
                    "scan_count": int(scan_left),
                    "incremental": hist_l,
                    "best": best_l,
                },
                {
                    "segment": "right",
                    "split_x": float(split_x),
                    "x_min": float(np.min(x_r)) if len(x_r) else None,
                    "x_max": float(np.max(x_r)) if len(x_r) else None,
                    "n_points": int(len(x_r)),
                    "scan_count": int(scan_right),
                    "incremental": hist_r,
                    "best": best_r,
                },
            ]
        mode_extra = baseline_stats.get(mode, {})
        result[mode] = {
            "best": best, "all": [], "history": mode_history,
            "best_left": best_l, "best_right": best_r,
            **mode_extra,
        }
    result["split_x"] = split_x
    result["baseline_stats"] = baseline_stats
    return result


def analyze_fine_scan(
    timestamp, stepsize=None, r2_gate=None, scan_count=None, max_n_peaks=None,
    n_jobs=None, max_nfev=None, data_root=None, use_find_peaks_init=True, return_fit_history=False
):
    """Run peak fitting for one scan."""
    x, ys = load_data(timestamp, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
    y = {"lost": ys["ratio_lost"], "trapped": ys["ratio_signal"]}
    return analyze_fine_scan_from_arrays(
        x, y, stepsize=stepsize, r2_gate=r2_gate, scan_count=scan_count,
        max_n_peaks=max_n_peaks, n_jobs=n_jobs, max_nfev=max_nfev,
        use_find_peaks_init=use_find_peaks_init, return_fit_history=return_fit_history,
    )
