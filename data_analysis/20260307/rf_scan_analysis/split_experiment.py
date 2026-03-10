"""
Split tickling spectrum at baseline and generate initial peak guesses.
split_experiment: finds split_x where baseline crosses; repair masks for small spikes/drift.
initial_peaks_guess: find_peaks on residual; returns [(amp, mu, sigma), ...] per mode.
Depends on research/baseline_estimation for baseline_regions_mask.
Usage:
  from split_experiment import split_experiment, initial_peaks_guess

  result = split_experiment(x, ratio_signal, ratio_lost)
  split_x = result["split_x"]

  guesses = initial_peaks_guess(x, ratio_signal, ratio_lost)
  # guesses["trapped"] / ["lost"] = [(amp, mu, sigma), ...] per peak
"""
import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH = os.path.join(_THIS_DIR, "research")
if _RESEARCH not in sys.path:
    sys.path.insert(0, _RESEARCH)
sys.path.insert(0, _THIS_DIR)

from scipy.signal import find_peaks
from baseline_estimation import baseline_estimate, baseline_regions_mask


# Defaults for baseline and split
DEFAULT_METHOD = "iterative_2.0"
DEFAULT_N_SIGMA = 1.0

# Mask repair: fill small baseline gaps (spikes/drift)
REPAIR_CFG = {
    "small_spike_max_len": 2,
    "small_spike_max_dev_z": 2.2,
    "drift_min_len": 5,
    "drift_mean_dev_z": 1.1,
    "baseline_island_max_len": 2,
}

# Split at central baseline crossing
SPLIT_CFG = {
    "safe_margin_points": 8,
    "min_interval_points": 10,
    "edge_prefer_frac": 0.2,
}

# Conservative find_peaks for initial guess (no smoothing)
PEAK_CFG = {
    "k_sigma_height": 2.5,
    "k_sigma_prom": 3.5,
    "range_prom_frac": 0.22,
    "min_prom_abs": 0.010,
    "min_distance_mhz": 0.6,
    "min_width_pts": 1.0,
    "smooth_window_pts": 0,
    "smooth_polyorder": 2,
}


def _run_spans(mask):
    arr = np.asarray(mask, dtype=bool)
    if len(arr) == 0:
        return []
    spans = []
    s, val = 0, arr[0]
    for i in range(1, len(arr)):
        if arr[i] != val:
            spans.append((s, i - 1, bool(val)))
            s, val = i, arr[i]
    spans.append((s, len(arr) - 1, bool(val)))
    return spans


def _deviation_z(y, mode, baseline_mean, baseline_std, n_sigma):
    eps = max(float(baseline_std), 1e-8)
    if mode == "trapped":
        threshold = baseline_mean - n_sigma * baseline_std
        return np.maximum(0.0, (threshold - np.asarray(y)) / eps)
    threshold = baseline_mean + n_sigma * baseline_std
    return np.maximum(0.0, (np.asarray(y) - threshold) / eps)


def _repair_mask(raw_mask, dev_z, cfg):
    mask = np.asarray(raw_mask, dtype=bool).copy()
    spans = _run_spans(mask)
    for s, e, val in spans:
        if val:
            continue
        run_len = e - s + 1
        run_dev = dev_z[s : e + 1]
        max_dev = float(np.max(run_dev)) if run_len > 0 else 0.0
        mean_dev = float(np.mean(run_dev)) if run_len > 0 else 0.0
        if (run_len <= cfg["small_spike_max_len"] and max_dev <= cfg["small_spike_max_dev_z"]) or (
            run_len >= cfg["drift_min_len"] and mean_dev <= cfg["drift_mean_dev_z"]
        ):
            mask[s : e + 1] = True
    spans = _run_spans(mask)
    for idx, (s, e, val) in enumerate(spans):
        if not val or (e - s + 1) > cfg["baseline_island_max_len"]:
            continue
        left_sig = idx > 0 and not spans[idx - 1][2]
        right_sig = idx < len(spans) - 1 and not spans[idx + 1][2]
        if left_sig and right_sig:
            mask[s : e + 1] = False
    return mask


def _adaptive_smooth(y, cfg):
    w = int(cfg.get("smooth_window_pts", 0))
    if w <= 0 or w < 5:
        return np.asarray(y)
    from scipy.signal import savgol_filter
    n = len(y)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 5:
        return np.asarray(y)
    p = min(int(cfg["smooth_polyorder"]), w - 2)
    return savgol_filter(y, window_length=w, polyorder=p)


def _detect_mode_peaks_segment(x, y, mode, bl_mean, bl_std, cfg):
    """Return (indices, {}) for peaks in one segment."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 8:
        return np.array([], dtype=int), {}
    y_s = _adaptive_smooth(y, cfg)
    if mode == "trapped":
        amp = bl_mean - y_s
    else:
        amp = y_s - bl_mean
    amp = np.asarray(amp)
    amp_range = float(np.percentile(amp, 95) - np.percentile(amp, 5))
    sigma = max(float(bl_std), 1e-8)
    height = cfg["k_sigma_height"] * sigma
    prom = max(cfg["k_sigma_prom"] * sigma, cfg["range_prom_frac"] * max(amp_range, 0.0), cfg["min_prom_abs"])
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 0.5
    dist = max(1, int(round(cfg["min_distance_mhz"] / max(dx, 1e-6))))
    idx, _ = find_peaks(amp, height=height, prominence=prom, distance=dist, width=cfg["min_width_pts"])
    return idx, {}


def _sigma_from_hwhm(x, amp, peak_idx, amp_max):
    """Estimate sigma from half-width at half max. Returns float (MHz)."""
    x, amp = np.asarray(x), np.asarray(amp)
    n = len(x)
    half = amp_max / 2.0
    i = int(peak_idx)
    lo = i
    while lo > 0 and amp[lo] > half:
        lo -= 1
    hi = i
    while hi < n - 1 and amp[hi] > half:
        hi += 1
    if hi <= lo:
        return max(1.0, 2.0 * (float(x[1] - x[0]) if n > 1 else 0.5))
    hw = float(x[hi] - x[lo]) / 2.0  # half-width in MHz
    sigma = hw / np.sqrt(2.0 * np.log(2.0))  # HWHM = sigma * sqrt(2 ln 2)
    return max(0.3, min(sigma, 30.0))


def _detect_peak_positions(x, rs, rl):
    x, rs, rl = np.asarray(x), np.asarray(rs), np.asarray(rl)
    pr_rs = max(0.01, 0.08 * (np.percentile(rs, 90) - np.percentile(rs, 10)))
    pr_rl = max(0.01, 0.08 * (np.percentile(rl, 90) - np.percentile(rl, 10)))
    p_t, _ = find_peaks(-rs, prominence=pr_rs)
    p_l, _ = find_peaks(rl, prominence=pr_rl)
    idx = np.unique(np.concatenate([p_t, p_l]))
    return x[idx]


def _choose_split_index(x, combined_baseline_mask, peak_x, cfg):
    x = np.asarray(x)
    mask = np.asarray(combined_baseline_mask, dtype=bool)
    spans = _run_spans(mask)
    center = float(0.5 * (x[0] + x[-1]))
    candidates = []
    for s, e, val in spans:
        if not val:
            continue
        run_len = e - s + 1
        if run_len < cfg["min_interval_points"]:
            continue
        margin = max(cfg["safe_margin_points"], int(run_len * cfg["edge_prefer_frac"]))
        lo, hi = s + margin, e - margin
        if lo > hi:
            continue
        for i in range(lo, hi + 1):
            xi = float(x[i])
            n_left = int(np.sum(peak_x < xi))
            n_right = int(np.sum(peak_x > xi))
            edge_dist = min(i - s, e - i)
            score = (abs(n_left - n_right), abs(xi - center), -edge_dist, -run_len)
            candidates.append((score, i, n_left, n_right, run_len))
    if not candidates:
        return None, None
    candidates.sort(key=lambda t: t[0])
    _, idx, n_left, n_right, run_len = candidates[0]
    return idx, {"n_left": n_left, "n_right": n_right, "interval_len": run_len}


def split_experiment(
    x,
    ratio_signal,
    ratio_lost,
    method=DEFAULT_METHOD,
    n_sigma=DEFAULT_N_SIGMA,
    repair_cfg=None,
    split_cfg=None,
):
    """
    Find a safe baseline point to split the spectrum into left and right segments.

    Parameters
    ----------
    x : array-like
        Frequency axis (MHz).
    ratio_signal : array-like
        Trapped/loading ratio (trapped dips at resonance).
    ratio_lost : array-like
        Lost ratio (peaks at resonance).
    method : str
        Baseline method; default "iterative_2.0".
    n_sigma : float
        Baseline region threshold; default 1.0.
    repair_cfg : dict, optional
        Override REPAIR_CFG for baseline/signal mask repair.
    split_cfg : dict, optional
        Override SPLIT_CFG for split point selection.

    Returns
    -------
    dict
        split_x : float or None
            Split frequency (MHz).
        split_idx : int or None
            Index of split point.
        baseline_mask_t : ndarray
            Repaired baseline mask for trapped (True = baseline).
        baseline_mask_l : ndarray
            Repaired baseline mask for lost.
        baseline_info : dict
            trapped_mean, trapped_std, lost_mean, lost_std.
        peaks_left, peaks_right : int
            Peak counts left/right of split.
    """
    x = np.asarray(x, dtype=float)
    rs = np.asarray(ratio_signal, dtype=float)
    rl = np.asarray(ratio_lost, dtype=float)
    repair = repair_cfg if repair_cfg is not None else REPAIR_CFG
    split = split_cfg if split_cfg is not None else SPLIT_CFG

    bl = baseline_estimate(x, rs, rl, method=method)
    raw_t = baseline_regions_mask(x, rs, "trapped", bl["trapped_mean"], bl["trapped_std"], n_sigma=n_sigma)
    raw_l = baseline_regions_mask(x, rl, "lost", bl["lost_mean"], bl["lost_std"], n_sigma=n_sigma)

    dev_t = _deviation_z(rs, "trapped", bl["trapped_mean"], bl["trapped_std"], n_sigma)
    dev_l = _deviation_z(rl, "lost", bl["lost_mean"], bl["lost_std"], n_sigma)
    fix_t = _repair_mask(raw_t, dev_t, repair)
    fix_l = _repair_mask(raw_l, dev_l, repair)

    combined = fix_t & fix_l
    peak_x = _detect_peak_positions(x, rs, rl)
    split_idx, split_info = _choose_split_index(x, combined, peak_x, split)

    if split_idx is None:
        return {
            "split_x": None,
            "split_idx": None,
            "baseline_mask_t": fix_t,
            "baseline_mask_l": fix_l,
            "baseline_info": bl,
            "peaks_left": None,
            "peaks_right": None,
        }

    split_x = float(x[split_idx])
    n_left = split_info["n_left"]
    n_right = split_info["n_right"]

    return {
        "split_x": split_x,
        "split_idx": int(split_idx),
        "baseline_mask_t": fix_t,
        "baseline_mask_l": fix_l,
        "baseline_info": bl,
        "peaks_left": n_left,
        "peaks_right": n_right,
    }


def initial_peaks_guess(
    x,
    ratio_signal,
    ratio_lost,
    method=DEFAULT_METHOD,
    n_sigma=DEFAULT_N_SIGMA,
    peak_cfg=None,
):
    """
    Find peaks via split + conservative find_peaks and return per-peak initial
    guesses (amp, mu, sigma) for fitting.

    Returns
    -------
    dict
        trapped : list of (amp, mu, sigma)
        lost : list of (amp, mu, sigma)
    """
    x = np.asarray(x, dtype=float)
    rs = np.asarray(ratio_signal, dtype=float)
    rl = np.asarray(ratio_lost, dtype=float)
    cfg = peak_cfg if peak_cfg is not None else PEAK_CFG

    split = split_experiment(x, rs, rl, method=method, n_sigma=n_sigma)
    sidx = split["split_idx"]
    if sidx is None:
        sidx = len(x) // 2
    bl = split["baseline_info"]
    l_slice = slice(0, sidx + 1)
    r_slice = slice(sidx + 1, len(x))

    out = {"trapped": [], "lost": []}
    for mode, y, bl_mean, bl_std in [
        ("trapped", rs, bl["trapped_mean"], bl["trapped_std"]),
        ("lost", rl, bl["lost_mean"], bl["lost_std"]),
    ]:
        idx_l, _ = _detect_mode_peaks_segment(x[l_slice], y[l_slice], mode, bl_mean, bl_std, cfg)
        idx_r, _ = _detect_mode_peaks_segment(x[r_slice], y[r_slice], mode, bl_mean, bl_std, cfg)
        idx_r_glob = idx_r + (sidx + 1)
        indices = np.unique(np.sort(np.concatenate([idx_l, idx_r_glob])))
        guesses = []
        for i in indices:
            i = int(i)
            mu = float(x[i])
            if mode == "trapped":
                amp_val = bl_mean - y[i]
                amp_val = min(amp_val, -1e-4)
            else:
                amp_val = y[i] - bl_mean
                amp_val = max(amp_val, 1e-4)
            if mode == "trapped":
                amp_arr = bl_mean - y
            else:
                amp_arr = y - bl_mean
            sigma = _sigma_from_hwhm(x, amp_arr, i, abs(amp_val))
            guesses.append((float(amp_val), mu, float(sigma)))
        out[mode] = guesses
    return out
