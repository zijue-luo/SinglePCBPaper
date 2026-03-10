"""
Stage 2 baseline repair and safe split selection.

Selected algorithm settings:
1) iterative_2.0 + n_sigma=1.0
2) iterative_2.0 + n_sigma=1.5
3) iterative_2.5 + n_sigma=1.0
4) percentile_direct + n_sigma=2.0
"""
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_SCAN_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(RF_SCAN_DIR)
sys.path.insert(0, RF_SCAN_DIR)
sys.path.insert(0, SCRIPT_DIR)

from scipy.signal import find_peaks
from perform_fitting_tickling import build_tickling_groups, load_data
from baseline_estimation import baseline_estimate, baseline_regions_mask


SELECTED_CONFIGS = [
    {"method": "iterative_2.0", "n_sigma": 1.0},
    {"method": "iterative_2.0", "n_sigma": 1.5},
    {"method": "iterative_2.5", "n_sigma": 1.0},
    {"method": "percentile_direct", "n_sigma": 2.0},
]


# Step 1 tuning: baseline region repair
REPAIR_CFG = {
    "small_spike_max_len": 2,       # short out-of-range runs treated as spikes
    "small_spike_max_dev_z": 2.2,   # and not too far away from baseline threshold
    "drift_min_len": 5,             # continuous short drift
    "drift_mean_dev_z": 1.1,        # mild deviation -> recover as baseline
    "baseline_island_max_len": 2,   # remove tiny baseline islands inside signal runs
}


# Step 2 tuning: safe split choice (avoid edge-cutting)
SPLIT_CFG = {
    "safe_margin_points": 8,        # stay well away from baseline interval edges
    "min_interval_points": 10,      # minimum continuous baseline interval length
    "edge_prefer_frac": 0.2,        # prefer split in center 60% of interval (ignore 20% from each edge)
}


def _run_spans(mask):
    """Yield (start, end, value) for run-length encoded boolean mask."""
    arr = np.asarray(mask, dtype=bool)
    if len(arr) == 0:
        return []
    spans = []
    s = 0
    val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] != val:
            spans.append((s, i - 1, bool(val)))
            s = i
            val = arr[i]
    spans.append((s, len(arr) - 1, bool(val)))
    return spans


def _deviation_z(y, mode, baseline_mean, baseline_std, n_sigma):
    """Directional z-deviation from baseline threshold (>=0 only outside baseline)."""
    eps = max(float(baseline_std), 1e-8)
    if mode == "trapped":
        threshold = baseline_mean - n_sigma * baseline_std
        return np.maximum(0.0, (threshold - y) / eps)
    threshold = baseline_mean + n_sigma * baseline_std
    return np.maximum(0.0, (y - threshold) / eps)


def repair_mask(raw_mask, dev_z, cfg):
    """
    Repair baseline mask using clustering/run-length and deviation scale.
    True means baseline, False means signal.
    """
    mask = np.asarray(raw_mask, dtype=bool).copy()

    # Pass 1: recover short/mild out-of-range clusters as baseline.
    spans = _run_spans(mask)
    for s, e, val in spans:
        if val:
            continue
        run_len = e - s + 1
        run_dev = dev_z[s : e + 1]
        max_dev = float(np.max(run_dev)) if run_len > 0 else 0.0
        mean_dev = float(np.mean(run_dev)) if run_len > 0 else 0.0

        is_small_spike = (
            run_len <= cfg["small_spike_max_len"]
            and max_dev <= cfg["small_spike_max_dev_z"]
        )
        is_mild_drift = (
            run_len >= cfg["drift_min_len"]
            and mean_dev <= cfg["drift_mean_dev_z"]
        )
        if is_small_spike or is_mild_drift:
            mask[s : e + 1] = True

    # Pass 2: remove tiny baseline islands surrounded by signal.
    spans = _run_spans(mask)
    for idx, (s, e, val) in enumerate(spans):
        if not val:
            continue
        run_len = e - s + 1
        if run_len > cfg["baseline_island_max_len"]:
            continue
        left_signal = idx > 0 and spans[idx - 1][2] is False
        right_signal = idx < len(spans) - 1 and spans[idx + 1][2] is False
        if left_signal and right_signal:
            mask[s : e + 1] = False

    return mask


def detect_peak_positions(x, rs, rl):
    """Identify peaks from trapped dips and lost bumps."""
    x = np.asarray(x)
    rs = np.asarray(rs)
    rl = np.asarray(rl)

    pr_rs = max(0.01, 0.08 * (np.percentile(rs, 90) - np.percentile(rs, 10)))
    pr_rl = max(0.01, 0.08 * (np.percentile(rl, 90) - np.percentile(rl, 10)))

    p_t, _ = find_peaks(-rs, prominence=pr_rs)
    p_l, _ = find_peaks(rl, prominence=pr_rl)
    idx = np.unique(np.concatenate([p_t, p_l]))
    return x[idx], x[p_t], x[p_l]


def choose_split_index(x, combined_baseline_mask, peak_x, cfg):
    """
    Choose split point from central-safe baseline region:
    1) prefer equal left/right peak count,
    2) then closer to spectrum center,
    3) prefer points farther from interval edges (avoid edge-cutting),
    4) then longer baseline interval.
    """
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
        lo = s + margin
        hi = e - margin
        if lo > hi:
            continue
        for i in range(lo, hi + 1):
            xi = float(x[i])
            n_left = int(np.sum(peak_x < xi))
            n_right = int(np.sum(peak_x > xi))
            diff = abs(n_left - n_right)
            center_dist = abs(xi - center)
            edge_dist = min(i - s, e - i)  # distance from nearest interval edge
            score = (diff, center_dist, -edge_dist, -run_len)
            candidates.append((score, i, n_left, n_right, run_len))

    if not candidates:
        return None, None
    candidates.sort(key=lambda t: t[0])
    _, idx, n_left, n_right, run_len = candidates[0]
    info = {"n_left": n_left, "n_right": n_right, "interval_len": run_len}
    return idx, info


def process_one_run(ts, rf, method, n_sigma, out_dir):
    data_root = os.path.join(PROJECT_ROOT, "data_rf")
    x, ys = load_data(ts, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
    rs = np.asarray(ys["ratio_signal"])
    rl = np.asarray(ys["ratio_lost"])

    bl = baseline_estimate(x, rs, rl, method=method)
    raw_t = baseline_regions_mask(x, rs, "trapped", bl["trapped_mean"], bl["trapped_std"], n_sigma=n_sigma)
    raw_l = baseline_regions_mask(x, rl, "lost", bl["lost_mean"], bl["lost_std"], n_sigma=n_sigma)

    dev_t = _deviation_z(rs, "trapped", bl["trapped_mean"], bl["trapped_std"], n_sigma)
    dev_l = _deviation_z(rl, "lost", bl["lost_mean"], bl["lost_std"], n_sigma)
    fix_t = repair_mask(raw_t, dev_t, REPAIR_CFG)
    fix_l = repair_mask(raw_l, dev_l, REPAIR_CFG)

    combined_baseline = fix_t & fix_l
    peak_all, peak_t, peak_l = detect_peak_positions(x, rs, rl)
    split_idx, split_info = choose_split_index(x, combined_baseline, peak_all, SPLIT_CFG)

    split_x = float(x[split_idx]) if split_idx is not None else None
    left_points = int(split_idx + 1) if split_idx is not None else None
    right_points = int(len(x) - split_idx - 1) if split_idx is not None else None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, y, mode, raw_m, fix_m, peaks in [
        (axes[0], rs, "trapped", raw_t, fix_t, peak_t),
        (axes[1], rl, "lost", raw_l, fix_l, peak_l),
    ]:
        ax.plot(x, y, color="gray", linewidth=0.8, alpha=0.6, zorder=0)
        # Background bands: baseline = green, signal = red (very obvious)
        spans = _run_spans(fix_m)
        for s, e, val in spans:
            xlo, xhi = float(x[s]), float(x[e])
            if val:
                ax.axvspan(xlo, xhi, facecolor="lime", alpha=0.4, zorder=1)
            else:
                ax.axvspan(xlo, xhi, facecolor="orangered", alpha=0.35, zorder=1)
        ax.scatter(x[fix_m], y[fix_m], s=45, c="darkgreen", marker="o", edgecolors="black", linewidths=0.5, alpha=0.95, label="baseline", zorder=3)
        if np.any(~fix_m):
            ax.scatter(x[~fix_m], y[~fix_m], s=45, c="darkred", marker="s", edgecolors="black", linewidths=0.5, alpha=0.95, label="signal", zorder=3)
        for px in peaks:
            ax.axvline(px, color="gold", linestyle=":", linewidth=1.2, alpha=0.8, zorder=2)
        if split_x is not None:
            ax.axvline(split_x, color="navy", linestyle="--", linewidth=2.2, label="split", zorder=4)
        ax.set_ylabel(mode)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    title = f"{ts}  RF={rf} dBm  {method} n_sigma={n_sigma}"
    if split_x is not None and split_info is not None:
        title += f"  split={split_x:.2f}MHz  peaks(L/R)=({split_info['n_left']}/{split_info['n_right']})"
    axes[0].set_title(title)
    axes[1].set_xlabel("Tickle Frequency (MHz)")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{ts}_RF{rf:.1f}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "timestamp": ts,
        "RF_amplitude": rf,
        "method": method,
        "n_sigma": n_sigma,
        "split_x": split_x if split_x is not None else "",
        "left_points": left_points if left_points is not None else "",
        "right_points": right_points if right_points is not None else "",
        "peaks_total": int(len(peak_all)),
        "peaks_left": split_info["n_left"] if split_info else "",
        "peaks_right": split_info["n_right"] if split_info else "",
        "combined_baseline_points": int(np.sum(combined_baseline)),
    }


def run_stage2(out_root=None):
    out_root = out_root or os.path.join(SCRIPT_DIR, "stage2_split_with_repair")
    groups, _, _ = build_tickling_groups()
    all_runs = []
    for g in groups:
        for ts in g["timestamps"]:
            all_runs.append((ts, g["RF_amplitude"]))

    for cfg in SELECTED_CONFIGS:
        method = cfg["method"]
        n_sigma = cfg["n_sigma"]
        tag = f"{method}_n{str(n_sigma).replace('.', 'p')}"
        cfg_dir = os.path.join(out_root, tag)
        plot_dir = os.path.join(cfg_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        print(f"[Stage2] Running {tag} on {len(all_runs)} runs...")
        rows = []
        for i, (ts, rf) in enumerate(all_runs, start=1):
            try:
                row = process_one_run(ts, rf, method, n_sigma, plot_dir)
                rows.append(row)
            except Exception as exc:
                rows.append({
                    "timestamp": ts,
                    "RF_amplitude": rf,
                    "method": method,
                    "n_sigma": n_sigma,
                    "split_x": "",
                    "left_points": "",
                    "right_points": "",
                    "peaks_total": "",
                    "peaks_left": "",
                    "peaks_right": "",
                    "combined_baseline_points": "",
                    "error": str(exc),
                })
            if i % 10 == 0:
                print(f"  processed {i}/{len(all_runs)}")

        csv_path = os.path.join(cfg_dir, "split_summary.csv")
        fieldnames = [
            "timestamp", "RF_amplitude", "method", "n_sigma",
            "split_x", "left_points", "right_points",
            "peaks_total", "peaks_left", "peaks_right",
            "combined_baseline_points", "error",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[Stage2] Saved: {csv_path}")
        print(f"[Stage2] Plots: {plot_dir}")


if __name__ == "__main__":
    run_stage2()
