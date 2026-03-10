"""
Research: conservative find_peaks initialization based on split_experiment.

Goal:
- Use split_experiment first (iterative_2.0, n_sigma=1.0 default).
- Detect peaks separately on left/right segments.
- Merge results back to one experiment-level output (one figure per experiment).
- Be conservative: avoid treating weak fluctuations as peaks.
"""
import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_SCAN_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(RF_SCAN_DIR)
sys.path.insert(0, RF_SCAN_DIR)

from scipy.signal import find_peaks, savgol_filter
from perform_fitting_tickling import build_tickling_groups, load_data
from split_experiment import split_experiment


PEAK_CFG = {
    # Strict on peak strength; no smoothing (smoothing kills double peaks)
    "k_sigma_height": 2.5,
    "k_sigma_prom": 3.5,
    "range_prom_frac": 0.22,
    "min_prom_abs": 0.010,
    "min_distance_mhz": 0.6,   # allow double peaks (base+peak) ~0.6 MHz apart
    "min_width_pts": 1.0,
    "smooth_window_pts": 0,    # 0 = no smoothing; smoothing merges double peaks
    "smooth_polyorder": 2,
}


def _adaptive_smooth(y, cfg):
    n = len(y)
    w = int(cfg.get("smooth_window_pts", 0))
    if w <= 0 or w < 5:
        return np.asarray(y)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 5:
        return np.asarray(y)
    p = min(int(cfg["smooth_polyorder"]), w - 2)
    return savgol_filter(y, window_length=w, polyorder=p)


def _detect_mode_peaks_segment(x, y, mode, bl_mean, bl_std, cfg):
    """
    Conservative peak detection in one segment.
    mode='trapped': look for dips in y (use transformed positive peak signal)
    mode='lost':    look for bumps in y
    """
    x = np.asarray(x)
    y = np.asarray(y)
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

    idx, props = find_peaks(
        amp,
        height=height,
        prominence=prom,
        distance=dist,
        width=cfg["min_width_pts"],
    )
    return idx, {
        "height": float(height),
        "prominence": float(prom),
        "distance_points": int(dist),
        "amp_range": float(amp_range),
    }


def _merge_unique_sorted(*arrs):
    if not arrs:
        return np.array([], dtype=int)
    merged = np.concatenate([a for a in arrs if len(a) > 0]) if any(len(a) > 0 for a in arrs) else np.array([], dtype=int)
    if len(merged) == 0:
        return merged
    return np.unique(np.sort(merged))


def process_one_experiment(ts, rf, out_plot_dir, cfg):
    data_root = os.path.join(PROJECT_ROOT, "data_rf")
    x, ys = load_data(ts, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
    rs = np.asarray(ys["ratio_signal"])
    rl = np.asarray(ys["ratio_lost"])

    split = split_experiment(x, rs, rl, method="iterative_2.0", n_sigma=1.0)
    sidx = split["split_idx"]
    if sidx is None:
        sidx = len(x) // 2
    split_x = float(x[sidx])
    bl = split["baseline_info"]

    # Segment on split point, detect separately, then merge back.
    l_slice = slice(0, sidx + 1)
    r_slice = slice(sidx + 1, len(x))

    idx_t_l, meta_t_l = _detect_mode_peaks_segment(x[l_slice], rs[l_slice], "trapped", bl["trapped_mean"], bl["trapped_std"], cfg)
    idx_t_r, meta_t_r = _detect_mode_peaks_segment(x[r_slice], rs[r_slice], "trapped", bl["trapped_mean"], bl["trapped_std"], cfg)
    idx_l_l, meta_l_l = _detect_mode_peaks_segment(x[l_slice], rl[l_slice], "lost", bl["lost_mean"], bl["lost_std"], cfg)
    idx_l_r, meta_l_r = _detect_mode_peaks_segment(x[r_slice], rl[r_slice], "lost", bl["lost_mean"], bl["lost_std"], cfg)

    idx_t = _merge_unique_sorted(idx_t_l, idx_t_r + (sidx + 1))
    idx_l = _merge_unique_sorted(idx_l_l, idx_l_r + (sidx + 1))

    # Unreasonable result checks for tuning feedback.
    warn_flags = []
    if len(idx_t) == 0 and len(idx_l) == 0:
        warn_flags.append("no_peak_both_modes")
    if len(idx_t) > 14:
        warn_flags.append("too_many_trapped")
    if len(idx_l) > 14:
        warn_flags.append("too_many_lost")
    if abs(len(idx_t) - len(idx_l)) >= 8:
        warn_flags.append("large_mode_imbalance")

    os.makedirs(out_plot_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, y, mode, bl_mean, peak_idx in [
        (axes[0], rs, "trapped", bl["trapped_mean"], idx_t),
        (axes[1], rl, "lost", bl["lost_mean"], idx_l),
    ]:
        ax.plot(x, y, color="black", linewidth=1.1, alpha=0.9, label="data")
        ax.axhline(bl_mean, color="tab:gray", linestyle="--", linewidth=1.0, label="baseline mean")
        ax.axvline(split_x, color="navy", linestyle="--", linewidth=2.0, label="split")
        if len(peak_idx) > 0:
            ax.scatter(x[peak_idx], y[peak_idx], s=70, c="gold", edgecolors="black", linewidths=0.7, marker="^", label="find_peaks init")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_ylabel(mode)
        ax.legend(loc="upper right", fontsize=8)

    ttl = f"{ts} RF={rf} dBm | split={split_x:.2f} MHz | peaks(trapped/lost)=({len(idx_t)}/{len(idx_l)})"
    if warn_flags:
        ttl += " | WARN:" + ",".join(warn_flags)
    axes[0].set_title(ttl)
    axes[1].set_xlabel("Tickle Frequency (MHz)")
    fig.savefig(os.path.join(out_plot_dir, f"{ts}_RF{rf:.1f}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "timestamp": ts,
        "RF_amplitude": rf,
        "split_x": split_x,
        "n_peaks_trapped": int(len(idx_t)),
        "n_peaks_lost": int(len(idx_l)),
        "warn_flags": ";".join(warn_flags),
        "trapped_prom_left": meta_t_l.get("prominence", ""),
        "trapped_prom_right": meta_t_r.get("prominence", ""),
        "lost_prom_left": meta_l_l.get("prominence", ""),
        "lost_prom_right": meta_l_r.get("prominence", ""),
    }


def run_all(output_root=None):
    output_root = output_root or os.path.join(SCRIPT_DIR, "find_peaks_from_split")
    plot_dir = os.path.join(output_root, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    groups, _, _ = build_tickling_groups()
    all_runs = [(ts, g["RF_amplitude"]) for g in groups for ts in g["timestamps"]]

    rows = []
    for i, (ts, rf) in enumerate(all_runs, start=1):
        try:
            rows.append(process_one_experiment(ts, rf, plot_dir, PEAK_CFG))
        except Exception as exc:
            rows.append({
                "timestamp": ts,
                "RF_amplitude": rf,
                "split_x": "",
                "n_peaks_trapped": "",
                "n_peaks_lost": "",
                "warn_flags": f"error:{exc}",
                "trapped_prom_left": "",
                "trapped_prom_right": "",
                "lost_prom_left": "",
                "lost_prom_right": "",
            })
        if i % 10 == 0:
            print(f"[find_peaks] processed {i}/{len(all_runs)}")

    csv_path = os.path.join(output_root, "find_peaks_summary.csv")
    fields = [
        "timestamp", "RF_amplitude", "split_x",
        "n_peaks_trapped", "n_peaks_lost", "warn_flags",
        "trapped_prom_left", "trapped_prom_right",
        "lost_prom_left", "lost_prom_right",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[find_peaks] saved summary: {csv_path}")
    print(f"[find_peaks] plots: {plot_dir}")

    # Quick quality printout for tuning
    n_warn = sum(1 for r in rows if r.get("warn_flags"))
    t_counts = [int(r["n_peaks_trapped"]) for r in rows if str(r.get("n_peaks_trapped", "")).isdigit()]
    l_counts = [int(r["n_peaks_lost"]) for r in rows if str(r.get("n_peaks_lost", "")).isdigit()]
    if t_counts and l_counts:
        print(
            "[find_peaks] trapped peaks median/p90/max = "
            f"{np.median(t_counts):.1f}/{np.percentile(t_counts, 90):.1f}/{np.max(t_counts)}"
        )
        print(
            "[find_peaks] lost peaks median/p90/max = "
            f"{np.median(l_counts):.1f}/{np.percentile(l_counts, 90):.1f}/{np.max(l_counts)}"
        )
    print(f"[find_peaks] warnings: {n_warn}/{len(rows)}")


if __name__ == "__main__":
    run_all()
