"""
Deep analysis of raw tickling/RF scan data.
Loads actual column files and studies: structure, noise, rep variability, peak shapes.
"""
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_SCAN_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(RF_SCAN_DIR)
sys.path.insert(0, RF_SCAN_DIR)

from perform_fitting_tickling import (
    build_tickling_groups,
    load_data,
    load_and_average_group,
    _get_data_root,
)


def analyze_single_run(x, ratio_signal, ratio_lost, label=""):
    """Basic stats for one run."""
    n = len(x)
    rs = np.asarray(ratio_signal)
    rl = np.asarray(ratio_lost)

    edge_frac = 0.2
    n_edge = int(n * edge_frac)
    flat_rs = np.concatenate([rs[:n_edge], rs[-n_edge:]])
    flat_rl = np.concatenate([rl[:n_edge], rl[-n_edge:]])

    out = {
        "label": label,
        "n_points": n,
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "x_step": float(np.diff(x).mean()) if n > 1 else 0,
        "rs_mean": float(rs.mean()),
        "rs_std": float(rs.std()),
        "rs_range": (float(rs.min()), float(rs.max())),
        "rl_mean": float(rl.mean()),
        "rl_std": float(rl.std()),
        "rl_range": (float(rl.min()), float(rl.max())),
        "flat_rs_std": float(flat_rs.std()),
        "flat_rl_std": float(flat_rl.std()),
        "flat_rs_mean": float(flat_rs.mean()),
        "flat_rl_mean": float(flat_rl.mean()),
    }
    return out


def analyze_rep_variability(timestamps, data_root=None):
    """Compare ratio_signal and ratio_lost across reps for one group."""
    if not timestamps:
        return {}
    all_rs = []
    all_rl = []
    x_ref = None
    for ts in timestamps:
        x, ys = load_data(ts, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
        if x_ref is None:
            x_ref = x
        all_rs.append(np.asarray(ys["ratio_signal"]))
        all_rl.append(np.asarray(ys["ratio_lost"]))

    all_rs = np.array(all_rs)
    all_rl = np.array(all_rl)
    mean_rs = all_rs.mean(axis=0)
    mean_rl = all_rl.mean(axis=0)
    std_rs = all_rs.std(axis=0)
    std_rl = all_rl.std(axis=0)

    return {
        "n_reps": len(timestamps),
        "n_points": len(x_ref),
        "std_rs_mean": float(std_rs.mean()),
        "std_rs_max": float(std_rs.max()),
        "std_rl_mean": float(std_rl.mean()),
        "std_rl_max": float(std_rl.max()),
        "std_rs_percentile_90": float(np.percentile(std_rs, 90)),
        "std_rl_percentile_90": float(np.percentile(std_rl, 90)),
        "rs_mean_range": (float(mean_rs.min()), float(mean_rs.max())),
        "rl_mean_range": (float(mean_rl.min()), float(mean_rl.max())),
    }


def estimate_peak_regions(y, x, threshold_std=2.0):
    window = 20
    n = len(y)
    baseline = np.convolve(y, np.ones(window) / window, mode="same")
    dev = y - baseline
    std_dev = np.std(dev)
    if std_dev < 1e-10:
        return []
    mask = np.abs(dev) > threshold_std * std_dev
    regions = []
    in_region = False
    start = 0
    for i in range(n):
        if mask[i] and not in_region:
            start = i
            in_region = True
        elif not mask[i] and in_region:
            regions.append((float(x[start]), float(x[i - 1])))
            in_region = False
    if in_region:
        regions.append((float(x[start]), float(x[-1])))
    return regions


def count_visible_dips(ratio_signal, x, prominence_frac=0.02):
    neg = -np.asarray(ratio_signal)
    from scipy.signal import find_peaks
    peaks, props = find_peaks(neg, prominence=prominence_frac)
    return len(peaks), list(x[peaks])


def main():
    data_root = os.path.join(PROJECT_ROOT, "data_rf")
    groups, _, _ = build_tickling_groups()
    if not groups:
        print("No groups found.")
        return

    print("=" * 70)
    print("RAW DATA STRUCTURE")
    print("=" * 70)

    g0 = groups[0]
    ts0 = g0["timestamps"][0]
    x, ys = load_data(ts0, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
    info = analyze_single_run(x, ys["ratio_signal"], ys["ratio_lost"], ts0)
    print(f"Single run: {ts0}")
    print(f"  N points: {info['n_points']}")
    print(f"  x range: {info['x_min']} - {info['x_max']} MHz, step ~{info['x_step']:.3f} MHz")
    print(f"  ratio_signal (trapped/loading): mean={info['rs_mean']:.5f}, range=[{info['rs_range'][0]:.5f}, {info['rs_range'][1]:.5f}]")
    print(f"  ratio_lost: mean={info['rl_mean']:.5f}, range=[{info['rl_range'][0]:.5f}, {info['rl_range'][1]:.5f}]")
    print(f"  Flat-region std (edges 20%): rs={info['flat_rs_std']:.6f}, rl={info['flat_rl_std']:.6f}")
    print()

    x2, ys2 = load_data(ts0, ynames=["trapped_signal", "lost_signal", "loading_signal"], data_root=data_root)
    if ys2:
        tr = np.array(ys2.get("trapped_signal", []))
        ls = np.array(ys2.get("lost_signal", []))
        ld = np.array(ys2.get("loading_signal", []))
        if len(tr) == len(x) and len(ld) > 0:
            print("RAW COUNTS (trapped, lost, loading):")
            print(f"  loading: min={ld.min():.0f}, max={ld.max():.0f}, mean={ld.mean():.0f}")
            print(f"  trapped: min={tr.min():.0f}, max={tr.max():.0f}")
            print(f"  lost: min={ls.min():.0f}, max={ls.max():.0f}")
            edge = int(len(x) * 0.2)
            ld_flat = ld[:edge]
            print(f"  In flat region (first 20%): loading mean={ld_flat.mean():.0f}, sqrt(loading)~={np.sqrt(ld_flat.mean()):.0f} (Poisson scale)")
            print()
    print()

    print("=" * 70)
    print("REP-TO-REP VARIABILITY (first 5 groups)")
    print("=" * 70)
    for i, g in enumerate(groups[:5]):
        rf = g["RF_amplitude"]
        ts_list = g["timestamps"]
        var = analyze_rep_variability(ts_list, data_root=data_root)
        if var:
            print(f"RF={rf} dBm, n_reps={var['n_reps']}:")
            print(f"  Pointwise std of ratio_signal: mean={var['std_rs_mean']:.6f}, max={var['std_rs_max']:.6f}, p90={var['std_rs_percentile_90']:.6f}")
            print(f"  Pointwise std of ratio_lost:   mean={var['std_rl_mean']:.6f}, max={var['std_rl_max']:.6f}, p90={var['std_rl_percentile_90']:.6f}")
            print()
    print()

    print("=" * 70)
    print("PEAK STRUCTURE (averaged data, first 5 groups)")
    print("=" * 70)
    try:
        from scipy.signal import find_peaks
    except ImportError:
        find_peaks = None

    for i, g in enumerate(groups[:5]):
        rf = g["RF_amplitude"]
        x_avg, ys_avg = load_and_average_group(g["timestamps"], data_root=data_root)
        rs = np.array(ys_avg["ratio_signal"])
        rl = np.array(ys_avg["ratio_lost"])
        regions_rs = estimate_peak_regions(rs, x_avg, threshold_std=2.0)
        regions_rl = estimate_peak_regions(rl, x_avg, threshold_std=2.0)
        if find_peaks:
            n_dips, mu_dips = count_visible_dips(rs, x_avg, prominence_frac=0.015)
            n_bumps, mu_bumps = count_visible_dips(rl, x_avg, prominence_frac=0.015)
            print(f"RF={rf} dBm (averaged over {len(g['timestamps'])} reps):")
            print(f"  ratio_signal dips (trapped drop): {n_dips} at mu ~ {[f'{m:.1f}' for m in mu_dips[:12]]}")
            print(f"  ratio_lost bumps: {n_bumps}")
            print(f"  Heuristic peak regions (2sigma): rs={len(regions_rs)}, rl={len(regions_rl)}")
        else:
            print(f"RF={rf} dBm: peak regions rs={len(regions_rs)}, rl={len(regions_rl)}")
        print()
    print()

    print("=" * 70)
    print("KEY METRICS FOR FITTING OPTIMIZATION")
    print("=" * 70)
    x_avg, ys_avg = load_and_average_group(groups[0]["timestamps"], data_root=data_root)
    rs = np.array(ys_avg["ratio_signal"])
    n = len(rs)
    flat = np.concatenate([rs[: int(n * 0.15)], rs[int(n * 0.85) :]])
    peak_region = rs[int(n * 0.25) : int(n * 0.75)]
    flat_std = np.std(flat)
    peak_dynamic_range = peak_region.max() - peak_region.min()
    print(f"  Flat baseline std (ratio_signal): {flat_std:.6f}")
    print(f"  Peak region dynamic range: {peak_dynamic_range:.5f}")
    print(f"  Effective SNR (range/flat_std): {peak_dynamic_range / flat_std:.1f}")
    print(f"  N points in scan: {n} -> step = {(x_avg.max()-x_avg.min())/(n-1):.3f} MHz")
    print()
    print("  Conclusion: raw data has ~241 points, 0.5 MHz step.")
    print("  Fitter scans 120 mu0 values -> ~2x oversampling relative to data points in peak region.")
    print("  Rep-averaging reduces pointwise std by ~1/sqrt(n_reps).")


if __name__ == "__main__":
    main()