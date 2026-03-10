"""
Robust baseline estimation and validation for tickling RF scan data.
Goal: reliably compute baseline level and fluctuation (std) for ratio_signal and ratio_lost.
Critical for splitting data at baseline regions and piecewise fitting.

PRODUCTION API:
  from baseline_estimation import baseline_estimate, baseline_regions_mask, get_baseline_regions

  bl = baseline_estimate(x, ratio_signal, ratio_lost, method="iterative_2.5")
  # bl = {trapped_mean, trapped_std, lost_mean, lost_std}

  # For splitting: use strict n_sigma=1.0 to avoid cutting in peak tails
  mask_t = baseline_regions_mask(x, ratio_signal, "trapped", bl["trapped_mean"], bl["trapped_std"], n_sigma=1.0)
  regions = get_baseline_regions(x, mask_t, min_points=5)  # [(x_lo, x_hi), ...]

  # Cut data at baseline regions for piecewise fitting
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

from perform_fitting_tickling import build_tickling_groups, load_data, _get_data_root, _get_tickling_json_path

# =============================================================================
# Baseline estimation methods
# =============================================================================
# mode: "trapped" -> ratio_signal, baseline is HIGH (dips at resonance)
# mode: "lost"   -> ratio_lost,   baseline is LOW  (peaks at resonance)


def baseline_edge(x, y, mode, edge_frac=0.15):
    """
    M1: Use first and last edge_frac of points.
    Risk: peaks near 20 or 140 MHz would contaminate. Must validate.
    Returns: (mean, std)
    """
    n = len(y)
    n_edge = max(1, int(n * edge_frac))
    vals = np.concatenate([np.asarray(y[:n_edge]), np.asarray(y[-n_edge:])])
    return float(np.mean(vals)), float(np.std(vals))


def baseline_quantile(y, mode, upper_frac=0.35, lower_frac=0.35):
    """
    M2: For trapped (high baseline): use upper_frac of values (points likely away from dips).
        For lost (low baseline): use lower_frac of values.
    fluctuation = MAD of those points (robust to outliers).
    Returns: (mean, std_equiv) where std_equiv uses 1.4826*MAD for normal consistency
    """
    y = np.asarray(y, dtype=float)
    if mode == "trapped":
        thresh = np.percentile(y, 100 * (1 - upper_frac))
        mask = y >= thresh
    else:
        thresh = np.percentile(y, 100 * lower_frac)
        mask = y <= thresh
    vals = y[mask]
    if len(vals) < 3:
        return float(np.median(y)), float(np.std(y))
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    std_equiv = 1.4826 * mad if mad > 1e-12 else np.std(vals)
    return float(np.mean(vals)), float(std_equiv)


def baseline_iterative_sigma(y, mode, n_sigma=2.5, max_iter=10):
    """
    M3: Iteratively exclude points > n_sigma from median, recompute robust mean and std.
    Start with all points. Stop when no more exclusions or max_iter.
    Returns: (mean, std)
    """
    vals = np.asarray(y, dtype=float).copy()
    for _ in range(max_iter):
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        sigma = 1.4826 * mad if mad > 1e-12 else np.std(vals)
        if sigma < 1e-12:
            break
        if mode == "trapped":
            # Baseline high; exclude points much below median (dips)
            mask = vals >= med - n_sigma * sigma
        else:
            # Baseline low; exclude points much above median (peaks)
            mask = vals <= med + n_sigma * sigma
        new_vals = vals[mask]
        if len(new_vals) == len(vals):
            break
        vals = new_vals
    if len(vals) < 3:
        vals = np.asarray(y)
    return float(np.mean(vals)), float(np.std(vals))


def baseline_rolling_lowvar(x, y, mode, window_mhz=15.0, keep_frac=0.4):
    """
    M4: Rolling window std. Points in low local-variance regions are baseline.
    window_mhz: window size in MHz
    keep_frac: keep points with local_std in the lowest keep_frac.
    Returns: (mean, std)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    dx = np.median(np.diff(x)) if len(x) > 1 else 0.5
    w = max(3, int(window_mhz / dx))
    local_std = np.full_like(y, np.nan)
    for i in range(len(y)):
        lo = max(0, i - w // 2)
        hi = min(len(y), i + w // 2 + 1)
        local_std[i] = np.std(y[lo:hi])
    thresh = np.nanpercentile(local_std, 100 * keep_frac)
    mask = ~np.isnan(local_std) & (local_std <= thresh)
    vals = y[mask]
    if len(vals) < 5:
        return float(np.median(y)), float(np.std(y))
    return float(np.mean(vals)), float(np.std(vals))


def baseline_trimmed_tails(y, mode, trim_both_frac=0.25):
    """
    M5: Exclude trim_both_frac from both tails. Remaining = "middle" values.
    For trapped: dips are in lower tail, so trimmed mean excludes them.
    For lost: peaks are in upper tail.
    Actually: trim top and bottom trim_both_frac each.
    Returns: (mean, std)
    """
    y = np.asarray(y)
    k = int(len(y) * trim_both_frac)
    k = max(0, min(k, len(y) // 2 - 1))
    if k == 0:
        return float(np.mean(y)), float(np.std(y))
    sorted_y = np.sort(y)
    trimmed = sorted_y[k : len(y) - k]
    return float(np.mean(trimmed)), float(np.std(trimmed))


def baseline_percentile_direct(y, mode, center_pct=50, half_width=15):
    """
    M6: For trapped: baseline = median of values in [50-half_width, 50+half_width] percentile.
        (Middle band, excludes deep dips)
    For lost: same idea.
    Returns: (median, mad_based_std)
    """
    y = np.asarray(y)
    lo = np.percentile(y, max(1, center_pct - half_width))
    hi = np.percentile(y, min(99, center_pct + half_width))
    if mode == "trapped":
        # Baseline high; use upper-middle band
        lo = np.percentile(y, 50)
        hi = np.percentile(y, 99)
    else:
        lo = np.percentile(y, 1)
        hi = np.percentile(y, 50)
    mask = (y >= lo) & (y <= hi)
    vals = y[mask]
    if len(vals) < 5:
        vals = y
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    std_equiv = 1.4826 * mad if mad > 1e-12 else np.std(vals)
    return float(np.mean(vals)), float(std_equiv)


# =============================================================================
# Run all methods and compare
# =============================================================================

METHODS = {
    "edge_15": lambda x, y, m: baseline_edge(x, y, m, 0.15),
    "edge_20": lambda x, y, m: baseline_edge(x, y, m, 0.20),
    "edge_25": lambda x, y, m: baseline_edge(x, y, m, 0.25),
    "quantile_35": lambda x, y, m: baseline_quantile(y, m, 0.35, 0.35),
    "iterative_2.5": lambda x, y, m: baseline_iterative_sigma(y, m, 2.5),
    "iterative_2.0": lambda x, y, m: baseline_iterative_sigma(y, m, 2.0),
    "rolling_15_40": lambda x, y, m: baseline_rolling_lowvar(x, y, m, 15.0, 0.4),
    "trimmed_25": lambda x, y, m: baseline_trimmed_tails(y, m, 0.25),
    "percentile_direct": lambda x, y, m: baseline_percentile_direct(y, m),
}


def estimate_baseline_all_methods(x, rs, rl):
    """Run all methods for both modes. Return dict."""
    out = {}
    for name, fn in METHODS.items():
        out[name] = {
            "trapped_mean": fn(x, rs, "trapped")[0],
            "trapped_std": fn(x, rs, "trapped")[1],
            "lost_mean": fn(x, rl, "lost")[0],
            "lost_std": fn(x, rl, "lost")[1],
        }
    return out


def load_known_peaks_from_csv(csv_path):
    """
    Load mu, sigma from averaged fit CSV for validation.
    Returns: dict rf -> {"trapped": [(mu,sig),...], "lost": [(mu,sig),...]}
    """
    by_rf = {}
    if not os.path.isfile(csv_path):
        return by_rf
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row.get("mode", "")
            if mode not in ("trapped", "lost"):
                continue
            try:
                rf = float(row.get("RF_amplitude", 0))
            except (ValueError, TypeError):
                continue
            if rf not in by_rf:
                by_rf[rf] = {"trapped": [], "lost": []}
            n = int(row.get("n_peaks", 0))
            for i in range(1, n + 1):
                mu = row.get(f"mu{i}")
                sig = row.get(f"sigma{i}")
                if mu and sig:
                    try:
                        by_rf[rf][mode].append((float(mu), max(float(sig), 0.3)))
                    except ValueError:
                        pass
    return by_rf


def baseline_excluding_peaks(x, y, mode, peak_mus_sigmas, exclude_n_sigma=5.0):
    """
    Ground truth: exclude points within exclude_n_sigma * sigma of each peak.
    Remaining points = baseline. Returns (mean, std).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.ones(len(x), dtype=bool)
    for mu, sig in peak_mus_sigmas:
        half_width = exclude_n_sigma * max(sig, 0.5)
        mask &= (x < mu - half_width) | (x > mu + half_width)
    vals = y[mask]
    if len(vals) < 10:
        return float(np.nan), float(np.nan)
    return float(np.mean(vals)), float(np.std(vals))


# =============================================================================
# Main: run on ALL runs, validate
# =============================================================================

def main():
    data_root = os.path.join(PROJECT_ROOT, "data_rf")
    csv_path = os.path.join(RF_SCAN_DIR, "final_analysis_plots", "run_best_models_averaged_20260227_201811.csv")
    known_peaks = load_known_peaks_from_csv(csv_path)

    groups, _, _ = build_tickling_groups()
    all_timestamps = []
    ts_to_rf = {}
    for g in groups:
        rf = g["RF_amplitude"]
        for ts in g["timestamps"]:
            all_timestamps.append(ts)
            ts_to_rf[ts] = rf

    print("=" * 75)
    print("BASELINE ESTIMATION: METHOD COMPARISON ON ALL RUNS")
    print("=" * 75)
    print(f"Total runs: {len(all_timestamps)}")
    print(f"Methods: {list(METHODS.keys())}")
    print()

    # Collect results for every run
    results = []
    failed = []
    for ts in all_timestamps:
        try:
            x, ys = load_data(ts, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
        except Exception as e:
            failed.append((ts, str(e)))
            continue
        rs = np.array(ys["ratio_signal"])
        rl = np.array(ys["ratio_lost"])

        row = {"ts": ts, "rf": ts_to_rf.get(ts, None)}
        row["all_methods"] = estimate_baseline_all_methods(x, rs, rl)

        # Ground truth from known peaks (filter by RF)
        rf = ts_to_rf.get(ts)
        peaks_rf = known_peaks.get(rf, {}) if rf is not None else {}
        if peaks_rf.get("trapped") or peaks_rf.get("lost"):
            gt_t_mean, gt_t_std = baseline_excluding_peaks(
                x, rs, "trapped", peaks_rf.get("trapped", []))
            gt_l_mean, gt_l_std = baseline_excluding_peaks(
                x, rl, "lost", peaks_rf.get("lost", []))
            row["ground_truth"] = {
                "trapped_mean": gt_t_mean, "trapped_std": gt_t_std,
                "lost_mean": gt_l_mean, "lost_std": gt_l_std,
            }
        else:
            row["ground_truth"] = None

        results.append(row)

    if failed:
        print(f"Failed to load: {len(failed)} runs")
        for ts, err in failed[:5]:
            print(f"  {ts}: {err}")
        if len(failed) > 5:
            print(f"  ... and {len(failed)-5} more")
        print()

    n_ok = len(results)

    # 1. Method agreement: std of baseline_mean across methods per run
    print("1. METHOD AGREEMENT (per run)")
    print("-" * 50)
    agreement = {m: [] for m in METHODS}
    for r in results:
        for m in METHODS:
            t_mean = r["all_methods"][m]["trapped_mean"]
            t_std = r["all_methods"][m]["trapped_std"]
            l_mean = r["all_methods"][m]["lost_mean"]
            l_std = r["all_methods"][m]["lost_std"]
            agreement[m].append((t_mean, t_std, l_mean, l_std))

    # Cross-method variance per run
    run_disagreements = []
    for r in results:
        t_means = [r["all_methods"][m]["trapped_mean"] for m in METHODS]
        t_stds = [r["all_methods"][m]["trapped_std"] for m in METHODS]
        l_means = [r["all_methods"][m]["lost_mean"] for m in METHODS]
        l_stds = [r["all_methods"][m]["lost_std"] for m in METHODS]
        t_mean_std = np.std(t_means)
        t_std_std = np.std(t_stds)
        l_mean_std = np.std(l_means)
        l_std_std = np.std(l_stds)
        run_disagreements.append({
            "ts": r["ts"], "rf": r["rf"],
            "t_mean_sd": t_mean_std, "t_std_sd": t_std_std,
            "l_mean_sd": l_mean_std, "l_std_sd": l_std_std,
        })

    # Runs where methods disagree a lot
    thresh_mean = 0.005
    thresh_std = 0.003
    suspect = [d for d in run_disagreements
               if d["t_mean_sd"] > thresh_mean or d["l_mean_sd"] > thresh_mean
               or d["t_std_sd"] > thresh_std or d["l_std_sd"] > thresh_std]

    print(f"Runs where methods disagree (mean_sd>{thresh_mean} or std_sd>{thresh_std}): {len(suspect)}/{n_ok}")
    if suspect:
        for d in suspect[:15]:
            print(f"  {d['ts']} RF={d['rf']}: t_mean_sd={d['t_mean_sd']:.5f} t_std_sd={d['t_std_sd']:.5f} "
                  f"l_mean_sd={d['l_mean_sd']:.5f} l_std_sd={d['l_std_sd']:.5f}")
        if len(suspect) > 15:
            print(f"  ... and {len(suspect)-15} more")
    print()

    # 2. Comparison to ground truth (known peaks excluded)
    n_gt = sum(1 for r in results if r.get("ground_truth") and not np.isnan((r["ground_truth"] or {}).get("trapped_mean", np.nan)))
    if n_gt > 0:
        print("2. COMPARISON TO GROUND TRUTH (exclude 5*sigma around fitted peaks)")
        print("-" * 50)
        gt_errors = {m: [] for m in METHODS}
        for r in results:
            if r["ground_truth"] is None:
                continue
            gt = r["ground_truth"]
            for m in METHODS:
                if np.isnan(gt["trapped_mean"]):
                    continue
                e_t_mean = abs(r["all_methods"][m]["trapped_mean"] - gt["trapped_mean"])
                e_t_std = abs(r["all_methods"][m]["trapped_std"] - gt["trapped_std"])
                e_l_mean = abs(r["all_methods"][m]["lost_mean"] - gt["lost_mean"])
                e_l_std = abs(r["all_methods"][m]["lost_std"] - gt["lost_std"])
                gt_errors[m].append((e_t_mean, e_t_std, e_l_mean, e_l_std))

        for m in METHODS:
            arr = np.array(gt_errors[m])
            if len(arr) == 0:
                continue
            print(f"  {m}:")
            print(f"    trapped mean error: med={np.median(arr[:,0]):.6f} max={np.max(arr[:,0]):.6f}")
            print(f"    trapped std  error: med={np.median(arr[:,1]):.6f} max={np.max(arr[:,1]):.6f}")
            print(f"    lost   mean error: med={np.median(arr[:,2]):.6f} max={np.max(arr[:,2]):.6f}")
            print(f"    lost   std  error: med={np.median(arr[:,3]):.6f} max={np.max(arr[:,3]):.6f}")
        print()

    # 3. Aggregate statistics per method
    print("3. AGGREGATE BASELINE STATS PER METHOD (over all runs)")
    print("-" * 50)
    for m in METHODS:
        t_means = [r["all_methods"][m]["trapped_mean"] for r in results]
        t_stds = [r["all_methods"][m]["trapped_std"] for r in results]
        l_means = [r["all_methods"][m]["lost_mean"] for r in results]
        l_stds = [r["all_methods"][m]["lost_std"] for r in results]
        print(f"  {m}:")
        print(f"    trapped: mean={np.mean(t_means):.5f} +/- {np.std(t_means):.5f}  "
              f"std_avg={np.mean(t_stds):.5f} range=[{np.min(t_stds):.5f},{np.max(t_stds):.5f}]")
        print(f"    lost:    mean={np.mean(l_means):.5f} +/- {np.std(l_means):.5f}  "
              f"std_avg={np.mean(l_stds):.5f} range=[{np.min(l_stds):.5f},{np.max(l_stds):.5f}]")
    print()

    # 4. RECOMMENDED METHOD
    print("=" * 75)
    print("4. RECOMMENDED METHOD & API")
    print("=" * 75)
    # Choose method with smallest median error to ground truth, or if no GT, smallest cross-method variance
    has_gt = any(p.get("trapped") or p.get("lost") for p in known_peaks.values())
    if has_gt and n_gt > 0:
        def err(m):
            errs = []
            for r in results:
                gt = r.get("ground_truth")
                if not gt or np.isnan(gt.get("trapped_mean", np.nan)):
                    continue
                errs.append(abs(r["all_methods"][m]["trapped_mean"] - gt["trapped_mean"]))
            return np.median(errs) if errs else 1e9
        best = min(METHODS.keys(), key=err)
    else:
        best = min(METHODS.keys(),
                   key=lambda m: np.mean([run_disagreements[i]["t_mean_sd"] for i in range(len(run_disagreements))]))
    print(f"  Best method (lowest error to ground truth): {best}")
    print()
    print("  API: baseline_estimate(x, ratio_signal, ratio_lost)")
    print("       -> dict with trapped_mean, trapped_std, lost_mean, lost_std")
    print("  Recommended function: use iterative_2.5 (robust, no edge assumption)")
    print("  Alternative: quantile_35 (also robust, explicit quantile)")
    print()

    # 5. Edge contamination check: do edge methods sometimes fail?
    print("5. EDGE CONTAMINATION CHECK")
    print("-" * 50)
    # Compare edge_15 vs iterative: if edge gives very different result, edges might have peaks
    edge_vs_iter = []
    for r in results:
        e = r["all_methods"]["edge_15"]
        i = r["all_methods"]["iterative_2.5"]
        edge_vs_iter.append({
            "ts": r["ts"], "rf": r["rf"],
            "t_diff": abs(e["trapped_mean"] - i["trapped_mean"]),
            "l_diff": abs(e["lost_mean"] - i["lost_mean"]),
        })
    bad_edge = [x for x in edge_vs_iter if x["t_diff"] > 0.01 or x["l_diff"] > 0.01]
    print(f"  Runs where edge_15 differs from iterative_2.5 by >0.01: {len(bad_edge)}/{n_ok}")
    if bad_edge:
        for x in bad_edge[:10]:
            print(f"    {x['ts']} RF={x['rf']}: trapped_diff={x['t_diff']:.5f} lost_diff={x['l_diff']:.5f}")
    print()

    # 6. Per-RF stability
    print("6. BASELINE STABILITY BY RF (iterative_2.5)")
    print("-" * 50)
    by_rf = {}
    for r in results:
        rf = r["rf"]
        if rf not in by_rf:
            by_rf[rf] = {"t_mean": [], "t_std": [], "l_mean": [], "l_std": []}
        m = r["all_methods"]["iterative_2.5"]
        by_rf[rf]["t_mean"].append(m["trapped_mean"])
        by_rf[rf]["t_std"].append(m["trapped_std"])
        by_rf[rf]["l_mean"].append(m["lost_mean"])
        by_rf[rf]["l_std"].append(m["lost_std"])
    for rf in sorted(by_rf.keys()):
        d = by_rf[rf]
        n = len(d["t_mean"])
        print(f"  RF={rf:5.1f} dBm (n={n}): "
              f"trapped mean {np.mean(d['t_mean']):.5f} +/- {np.std(d['t_mean']):.5f}  "
              f"trapped std {np.mean(d['t_std']):.5f}  "
              f"lost mean {np.mean(d['l_mean']):.5f} +/- {np.std(d['l_mean']):.5f}  "
              f"lost std {np.mean(d['l_std']):.5f}")
    # 7. CRITICAL: validate baseline regions do NOT overlap known peaks
    print("7. BASELINE REGION vs KNOWN PEAK OVERLAP")
    print("-" * 50)
    print("  Point is 'baseline' if within n_sigma * baseline_std of baseline_mean.")
    print("  Violation = baseline point falls within 3*peak_sigma of a fitted peak.")
    print("  Sensitivity: try different n_sigma to find safe threshold.")
    print()
    exclude_sigma = 3.0
    for n_sigma_baseline in [1.0, 1.5, 2.0, 2.5]:
        violations = []
        for r in results:
            rf = r["rf"]
            peaks_rf = known_peaks.get(rf, {}) if rf is not None else {}
            if not peaks_rf.get("trapped") and not peaks_rf.get("lost"):
                continue
            try:
                x, ys = load_data(r["ts"], ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
            except Exception:
                continue
            rs = np.array(ys["ratio_signal"])
            rl = np.array(ys["ratio_lost"])
            bl = r["all_methods"]["iterative_2.5"]
            mask_t = baseline_regions_mask(x, rs, "trapped", bl["trapped_mean"], bl["trapped_std"], n_sigma_baseline)
            mask_l = baseline_regions_mask(x, rl, "lost", bl["lost_mean"], bl["lost_std"], n_sigma_baseline)
            ok_t, nv_t = validate_baseline_no_peak_overlap(
                x, mask_t, peaks_rf.get("trapped", []), exclude_sigma, sigma_cap_mhz=5.0)
            ok_l, nv_l = validate_baseline_no_peak_overlap(
                x, mask_l, peaks_rf.get("lost", []), exclude_sigma, sigma_cap_mhz=5.0)
            if not ok_t or not ok_l:
                violations.append({
                    "ts": r["ts"], "rf": rf,
                    "trapped_violations": nv_t, "lost_violations": nv_l,
                })
        n_validated = sum(1 for r in results if known_peaks.get(r["rf"], {}).get("trapped") or known_peaks.get(r["rf"], {}).get("lost"))
        print(f"  n_sigma={n_sigma_baseline}: {len(violations)}/{n_validated} runs with violations", end="")
        if violations:
            total_v = sum(v["trapped_violations"] + v["lost_violations"] for v in violations)
            print(f" (total {total_v} baseline-in-peak counts)")
        else:
            print(" -> SAFE")
    print()
    print("  Recommended: use n_sigma where violations=0 for splitting.")
    print()
    print("  NOTE: RF=-0.5 has dense peaks; baseline regions may overlap peak tails.")
    print("  For splitting: use iterative_2.5 + n_sigma=1.0 (strictest) or edge regions.")
    print()
    print("DONE. Use iterative_2.5 for production.")

    # 8. Plot baseline-selected vs excluded points
    print()
    print("8. PLOTTING BASELINE SELECTION")
    print("-" * 50)
    run_plot_baseline_selection(methods=list(METHODS.keys()), n_sigma_list=[1.0, 1.5, 2.0])


def baseline_estimate(x, ratio_signal, ratio_lost, method="iterative_2.5"):
    """
    Production API: estimate baseline level and fluctuation.
    Returns: dict with trapped_mean, trapped_std, lost_mean, lost_std
    """
    x = np.asarray(x)
    rs = np.asarray(ratio_signal)
    rl = np.asarray(ratio_lost)
    fn = METHODS.get(method, METHODS["iterative_2.5"])
    t_mean, t_std = fn(x, rs, "trapped")
    l_mean, l_std = fn(x, rl, "lost")
    return {
        "trapped_mean": t_mean,
        "trapped_std": t_std,
        "lost_mean": l_mean,
        "lost_std": l_std,
    }


def baseline_regions_mask(x, y, mode, baseline_mean, baseline_std, n_sigma=2.0):
    """
    Identify points that are "baseline" (within n_sigma of baseline_mean).
    Returns: boolean mask, True = baseline point.
    Used for splitting: baseline regions are where we can safely cut.
    """
    y = np.asarray(y)
    if mode == "trapped":
        mask = y >= baseline_mean - n_sigma * baseline_std
    else:
        mask = y <= baseline_mean + n_sigma * baseline_std
    return mask


def get_baseline_regions(x, mask, min_points=5):
    """
    Convert baseline mask to contiguous intervals [(x_lo, x_hi), ...].
    Discards regions with fewer than min_points.
    Use for splitting: cut only at these x ranges.
    """
    x = np.asarray(x)
    regions = []
    in_region = False
    start_idx = 0
    for i in range(len(mask)):
        if mask[i] and not in_region:
            start_idx = i
            in_region = True
        elif not mask[i] and in_region:
            if i - start_idx >= min_points:
                regions.append((float(x[start_idx]), float(x[i - 1])))
            in_region = False
    if in_region and len(mask) - start_idx >= min_points:
        regions.append((float(x[start_idx]), float(x[-1])))
    return regions


def _plot_baseline_selection(ax, x, y, mask, baseline_mean, baseline_std, n_sigma, mode):
    """Plot data with selected (baseline) vs non-selected points. In-place on ax."""
    x, y, mask = np.asarray(x), np.asarray(y), np.asarray(mask, dtype=bool)
    sel_x, sel_y = x[mask], y[mask]
    rej_x, rej_y = x[~mask], y[~mask]
    ax.scatter(rej_x, rej_y, c="gray", s=15, alpha=0.6, label="excluded", marker="o")
    ax.scatter(sel_x, sel_y, c="green", s=35, alpha=0.9, label="baseline", marker="s", edgecolors="darkgreen")
    ax.axhline(baseline_mean, color="blue", linestyle="--", linewidth=1.5, label=f"mean={baseline_mean:.4f}")
    ax.axhspan(baseline_mean - n_sigma * baseline_std, baseline_mean + n_sigma * baseline_std,
               alpha=0.15, color="blue")
    ax.set_ylabel(f"{mode} / loading")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)


def run_plot_baseline_selection(out_dir=None, methods=None, n_sigma_list=None):
    """
    Plot baseline-selected vs excluded points for ALL runs in the JSON.
    Saves to out_dir / {method} / n{n_sigma} / baseline_sel_{ts}_RF{rf}.png
    """
    if methods is None:
        methods = list(METHODS.keys())
    if n_sigma_list is None:
        n_sigma_list = [1.0, 1.5, 2.0]
    out_dir = out_dir or os.path.join(SCRIPT_DIR, "baseline_estimation_plots")
    data_root = os.path.join(PROJECT_ROOT, "data_rf")

    groups, _, _ = build_tickling_groups()
    all_runs = []
    for g in groups:
        rf = g["RF_amplitude"]
        for ts in g["timestamps"]:
            all_runs.append((rf, ts))

    for rf, ts in all_runs:
        try:
            x, ys = load_data(ts, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
        except Exception:
            continue
        rs = np.array(ys["ratio_signal"])
        rl = np.array(ys["ratio_lost"])

        for method in methods:
            fn = METHODS.get(method, METHODS["iterative_2.5"])
            t_mean, t_std = fn(x, rs, "trapped")
            l_mean, l_std = fn(x, rl, "lost")
            for n_sigma in n_sigma_list:
                mask_t = baseline_regions_mask(x, rs, "trapped", t_mean, t_std, n_sigma)
                mask_l = baseline_regions_mask(x, rl, "lost", l_mean, l_std, n_sigma)
                fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                _plot_baseline_selection(axs[0], x, rs, mask_t, t_mean, t_std, n_sigma, "trapped")
                _plot_baseline_selection(axs[1], x, rl, mask_l, l_mean, l_std, n_sigma, "lost")
                axs[0].set_title(f"{ts}  RF={rf} dBm  {method}  n_sigma={n_sigma}")
                axs[1].set_xlabel("Tickle Frequency (MHz)")
                subdir = os.path.join(out_dir, method, f"n{n_sigma}")
                os.makedirs(subdir, exist_ok=True)
                path = os.path.join(subdir, f"baseline_sel_{ts}_RF{rf:.1f}.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"[Plot] Saved: {path}")

    print(f"[Plot] Done. Output dir: {out_dir}")


def validate_baseline_no_peak_overlap(x, baseline_mask, peak_mus_sigmas, exclude_n_sigma=3.0, sigma_cap_mhz=15.0):
    """
    Check: do any "baseline" points fall within exclude_n_sigma * sigma of a known peak?
    sigma_cap_mhz: cap sigma to avoid bad fits (e.g. sigma=170) polluting the check.
    Returns: (ok: bool, n_violations: int)
    """
    x = np.asarray(x)
    n_violations = 0
    for mu, sig in peak_mus_sigmas:
        sig_eff = min(max(sig, 0.3), sigma_cap_mhz)  # physical peaks ~0.5-5 MHz
        half = exclude_n_sigma * sig_eff
        in_peak_zone = (x >= mu - half) & (x <= mu + half)
        baseline_in_zone = baseline_mask & in_peak_zone
        n_violations += int(np.sum(baseline_in_zone))
    return (n_violations == 0, n_violations)


if __name__ == "__main__":
    main()
