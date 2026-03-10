"""
A/B test: max_nfev = 20000 (baseline) vs 10000 vs 5000
Compares: fit success, n_peaks, R², mu/sigma values, and wall-clock time.
Fixed params: SCAN_COUNT=120, MAX_N_PEAKS=9, R2_THRESHOLD=0.995 (no limit on runtime).
"""
import os
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_SCAN_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(RF_SCAN_DIR)
sys.path.insert(0, RF_SCAN_DIR)

from perform_fitting_tickling import (
    build_tickling_groups,
    load_and_average_group,
    analyze_fine_scan_from_arrays,
)

# Fixed params for this study
SCAN_COUNT = 120
MAX_N_PEAKS = 9
R2_THRESHOLD = 0.995

# Use first 3 groups for representative A/B test
N_GROUPS = 3


def extract_summary(result):
    """Extract n_peaks, r2, aicc, mus, sigmas from fit result."""
    out = {}
    for mode in ("lost", "trapped"):
        best = result.get(mode, {}).get("best")
        if best is None or not best.get("ok"):
            out[mode] = {"ok": False, "n_peaks": None, "r2": None, "aicc": None, "mus": [], "sigmas": []}
            continue
        popt = best.get("popt")
        if popt is None:
            out[mode] = {"ok": False}
            continue
        popt = np.asarray(popt)
        n = int(best.get("n_peaks", 0))
        mus = [float(popt[1 + 3 * i + 1]) for i in range(n) if 1 + 3 * i + 1 < len(popt)]
        sigmas = [float(popt[1 + 3 * i + 2]) for i in range(n) if 1 + 3 * i + 2 < len(popt)]
        out[mode] = {
            "ok": True,
            "n_peaks": n,
            "r2": best.get("r2"),
            "aicc": best.get("aicc"),
            "mus": mus,
            "sigmas": sigmas,
        }
    return out


def compare_summaries(base, test, label):
    """Compare test vs base. Return (n_diff, r2_diff, mu_diff_str)."""
    diffs = []
    for mode in ("lost", "trapped"):
        b, t = base.get(mode, {}), test.get(mode, {})
        if not b.get("ok") or not t.get("ok"):
            if b.get("ok") != t.get("ok"):
                diffs.append(f"{mode}: ok {b.get('ok')}->{t.get('ok')}")
            continue
        if b.get("n_peaks") != t.get("n_peaks"):
            diffs.append(f"{mode} n_peaks {b.get('n_peaks')}->{t.get('n_peaks')}")
        if b.get("r2") is not None and t.get("r2") is not None:
            dr2 = t["r2"] - b["r2"]
            if abs(dr2) > 1e-6:
                diffs.append(f"{mode} R2 {b['r2']:.6f}->{t['r2']:.6f} (dR2={dr2:+.6f})")
        bmus, tmus = b.get("mus", []), t.get("mus", [])
        if len(bmus) == len(tmus) and len(bmus) > 0:
            mu_diffs = [abs(tmus[i] - bmus[i]) for i in range(len(bmus))]
            max_mu_diff = max(mu_diffs)
            if max_mu_diff > 0.01:
                diffs.append(f"{mode} max|Δμ|={max_mu_diff:.3f} MHz")
    return diffs


def main():
    groups, _, _ = build_tickling_groups()
    if not groups:
        print("No groups found.")
        return
    groups = groups[:N_GROUPS]
    print(f"A/B test: max_nfev on {N_GROUPS} groups")
    print(f"Fixed: SCAN_COUNT={SCAN_COUNT} MAX_N_PEAKS={MAX_N_PEAKS} R2_THRESHOLD={R2_THRESHOLD}")
    print("=" * 60)

    configs = [
        (20000, "baseline"),
        (10000, "10k"),
        (5000, "5k"),
    ]

    results = {}  # max_nfev -> list of (summary, time_s) per group
    for max_nfev, label in configs:
        results[max_nfev] = []
        print(f"\n--- max_nfev={max_nfev} ({label}) ---")
        for i, g in enumerate(groups):
            ts_list = g["timestamps"]
            rf = g["RF_amplitude"]
            t0 = time.perf_counter()
            x, ys_avg = load_and_average_group(ts_list)
            y = {"lost": ys_avg["ratio_lost"], "trapped": ys_avg["ratio_signal"]}
            result = analyze_fine_scan_from_arrays(
                x, y,
                scan_count=SCAN_COUNT,
                max_n_peaks=MAX_N_PEAKS,
                r2_gate=R2_THRESHOLD,
                max_nfev=max_nfev,
            )
            t1 = time.perf_counter()
            summary = extract_summary(result)
            elapsed = t1 - t0
            results[max_nfev].append((summary, elapsed))
            ok_l = summary["lost"]["ok"]
            ok_t = summary["trapped"]["ok"]
            n_l = summary["lost"].get("n_peaks", "?")
            n_t = summary["trapped"].get("n_peaks", "?")
            r2_l = summary["lost"].get("r2")
            r2_t = summary["trapped"].get("r2")
            r2_str = f"L={r2_l:.4f} T={r2_t:.4f}" if r2_l and r2_t else ""
            print(f"  RF={rf} dBm: ok=({ok_l},{ok_t}) n_peaks=({n_l},{n_t}) {r2_str} time={elapsed:.1f}s")

    # Compare vs baseline
    print("\n" + "=" * 60)
    print("COMPARISON vs baseline (max_nfev=20000)")
    print("=" * 60)
    base = 20000
    for max_nfev, label in configs:
        if max_nfev == base:
            continue
        total_time_base = sum(r[1] for r in results[base])
        total_time_test = sum(r[1] for r in results[max_nfev])
        speedup = total_time_base / total_time_test if total_time_test > 0 else 0
        diffs_all = []
        for i in range(N_GROUPS):
            diffs = compare_summaries(
                results[base][i][0], results[max_nfev][i][0], f"group{i}"
            )
            diffs_all.extend(diffs)
        print(f"\nmax_nfev={max_nfev} ({label}):")
        print(f"  Time: {total_time_base:.1f}s -> {total_time_test:.1f}s (speedup {speedup:.2f}x)")
        if diffs_all:
            print(f"  Differences: {len(diffs_all)}")
            for d in diffs_all[:10]:
                print(f"    - {d}")
            if len(diffs_all) > 10:
                print(f"    ... and {len(diffs_all)-10} more")
        else:
            print(f"  Differences: NONE (results match baseline)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    t20 = sum(r[1] for r in results[20000])
    t10 = sum(r[1] for r in results[10000])
    t5 = sum(r[1] for r in results[5000])
    print(f"Total time: 20k={t20:.1f}s  10k={t10:.1f}s  5k={t5:.1f}s")
    print(f"Speedup 10k: {t20/t10:.2f}x  5k: {t20/t5:.2f}x")


if __name__ == "__main__":
    main()
