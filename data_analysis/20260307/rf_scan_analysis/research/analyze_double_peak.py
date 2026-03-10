"""
Analyze double-peak (base + peak) structure in tickling spectrum fits.
Extract peak spacing, sigma ratios, and implications for scan_count/resolution.
"""
import csv
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_SCAN_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(RF_SCAN_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, "run_best_models_RF_20260227_201811.csv")


def load_fits():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                n = int(row.get("n_peaks", 0) or 0)
            except ValueError:
                continue
            if n < 2:
                continue
            mus, sigmas = [], []
            for i in range(1, n + 1):
                mu = row.get(f"mu{i}")
                sigma = row.get(f"sigma{i}")
                if mu and sigma:
                    try:
                        mus.append(float(mu))
                        sigmas.append(float(sigma))
                    except ValueError:
                        break
            if len(mus) == n and len(sigmas) == n:
                rows.append({"mode": row["mode"], "n_peaks": n, "mus": mus, "sigmas": sigmas})
    return rows


def main():
    fits = load_fits()
    print(f"Loaded {len(fits)} fits with n_peaks >= 2\n")

    # 1. Adjacent peak spacing (sorted by mu)
    spacings = []
    for r in fits:
        mus = sorted(r["mus"])
        for i in range(len(mus) - 1):
            d = mus[i + 1] - mus[i]
            spacings.append(d)

    spacings = np.array(spacings)
    # Exclude outliers (bad fits: mu outside [10,150], spacing > 150)
    spacings_clean = spacings[(spacings > 0) & (spacings < 150)]
    print("=" * 60)
    print("1. ADJACENT PEAK SPACING (|mu_i - mu_j|, sorted, exclude outliers)")
    print("=" * 60)
    s = spacings_clean
    print(f"  Min:    {np.min(s):.3f} MHz")
    print(f"  P1:     {np.percentile(s, 1):.3f} MHz")
    print(f"  P5:     {np.percentile(s, 5):.3f} MHz")
    print(f"  P10:    {np.percentile(s, 10):.3f} MHz")
    print(f"  Median: {np.median(s):.3f} MHz")
    print(f"  P90:    {np.percentile(s, 90):.3f} MHz")
    n_tight = np.sum(s < 1.0)
    n_very_tight = np.sum(s < 0.5)
    print(f"  Count spacing < 1.0 MHz: {n_tight} ({100*n_tight/len(s):.1f}%)")
    print(f"  Count spacing < 0.5 MHz: {n_very_tight} ({100*n_very_tight/len(s):.1f}%)")

    # 2. Sigma ratio of adjacent peaks (broad base vs narrow peak)
    # If "base + peak": one sigma large, one small. Ratio >> 1.
    ratio_list = []
    for r in fits:
        mus = np.array(r["mus"])
        sigmas = np.array(r["sigmas"])
        ord_ = np.argsort(mus)
        mus_s = mus[ord_]
        sigmas_s = sigmas[ord_]
        for i in range(len(mus_s) - 1):
            s1, s2 = sigmas_s[i], sigmas_s[i + 1]
            if min(s1, s2) > 0.1:  # exclude bound-hitting
                ratio = max(s1, s2) / min(s1, s2)
                ratio_list.append((ratio, mus_s[i + 1] - mus_s[i], s1, s2, r["mode"], mus_s[i], mus_s[i + 1]))

    sigma_ratios = np.array([x[0] for x in ratio_list])
    print("\n" + "=" * 60)
    print("2. SIGMA RATIO (max/min of adjacent peaks)")
    print("=" * 60)
    print(f"  Median: {np.median(sigma_ratios):.2f}")
    print(f"  P90:    {np.percentile(sigma_ratios, 90):.2f}")
    print(f"  P95:    {np.percentile(sigma_ratios, 95):.2f}")
    n_ratio_high = np.sum(sigma_ratios > 3)
    print(f"  Count ratio > 3 (broad+sharp pair): {n_ratio_high} ({100*n_ratio_high/len(sigma_ratios):.1f}%)")
    n_ratio_vhigh = np.sum(sigma_ratios > 5)
    print(f"  Count ratio > 5: {n_ratio_vhigh} ({100*n_ratio_vhigh/len(sigma_ratios):.1f}%)")

    # 3. Double-peak candidates: close spacing AND high sigma ratio
    double_candidates = [x for x in ratio_list if x[1] < 3.0 and x[0] > 2.5]

    print("\n" + "=" * 60)
    print("3. DOUBLE-PEAK CANDIDATES (spacing<3 MHz AND sigma_ratio>2.5)")
    print("=" * 60)
    print(f"  Count: {len(double_candidates)}")
    if double_candidates:
        for i, (ratio, d, s1, s2, mode, m1, m2) in enumerate(double_candidates[:15]):
            print(f"    [{i+1}] {mode}: mu={m1:.1f},{m2:.1f} d={d:.2f}MHz sigma={s1:.2f},{s2:.2f} ratio={ratio:.1f}")

    # 4. Resolution requirement for scan_count
    print("\n" + "=" * 60)
    print("4. IMPLICATIONS FOR mu0 SCAN")
    print("=" * 60)
    step_120 = 120.0 / 119  # 120 points over 120 MHz
    step_80 = 120.0 / 79
    step_60 = 120.0 / 59
    print(f"  scan_count=120: mu0 step ~ {step_120:.3f} MHz")
    print(f"  scan_count=80:  mu0 step ~ {step_80:.3f} MHz")
    print(f"  scan_count=60:  mu0 step ~ {step_60:.3f} MHz")
    print(f"  Data step:      0.5 MHz")
    min_spacing = np.percentile(spacings_clean, 1)  # conservative: P1
    print(f"\n  Min adjacent spacing in data: {min_spacing:.3f} MHz")
    print(f"  To resolve a double-peak with d={min_spacing:.2f} MHz, mu0 grid should be")
    print(f"    finer than d/2 = {min_spacing/2:.3f} MHz (Nyquist-like).")
    if min_spacing / 2 < step_60:
        print(f"  WARNING: scan_count=60 (step {step_60:.2f}) may be too coarse to resolve tightest double-peaks.")
    if min_spacing / 2 < step_80:
        print(f"  WARNING: scan_count=80 (step {step_80:.2f}) may be marginal.")

    # 5. Sample raw data to check one "double-peak" region
    print("\n" + "=" * 60)
    print("5. RAW SPECTRUM SAMPLING (one tight double-peak region)")
    print("=" * 60)
    if double_candidates:
        ratio, d, s1, s2, mode, m1, m2 = double_candidates[0]
        mid = (m1 + m2) / 2
        print(f"  Example: mu1={m1:.2f}, mu2={m2:.2f}, spacing={d:.2f} MHz")
        print(f"  At 0.5 MHz data step: {int(d/0.5)} points between peaks")
        print(f"  At scan_count=60: grid points in [m1,m2]: ~{max(0,int(d/step_60))}")
        print(f"  At scan_count=120: grid points in [m1,m2]: ~{max(0,int(d/step_120))}")


if __name__ == "__main__":
    main()
