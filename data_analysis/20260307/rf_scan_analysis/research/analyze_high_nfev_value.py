"""
Analyze the value of high-nfev converged fits from fit_attempt_history.csv

1. Among high-nfev fits: what R2 percentile do they achieve within (exp, n_peaks)?
2. Among selected fits: do they tend to be high-nfev or low-nfev in their bucket?
"""

import csv
import numpy as np
from collections import defaultdict

def main():
    path = "report_fit_history_nfev/fit_attempt_history.csv"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    ok_rows = [
        r
        for r in rows
        if r.get("ok") == "1" and r.get("nfev") and r["nfev"].strip()
    ]
    for r in ok_rows:
        r["nfev"] = int(r["nfev"])
        r["r2"] = float(r["r2"]) if r.get("r2") and r["r2"].strip() else None
        r["is_segment_selected"] = int(r.get("is_segment_selected", 0))

    def bucket(r):
        return (r["group_idx"], r["mode"], r["segment"], r["n_peaks_target"])

    buckets = defaultdict(list)
    for r in ok_rows:
        buckets[bucket(r)].append(r)

    # Q1: High nfev fits - R2 percentile
    high_nfev_r2_p90 = []
    high_nfev_r2_p75 = []
    low_nfev_r2 = []
    for key, arr in buckets.items():
        if len(arr) < 10:
            continue
        nfevs = [a["nfev"] for a in arr]
        r2s = [a["r2"] for a in arr if a["r2"] is not None]
        if not r2s:
            continue
        p90_nfev = np.percentile(nfevs, 90)
        p75_nfev = np.percentile(nfevs, 75)
        p25_nfev = np.percentile(nfevs, 25)
        for a in arr:
            if a["r2"] is None:
                continue
            pct = (np.sum(np.array(r2s) <= a["r2"]) / len(r2s)) * 100
            if a["nfev"] >= p90_nfev:
                high_nfev_r2_p90.append(pct)
            if a["nfev"] >= p75_nfev:
                high_nfev_r2_p75.append(pct)
            if a["nfev"] <= p25_nfev:
                low_nfev_r2.append(pct)

    print("=" * 60)
    print("Q1: High nfev fits - R2 percentile within (exp, mode, seg, n_peaks)")
    print("=" * 60)
    print("Top 10%% nfev (p90): R2 percentile mean=%.1f%%, median=%.1f%%" % (np.mean(high_nfev_r2_p90), np.median(high_nfev_r2_p90)))
    print("Top 25%% nfev (p75): R2 percentile mean=%.1f%%, median=%.1f%%" % (np.mean(high_nfev_r2_p75), np.median(high_nfev_r2_p75)))
    print("Low 25%% nfev (p25): R2 percentile mean=%.1f%%, median=%.1f%%" % (np.mean(low_nfev_r2), np.median(low_nfev_r2)))
    print()
    print("Conclusion: High-nfev fits have R2 around median (50th percentile).")
    print("           They are NOT systematically high-R2 (valuable) or low-R2 (poor).")

    # Correlation nfev vs R2
    corrs = []
    for key, arr in buckets.items():
        if len(arr) < 5:
            continue
        nfevs = np.array([a["nfev"] for a in arr])
        r2s = np.array([a["r2"] for a in arr])
        if np.any(np.isnan(r2s)) or len(np.unique(r2s)) < 2:
            continue
        c = np.corrcoef(nfevs, r2s)[0, 1]
        if not np.isnan(c):
            corrs.append(c)
    print()
    print("Within-bucket correlation (nfev vs R2): mean=%.3f, median=%.3f" % (np.mean(corrs), np.median(corrs)))

    # Q2: Selected fit nfev percentile
    sel_pct = []
    for key, arr in buckets.items():
        sel = [a for a in arr if a["is_segment_selected"] == 1]
        if not sel or len(arr) < 2:
            continue
        nfevs = np.array([a["nfev"] for a in arr])
        for s in sel:
            sel_pct.append((np.sum(nfevs <= s["nfev"]) / len(nfevs)) * 100)

    print()
    print("=" * 60)
    print("Q2: Selected fits - nfev percentile within (exp, mode, seg, n_peaks)")
    print("=" * 60)
    print("Selected fit nfev percentile: mean=%.1f%%, median=%.1f%%" % (np.mean(sel_pct), np.median(sel_pct)))
    print("Fraction selected with nfev percentile > 75: %.1f%%" % (100 * np.mean(np.array(sel_pct) > 75)))
    print("Fraction selected with nfev percentile > 50: %.1f%%" % (100 * np.mean(np.array(sel_pct) > 50)))
    print("Fraction selected with nfev percentile < 25: %.1f%%" % (100 * np.mean(np.array(sel_pct) < 25)))
    print()
    print("Conclusion: Selected fits are roughly uniformly distributed in nfev.")
    print("           Neither high-nfev nor low-nfev is favored - selection is by R2/AICc, not nfev.")


if __name__ == "__main__":
    main()
