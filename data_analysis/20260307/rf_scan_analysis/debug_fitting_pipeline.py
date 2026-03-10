"""
Debug fitting pipeline: trace split -> guesses -> segment fits -> combine.
Use raw data for RF=-0.5 and RF=0.0 to find where things go wrong.
"""
import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(script_dir)
data_root = os.path.join(parent, "data_rf")
sys.path.insert(0, script_dir)

from perform_fitting_tickling import (
    build_tickling_groups,
    load_and_average_group,
    analyze_fine_scan_from_arrays,
    load_data,
)
from split_experiment import split_experiment, initial_peaks_guess
from perform_fitting_tickling import (
    _fit_segment_incremental,
    _combine_segment_fits,
    STEP_SIZE_MHZ,
    R2_THRESHOLD,
    SCAN_COUNT,
    MAX_N_PEAKS,
    MAX_NFEV,
)


def debug_group(rf_val, ts_list, label):
    print("\n" + "=" * 70)
    print(f"DEBUG: {label} (RF={rf_val} dBm, n={len(ts_list)} reps)")
    print("=" * 70)

    x, ys_avg = load_and_average_group(ts_list, data_root=data_root)
    rs = np.asarray(ys_avg["ratio_signal"])
    rl = np.asarray(ys_avg["ratio_lost"])
    y = {"lost": rl, "trapped": rs}

    # 1. Split
    split_res = split_experiment(x, rs, rl)
    split_x = split_res.get("split_x")
    sidx = split_res.get("split_idx")
    print(f"\n1. SPLIT: split_x={split_x:.2f} MHz, split_idx={sidx}")

    mask_l = x <= split_x
    mask_r = x > split_x
    x_l, x_r = x[mask_l], x[mask_r]
    n_l, n_r = len(x_l), len(x_r)
    print(f"   Left segment: {n_l} pts, x=[{x_l[0]:.1f}, {x_l[-1]:.1f}] MHz")
    print(f"   Right segment: {n_r} pts, x=[{x_r[0]:.1f}, {x_r[-1]:.1f}] MHz")

    # 2. Initial guesses
    guesses = initial_peaks_guess(x, rs, rl)
    g_lost = guesses["lost"]
    g_left = [p for p in g_lost if p[1] <= split_x]
    g_right = [p for p in g_lost if p[1] > split_x]
    print(f"\n2. LOST initial guesses: total {len(g_lost)}, left {len(g_left)}, right {len(g_right)}")
    print(f"   Left mu:  {[f'{p[1]:.1f}' for p in g_left[:8]]}")
    print(f"   Right mu: {[f'{p[1]:.1f}' for p in g_right[:8]]}")

    # Where is the dominant peak (argmax of rl)?
    imax = np.argmax(rl)
    mu_dom = float(x[imax])
    print(f"   DOMINANT (argmax rl): mu={mu_dom:.1f} MHz, val={rl[imax]:.4f}")

    # 3. Segment fits for LOST
    y_l = rl[mask_l]
    y_r = rl[mask_r]
    scan_left = max(1, int(SCAN_COUNT * n_l / (n_l + n_r)))
    scan_right = SCAN_COUNT - scan_left

    best_l = _fit_segment_incremental(
        x_l, y_l, "lost", g_left, STEP_SIZE_MHZ, R2_THRESHOLD,
        scan_left, 6, 1, MAX_NFEV, return_history=False  # max 6 peaks for faster debug
    )
    best_r = _fit_segment_incremental(
        x_r, y_r, "lost", g_right, STEP_SIZE_MHZ, R2_THRESHOLD,
        scan_right, 6, 1, MAX_NFEV, return_history=False  # max 6 peaks for faster debug
    )

    print(f"\n3. LOST segment fits:")
    if best_l:
        popt_l = best_l["popt"]
        mus_l = [popt_l[1 + 3 * j + 1] for j in range(best_l["n_peaks"])]
        amps_l = [popt_l[1 + 3 * j] for j in range(best_l["n_peaks"])]
        print(f"   Left:  n_peaks={best_l['n_peaks']}, r2={best_l['r2']:.4f}")
        print(f"          mus={[f'{m:.1f}' for m in mus_l]}, amps={[f'{a:.4f}' for a in amps_l]}")
    else:
        print(f"   Left:  None (fit failed)")
    if best_r:
        popt_r = best_r["popt"]
        mus_r = [popt_r[1 + 3 * j + 1] for j in range(best_r["n_peaks"])]
        amps_r = [popt_r[1 + 3 * j] for j in range(best_r["n_peaks"])]
        print(f"   Right: n_peaks={best_r['n_peaks']}, r2={best_r['r2']:.4f}")
        print(f"          mus={[f'{m:.1f}' for m in mus_r]}, amps={[f'{a:.4f}' for a in amps_r]}")
    else:
        print(f"   Right: None (fit failed)")

    # 4. Combine
    best_merged = _combine_segment_fits(x, rl, best_l, best_r, "lost")
    if best_merged:
        popt = best_merged["popt"]
        mus_m = [popt[1 + 3 * j + 1] for j in range(best_merged["n_peaks"])]
        amps_m = [popt[1 + 3 * j] for j in range(best_merged["n_peaks"])]
        print(f"\n4. MERGED LOST: n_peaks={best_merged['n_peaks']}, r2={best_merged['r2']:.4f}")
        print(f"   mus ={[f'{m:.1f}' for m in mus_m]}")
        print(f"   amps={[f'{a:.4f}' for a in amps_m]}")

    # 5. Check: does left segment contain dominant peak?
    if mu_dom <= split_x:
        seg = "LEFT"
        amps = rl[mask_l]
    else:
        seg = "RIGHT"
        amps = rl[mask_r]
    print(f"\n5. Dominant peak at {mu_dom:.1f} MHz is in {seg} segment.")


def main():
    groups, _, _ = build_tickling_groups()
    # RF=-0.5 and RF=0.0
    for g in groups:
        rf = g["RF_amplitude"]
        if rf in (-0.5, 0.0):
            debug_group(rf, g["timestamps"], f"RF={rf}")


if __name__ == "__main__":
    main()
