"""
Test timing and results for 3 timestamps (after scan_count allocation change).
Fixed params: SCAN_COUNT=120 (full), MAX_N_PEAKS=9, R2_THRESHOLD=0.995.
No time limit.
"""
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_SCAN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, RF_SCAN_DIR)

from perform_fitting_tickling import list_tickling_timestamps, analyze_fine_scan

SCAN_COUNT = 120
MAX_N_PEAKS = 9
R2_THRESHOLD = 0.995


def main():
    parent = os.path.dirname(RF_SCAN_DIR)
    data_root = os.path.join(parent, "data_rf")
    json_path = os.path.join(parent, "tickling_experiment_run_job_list.json")

    timestamps = list_tickling_timestamps(json_path=json_path)
    if len(timestamps) < 3:
        print(f"Need at least 3 timestamps, got {len(timestamps)}")
        return

    test_ts = timestamps[:3]
    print(f"Testing 3 timestamps: {test_ts}")
    print(f"data_root: {data_root}")
    print(f"scan_count={SCAN_COUNT} max_n_peaks={MAX_N_PEAKS} r2_threshold={R2_THRESHOLD}")
    print("-" * 60)

    total_start = time.perf_counter()
    results = []
    for ts in test_ts:
        t0 = time.perf_counter()
        r = analyze_fine_scan(
            ts,
            data_root=data_root,
            scan_count=SCAN_COUNT,
            max_n_peaks=MAX_N_PEAKS,
            r2_gate=R2_THRESHOLD,
        )
        elapsed = time.perf_counter() - t0
        results.append((ts, elapsed, r))
        print(f"  {ts}: {elapsed:.2f}s")
    total_elapsed = time.perf_counter() - total_start

    print("-" * 60)
    print(f"Total: {total_elapsed:.2f}s  ({total_elapsed/3:.2f}s avg per run)")
    print("-" * 60)

    print("\nResults (n_peaks, r2, aicc):")
    for ts, elapsed, r in results:
        for mode in ("lost", "trapped"):
            best = r.get(mode, {}).get("best")
            if best and best.get("ok"):
                print(f"  {ts} {mode}: n={best['n_peaks']}  r2={best['r2']:.6f}  aicc={best['aicc']:.2f}")
            else:
                print(f"  {ts} {mode}: [NO FIT]")


if __name__ == "__main__":
    main()
