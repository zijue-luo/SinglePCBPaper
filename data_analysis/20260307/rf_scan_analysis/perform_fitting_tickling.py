"""
Final analysis for tickling/RF scan data.
Entry point: python perform_fitting_tickling.py
Phase 1: averaged fit per RF group. Phase 2: per-scan fit, incremental CSV save.
Re-exports from config, data_io, metadata, analysis, csv_export, plotting for backward compat.
"""
import csv
import os
import time

# Re-export for backward compatibility (research scripts import from here)
from config import (
    R2_THRESHOLD,
    STEP_SIZE_MHZ,
    SCAN_COUNT,
    MAX_N_PEAKS,
    MAX_NFEV,
    N_JOBS,
    FINAL_ANALYSIS_OUTPUT_DIR,
    RUN_AVERAGED_PHASE,
    USE_RF_SOURCE,
    USE_RF_UNIT,
    ACT_RF_TRIM_FRAC,
    EXCLUDE_DATES,
    DATA_SUBDIR,
    TICKLING_JSON,
    CONF_KEYS,
    CSV_HEADER,
    CSV_HEADER_PER_SCAN,
)
from data_io import (
    get_data_root as _get_data_root,
    get_tickling_json_path as _get_tickling_json_path,
    load_data,
    load_configuration,
    load_act_RF_trimmed_mean,
    load_and_average_group,
)
from metadata import (
    filter_excluded_timestamps as _filter_excluded_timestamps,
    build_tickling_groups,
    build_timestamp_meta_tickling,
    list_tickling_timestamps,
)
from analysis import analyze_fine_scan, analyze_fine_scan_from_arrays
from csv_export import fit_result_to_csv_rows, fit_result_to_csv_rows_averaged
from plotting import plot_fine_scan, plot_fine_scan_from_arrays


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(script_dir)
    data_root = os.path.join(parent, DATA_SUBDIR)
    out_dir = os.path.join(script_dir, FINAL_ANALYSIS_OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    groups, run_tag, _ = build_tickling_groups()
    timestamps = list_tickling_timestamps()
    meta_lookup = build_timestamp_meta_tickling(data_root=data_root)

    print(f"[Config] USE_RF_SOURCE={USE_RF_SOURCE}, USE_RF_UNIT={USE_RF_UNIT}")
    print(f"[Config] MAX_NFEV={MAX_NFEV}")
    print(f"[Config] EXCLUDE_DATES={EXCLUDE_DATES}")
    print(f"[Config] data_root={data_root}")
    print(f"[Config] Output dir={out_dir}")

    if not groups and not timestamps:
        print("No tickling data found.")
    else:
        # Phase 1: averaged fit per RF group
        if RUN_AVERAGED_PHASE and groups:
            csv_avg_name = f"run_best_models_averaged_{run_tag}.csv" if run_tag else "run_best_models_averaged.csv"
            csv_avg_path = os.path.join(out_dir, csv_avg_name)
            print(f"[Phase 1] Average-fit: {len(groups)} groups. Output: {csv_avg_path}")
            with open(csv_avg_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                writer.writeheader()
                for g in groups:
                    rf = g["RF_amplitude"]
                    ts_list = _filter_excluded_timestamps(g["timestamps"], EXCLUDE_DATES)
                    if not ts_list:
                        print(f"  avg RF={rf} dBm: skip (all excluded)")
                        continue
                    print(f"  avg RF={rf} dBm ({len(ts_list)} repeats)", end=" ... ")
                    x, ys_avg = load_and_average_group(ts_list, data_root=data_root)
                    y = {"lost": ys_avg["ratio_lost"], "trapped": ys_avg["ratio_signal"]}
                    t0 = time.perf_counter()
                    result_avg = analyze_fine_scan_from_arrays(x, y)
                    elapsed = time.perf_counter() - t0
                    print(f"fit {elapsed:.1f}s")
                    g_filtered = {**g, "timestamps": ts_list}
                    for row in fit_result_to_csv_rows_averaged(g_filtered, result_avg, data_root=data_root):
                        writer.writerow(row)
                    rf_str = f"{float(rf):.2f}".replace(".", "p").replace("-", "m")
                    plot_fine_scan_from_arrays(
                        x, ys_avg, result_avg,
                        os.path.join(out_dir, f"averaged_RF{rf_str}"),
                        rf_amp_label=f"{rf} dBm",
                        title_label=f"averaged (n={len(ts_list)})",
                    )
            print(f"[CSV] Saved: {csv_avg_path}")
        else:
            print("[Phase 1] Skipped.")

        # Phase 2: per-scan fit, incremental save
        timestamps = _filter_excluded_timestamps(timestamps, EXCLUDE_DATES)
        if timestamps:
            csv_name = f"run_best_models_{run_tag}.csv" if run_tag else "run_best_models.csv"
            csv_path = os.path.join(out_dir, csv_name)
            print(f"[Phase 2] Per-scan fit: {len(timestamps)} runs. Output: {csv_path} (incremental save)")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER_PER_SCAN)
                writer.writeheader()
                f.flush()
                for ts in timestamps:
                    print(f"  {ts}", end=" ... ")
                    t0 = time.perf_counter()
                    result = analyze_fine_scan(ts, data_root=data_root)
                    elapsed = time.perf_counter() - t0
                    print(f"fit {elapsed:.1f}s")
                    plot_fine_scan(ts, result, os.path.join(out_dir, ts), data_root=data_root)
                    for row in fit_result_to_csv_rows(ts, result, meta_lookup=meta_lookup, data_root=data_root):
                        writer.writerow(row)
                    f.flush()
            print(f"[CSV] Saved: {csv_path}")
        else:
            print("[Phase 2] No per-scan timestamps.")
