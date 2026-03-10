"""
CSV export for fit results.
Two formats: averaged (merged peaks) and per-scan (left+right segment params in same row).
"""
import numpy as np

from config import MAX_N_PEAKS, MAX_N_PEAKS_MERGED, CONF_KEYS, choose_RF_value
from data_io import load_configuration, load_act_RF_trimmed_mean


def _segment_to_peak_cols(best_seg, prefix):
    """Extract peak columns from segment fit. prefix is '_L' or '_R'."""
    cols = {}
    for i in range(1, MAX_N_PEAKS + 1):
        cols[f"amp{i}{prefix}"] = cols[f"mu{i}{prefix}"] = cols[f"sigma{i}{prefix}"] = ""
    if best_seg is None or not best_seg.get("ok"):
        return cols
    popt = best_seg.get("popt")
    if popt is None:
        return cols
    popt = list(popt)
    n_peaks = int(best_seg.get("n_peaks", 0))
    for i in range(1, MAX_N_PEAKS + 1):
        j = 1 + (i - 1) * 3
        if j + 2 < len(popt) and i <= n_peaks:
            cols[f"amp{i}{prefix}"] = popt[j]
            cols[f"mu{i}{prefix}"] = popt[j + 1]
            cols[f"sigma{i}{prefix}"] = popt[j + 2]
    return cols


def fit_result_to_csv_rows(timestamp, fit_result, meta_lookup=None, data_root=None):
    """Convert fit_result to CSV rows (one per mode). Left+right segment params in same row.
    Includes split_x_MHz (cut point for left/right segment stitching).
    """
    meta = (meta_lookup or {}).get(timestamp)
    if meta is None:
        conf = load_configuration(timestamp, conf_names=CONF_KEYS, data_root=data_root)
        rf_set = conf[0] if len(conf) > 0 else None
        u2 = conf[1] if len(conf) > 1 else -0.35
        min_s = conf[2] if len(conf) > 2 else 20.0
        max_s = conf[3] if len(conf) > 3 else 140.0
        act_dBm = load_act_RF_trimmed_mean(timestamp, data_root=data_root)
        meta = {
            "RF_amplitude_setpoint_dBm": rf_set,
            "RF_amplitude_act_trimmed_dBm": act_dBm,
            "U2": u2,
            "min_scan": min_s,
            "max_scan": max_s,
            "line_id": 0,
            "rep": "",
        }
    rf_set_dBm = meta.get("RF_amplitude_setpoint_dBm")
    rf_act_dBm = meta.get("RF_amplitude_act_trimmed_dBm")
    _, rf_val = choose_RF_value(rf_set_dBm or 0.0, rf_act_dBm)

    split_x = fit_result.get("split_x")
    split_x_str = float(split_x) if split_x is not None else ""

    def make_empty_row():
        left_cols = {}
        for i in range(1, MAX_N_PEAKS + 1):
            left_cols[f"amp{i}_L"] = left_cols[f"mu{i}_L"] = left_cols[f"sigma{i}_L"] = ""
        right_cols = {}
        for i in range(1, MAX_N_PEAKS + 1):
            right_cols[f"amp{i}_R"] = right_cols[f"mu{i}_R"] = right_cols[f"sigma{i}_R"] = ""
        return {
            "timestamp": timestamp, "mode": "",
            "U2": meta.get("U2", ""), "RF_amplitude": rf_val,
            "line_id": meta.get("line_id", ""), "rep": meta.get("rep", ""),
            "min_scan": meta.get("min_scan", ""), "max_scan": meta.get("max_scan", ""),
            "split_x_MHz": split_x_str,
            "baseline_left_mean": "", "baseline_left_std": "", "baseline_right_mean": "", "baseline_right_std": "",
            "y_min": "", "y_max": "",
            "n_peaks_L": "", "n_peaks_R": "", "r2_L": "", "r2_R": "", "aicc_L": "", "aicc_R": "",
            "c0_L": "", "c0_R": "", **left_cols, **right_cols,
            "RF_setpoint_dBm": rf_set_dBm if rf_set_dBm is not None else "",
            "RF_act_trimmed_dBm": rf_act_dBm if rf_act_dBm is not None else "",
        }

    rows = []
    for mode in ("lost", "trapped"):
        mode_data = fit_result.get(mode, {})
        best_l = mode_data.get("best_left")
        best_r = mode_data.get("best_right")
        if best_l is None and best_r is None:
            row = make_empty_row()
            row["mode"] = mode
            rows.append(row)
            continue

        n_l = int(best_l.get("n_peaks", 0)) if best_l and best_l.get("ok") else 0
        n_r = int(best_r.get("n_peaks", 0)) if best_r and best_r.get("ok") else 0
        r2_l = best_l.get("r2", "") if best_l and best_l.get("ok") else ""
        r2_r = best_r.get("r2", "") if best_r and best_r.get("ok") else ""
        aicc_l = best_l.get("aicc", "") if best_l and best_l.get("ok") else ""
        aicc_r = best_r.get("aicc", "") if best_r and best_r.get("ok") else ""
        popt_l = best_l.get("popt") if best_l and best_l.get("ok") else None
        popt_r = best_r.get("popt") if best_r and best_r.get("ok") else None
        c0_l = popt_l[0] if popt_l is not None and len(popt_l) > 0 else ""
        c0_r = popt_r[0] if popt_r is not None and len(popt_r) > 0 else ""

        left_cols = _segment_to_peak_cols(best_l, "_L")
        right_cols = _segment_to_peak_cols(best_r, "_R")

        def _v(k, d=""):
            v = mode_data.get(k)
            return v if v is not None and (not isinstance(v, float) or not np.isnan(v)) else d

        row = {
            "timestamp": timestamp, "mode": mode,
            "U2": meta.get("U2", ""), "RF_amplitude": rf_val,
            "line_id": meta.get("line_id", ""), "rep": meta.get("rep", ""),
            "min_scan": meta.get("min_scan", ""), "max_scan": meta.get("max_scan", ""),
            "split_x_MHz": split_x_str,
            "baseline_left_mean": _v("baseline_left_mean"), "baseline_left_std": _v("baseline_left_std"),
            "baseline_right_mean": _v("baseline_right_mean"), "baseline_right_std": _v("baseline_right_std"),
            "y_min": _v("y_min"), "y_max": _v("y_max"),
            "n_peaks_L": n_l, "n_peaks_R": n_r, "r2_L": r2_l, "r2_R": r2_r,
            "aicc_L": aicc_l, "aicc_R": aicc_r,
            "c0_L": c0_l, "c0_R": c0_r,
            **left_cols, **right_cols,
            "RF_setpoint_dBm": rf_set_dBm if rf_set_dBm is not None else "",
            "RF_act_trimmed_dBm": rf_act_dBm if rf_act_dBm is not None else "",
        }
        rows.append(row)
    return rows


def fit_result_to_csv_rows_averaged(group_meta, fit_result, data_root=None):
    """Convert averaged fit result to CSV rows. Uses merged peak format (not left+right)."""
    rf_set_dBm = group_meta.get("RF_amplitude")
    u2 = group_meta.get("U2", -0.35)
    min_s = group_meta.get("min_scan", 20.0)
    max_s = group_meta.get("max_scan", 140.0)
    ts_list = group_meta.get("timestamps", [])
    rf_act_dBm = None
    if ts_list:
        act_vals = [load_act_RF_trimmed_mean(ts, data_root=data_root) for ts in ts_list]
        act_vals = [v for v in act_vals if v is not None]
        rf_act_dBm = float(np.mean(act_vals)) if act_vals else None
    _, rf_val = choose_RF_value(rf_set_dBm or 0.0, rf_act_dBm)

    split_x = fit_result.get("split_x")
    split_x_str = float(split_x) if split_x is not None else ""

    def make_empty():
        peak_empty = {}
        for i in range(1, MAX_N_PEAKS_MERGED + 1):
            peak_empty[f"amp{i}"] = peak_empty[f"mu{i}"] = peak_empty[f"sigma{i}"] = ""
        return {
            "timestamp": "averaged", "mode": "",
            "U2": u2, "RF_amplitude": "",
            "line_id": 0, "rep": "avg",
            "min_scan": min_s, "max_scan": max_s,
            "split_x_MHz": split_x_str,
            "baseline_left_mean": "", "baseline_left_std": "", "baseline_right_mean": "", "baseline_right_std": "",
            "y_min": "", "y_max": "",
            "n_peaks": "", "r2": "", "aicc": "",
            "c0": "", **peak_empty,
            "RF_setpoint_dBm": rf_set_dBm if rf_set_dBm is not None else "",
            "RF_act_trimmed_dBm": rf_act_dBm if rf_act_dBm is not None else "",
        }

    rows = []
    for mode in ("lost", "trapped"):
        best = fit_result.get(mode, {}).get("best")
        if best is None or not best.get("ok"):
            row = make_empty()
            row["mode"] = mode
            row["RF_amplitude"] = rf_val
            rows.append(row)
            continue
        popt = best.get("popt")
        popt = [] if popt is None else list(popt)
        n_peaks = int(best.get("n_peaks", 0))
        c0 = popt[0] if len(popt) > 0 else ""
        peak_cols = {}
        for i in range(1, MAX_N_PEAKS_MERGED + 1):
            j = 1 + (i - 1) * 3
            if j + 2 < len(popt) and i <= n_peaks:
                peak_cols[f"amp{i}"] = popt[j]
                peak_cols[f"mu{i}"] = popt[j + 1]
                peak_cols[f"sigma{i}"] = popt[j + 2]
            else:
                peak_cols[f"amp{i}"] = peak_cols[f"mu{i}"] = peak_cols[f"sigma{i}"] = ""
        mode_data = fit_result.get(mode, {})
        def _v(k, d=""):
            v = mode_data.get(k)
            return v if v is not None and (not isinstance(v, float) or not np.isnan(v)) else d

        row = {
            "timestamp": "averaged", "mode": mode,
            "U2": u2, "RF_amplitude": rf_val,
            "line_id": 0, "rep": "avg",
            "min_scan": min_s, "max_scan": max_s,
            "split_x_MHz": split_x_str,
            "baseline_left_mean": _v("baseline_left_mean"), "baseline_left_std": _v("baseline_left_std"),
            "baseline_right_mean": _v("baseline_right_mean"), "baseline_right_std": _v("baseline_right_std"),
            "y_min": _v("y_min"), "y_max": _v("y_max"),
            "n_peaks": n_peaks, "r2": best.get("r2", ""), "aicc": best.get("aicc", ""),
            "c0": c0, **peak_cols,
            "RF_setpoint_dBm": rf_set_dBm if rf_set_dBm is not None else "",
            "RF_act_trimmed_dBm": rf_act_dBm if rf_act_dBm is not None else "",
        }
        rows.append(row)
    return rows
