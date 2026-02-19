"""
Final analysis for fine_scan_u2 data.
Data layout: <data_root>/<YYYYMMDD>/<timestamp>_<suffix> (e.g. 20260208/20260208_000934_ratio_signal).
"""
import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from fitting_functions import fit_n_peaks, _extract_mus_from_popt, gaussian_sum


# -----------------------------------------------------------------------------
# Analysis parameters — adjust these for final analysis
# -----------------------------------------------------------------------------
R2_THRESHOLD = 0.99          # Stop adding peaks when fit R² exceeds this
STEP_SIZE_MHZ = 0.2          # Frequency step (MHz) for sigma/bounds
SCAN_COUNT = 50               # Number of initial mu to scan when adding a peak
MAX_N_PEAKS = 5               # Maximum number of Gaussian peaks per mode
N_JOBS = 8                    # Parallel fitting jobs (1 = serial)
# Output folder for this final analysis (separate from in-loop autofit_plots)
FINAL_ANALYSIS_OUTPUT_DIR = "final_analysis_plots"
# Toggle averaged-fit phase (already computed separately)
RUN_AVERAGED_PHASE = True
# CSV columns (same order as run_best_models_*.csv)
CSV_HEADER = [
    "timestamp", "mode", "U2", "RF_amplitude", "line_id", "rep", "min_scan", "max_scan",
    "n_peaks", "r2", "aicc",
    "c0", "amp1", "mu1", "sigma1", "amp2", "mu2", "sigma2", "amp3", "mu3", "sigma3", "amp4", "mu4", "sigma4",
    "amp5", "mu5", "sigma5",
]
# -----------------------------------------------------------------------------


def _get_data_root(data_root=None):
    """Root directory containing date subdirs (20260204, 20260205, ...). Default: script dir."""
    if data_root is not None:
        return os.path.abspath(data_root)
    return os.path.dirname(os.path.abspath(__file__))


def load_data(timestamp, ynames, data_root=None):
    """
    Load x (arr_of_setpoints) and requested y arrays for one run.
    timestamp: e.g. '20260208_000934'
    ynames: e.g. ['ratio_signal', 'ratio_lost']
    Returns: (x, dict of y arrays).
    """
    root = _get_data_root(data_root)
    date = timestamp[:8]
    folder = os.path.join(root, date)

    def read_col(path):
        with open(path, "r") as f:
            return np.array([float(line.strip()) for line in f if line.strip()])

    x_path = os.path.join(folder, f"{timestamp}_arr_of_setpoints")
    x = read_col(x_path)
    ys = {}
    for name in ynames:
        path = os.path.join(folder, f"{timestamp}_{name}")
        ys[name] = read_col(path)
    return x, ys


def load_configuration(timestamp, conf_names, data_root=None):
    """
    Parse _conf file (INI-like) and return values for conf_names (e.g. ['U2']).
    Returns list of values in same order as conf_names.
    """
    root = _get_data_root(data_root)
    date = timestamp[:8]
    conf_path = os.path.join(root, date, f"{timestamp}_conf")
    section_vals = {}
    current = None
    with open(conf_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                section_vals[current] = None
            elif current is not None and line.startswith("val ="):
                raw = line.split("=", 1)[1].strip().split()
                val = raw[0] if raw else None
                try:
                    section_vals[current] = float(val)
                except ValueError:
                    section_vals[current] = val
    return [section_vals.get(k) for k in conf_names]


def list_fine_scan_timestamps_from_json(json_path=None, data_root=None):
    """
    Get timestamps of fine scans only from a run_summary JSON.
    Structure: runs[].fine_scans[].timestamps (each entry is a list of timestamp strings).
    json_path: path to run_summary_*.json; if None, uses first run_summary_*.json in data_root.
    Returns: list of unique timestamps (order preserved, first occurrence).
    """
    root = _get_data_root(data_root)
    if json_path is None:
        candidates = [f for f in os.listdir(root) if f.startswith("run_summary") and f.endswith(".json")]
        if not candidates:
            return []
        json_path = os.path.join(root, sorted(candidates)[0])
    else:
        json_path = os.path.abspath(json_path)
    if not os.path.isfile(json_path):
        return []

    with open(json_path, "r") as f:
        data = json.load(f)
    runs = data.get("runs", [])
    seen = set()
    out = []
    for run in runs:
        for fs in run.get("fine_scans", []):
            for ts in fs.get("timestamps", []):
                if ts and ts not in seen:
                    seen.add(ts)
                    out.append(ts)
    return out


def build_timestamp_meta_from_json(json_path=None, data_root=None):
    """
    Build lookup: timestamp -> {U2, RF_amplitude, line_id, rep, min_scan, max_scan}
    from run_summary JSON (for CSV export). Returns (meta_lookup dict, run_tag str or None).
    """
    root = _get_data_root(data_root)
    if json_path is None:
        candidates = [f for f in os.listdir(root) if f.startswith("run_summary") and f.endswith(".json")]
        if not candidates:
            return {}, None
        json_path = os.path.join(root, sorted(candidates)[0])
    else:
        json_path = os.path.abspath(json_path)
    if not os.path.isfile(json_path):
        return {}, None

    with open(json_path, "r") as f:
        data = json.load(f)
    meta_global = data.get("meta", {})
    rf_amp = meta_global.get("RF_amplitude")
    run_tag = meta_global.get("run_tag")
    lookup = {}
    for run in data.get("runs", []):
        u2 = run.get("U2")
        for fs in run.get("fine_scans", []):
            line_id = fs.get("line_id")
            min_scan = fs.get("min_scan")
            max_scan = fs.get("max_scan")
            for rep, ts in enumerate(fs.get("timestamps", [])):
                if ts:
                    lookup[ts] = {
                        "U2": u2,
                        "RF_amplitude": rf_amp,
                        "line_id": line_id,
                        "rep": rep,
                        "min_scan": min_scan,
                        "max_scan": max_scan,
                    }
    return lookup, run_tag


def build_fine_scan_groups(json_path=None, data_root=None):
    """
    Group fine-scan repeats by (U2, line_id). Each group has same U2 and line_id
    and contains all repeat timestamps for that line.
    Returns: (groups, run_tag, rf_amplitude).
    groups = list of {"U2", "line_id", "min_scan", "max_scan", "RF_amplitude", "timestamps"}.
    """
    root = _get_data_root(data_root)
    if json_path is None:
        candidates = [f for f in os.listdir(root) if f.startswith("run_summary") and f.endswith(".json")]
        if not candidates:
            return [], None, None
        json_path = os.path.join(root, sorted(candidates)[0])
    else:
        json_path = os.path.abspath(json_path)
    if not os.path.isfile(json_path):
        return [], None, None

    with open(json_path, "r") as f:
        data = json.load(f)
    meta_global = data.get("meta", {})
    rf_amp = meta_global.get("RF_amplitude")
    run_tag = meta_global.get("run_tag")
    groups = []
    for run in data.get("runs", []):
        u2 = run.get("U2")
        for fs in run.get("fine_scans", []):
            timestamps = [t for t in fs.get("timestamps", []) if t]
            if not timestamps:
                continue
            groups.append({
                "U2": u2,
                "line_id": fs.get("line_id"),
                "min_scan": fs.get("min_scan"),
                "max_scan": fs.get("max_scan"),
                "RF_amplitude": rf_amp,
                "timestamps": timestamps,
            })
    return groups, run_tag, rf_amp


def load_and_average_group(timestamps, data_root=None):
    """
    Load ratio_signal and ratio_lost for each timestamp and average (same x for all).
    Returns: (x, ys_avg) with ys_avg = {"ratio_signal": array, "ratio_lost": array}.
    Uses x from first timestamp; assumes all have same arr_of_setpoints.
    """
    if not timestamps:
        raise ValueError("need at least one timestamp")
    x, ys_first = load_data(timestamps[0], ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
    n = len(x)
    sum_signal = np.asarray(ys_first["ratio_signal"], dtype=float)
    sum_lost = np.asarray(ys_first["ratio_lost"], dtype=float)
    for ts in timestamps[1:]:
        _, ys = load_data(ts, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
        sum_signal += np.asarray(ys["ratio_signal"], dtype=float)
        sum_lost += np.asarray(ys["ratio_lost"], dtype=float)
    k = len(timestamps)
    return x, {"ratio_signal": sum_signal / k, "ratio_lost": sum_lost / k}


def fit_result_to_csv_rows(timestamp, fit_result, meta_lookup=None, data_root=None):
    """
    Convert (timestamp, fit_result) into CSV rows (one per mode), matching run_best_models_*.csv.
    meta_lookup: optional dict from build_timestamp_meta_from_json; else U2/RF from conf, line_id/rep/min/max empty.
    Returns: list of dicts with keys = CSV_HEADER.
    """
    meta = (meta_lookup or {}).get(timestamp)
    if meta is None:
        conf = load_configuration(timestamp, conf_names=["U2", "RF_amplitude"], data_root=data_root)
        u2 = conf[0]
        rf_amp = conf[1] if len(conf) > 1 else None
        meta = {"U2": u2, "RF_amplitude": rf_amp, "line_id": "", "rep": "", "min_scan": "", "max_scan": ""}
    rows = []
    for mode in ("lost", "trapped"):
        best = fit_result.get(mode, {}).get("best")
        if best is None or not best.get("ok"):
            row = {
                "timestamp": timestamp, "mode": mode,
                "U2": meta.get("U2", ""), "RF_amplitude": meta.get("RF_amplitude", ""),
                "line_id": meta.get("line_id", ""), "rep": meta.get("rep", ""),
                "min_scan": meta.get("min_scan", ""), "max_scan": meta.get("max_scan", ""),
                "n_peaks": "", "r2": "", "aicc": "",
                "c0": "", "amp1": "", "mu1": "", "sigma1": "", "amp2": "", "mu2": "", "sigma2": "",
                "amp3": "", "mu3": "", "sigma3": "", "amp4": "", "mu4": "", "sigma4": "",
                "amp5": "", "mu5": "", "sigma5": "",
            }
            rows.append(row)
            continue
        _popt = best.get("popt")
        popt = [] if _popt is None else _popt
        n_peaks = int(best.get("n_peaks", 0))
        # popt = [c0, amp1, mu1, sigma1, amp2, mu2, sigma2, ...]
        c0 = popt[0] if len(popt) > 0 else ""
        peak_cols = {}
        for i in range(1, MAX_N_PEAKS + 1):
            j = 1 + (i - 1) * 3
            if j + 2 < len(popt) and i <= n_peaks:
                peak_cols[f"amp{i}"] = popt[j]
                peak_cols[f"mu{i}"] = popt[j + 1]
                peak_cols[f"sigma{i}"] = popt[j + 2]
            else:
                peak_cols[f"amp{i}"] = peak_cols[f"mu{i}"] = peak_cols[f"sigma{i}"] = ""
        row = {
            "timestamp": timestamp, "mode": mode,
            "U2": meta.get("U2", ""), "RF_amplitude": meta.get("RF_amplitude", ""),
            "line_id": meta.get("line_id", ""), "rep": meta.get("rep", ""),
            "min_scan": meta.get("min_scan", ""), "max_scan": meta.get("max_scan", ""),
            "n_peaks": n_peaks, "r2": best.get("r2", ""), "aicc": best.get("aicc", ""),
            "c0": c0, **peak_cols,
        }
        rows.append(row)
    return rows


def list_fine_scan_timestamps(data_root=None):
    """Discover all run timestamps from scan_list_YYYYMMDD in each date folder (any scan, not only fine)."""
    root = _get_data_root(data_root)
    out = []
    for name in sorted(os.listdir(root)):
        if not name.isdigit() or len(name) != 8:
            continue
        scan_list = os.path.join(root, name, f"scan_list_{name}")
        if not os.path.isfile(scan_list):
            continue
        with open(scan_list, "r") as f:
            for line in f:
                ts = line.strip()
                if ts:
                    out.append(ts)
    return out


def analyze_fine_scan(
    timestamp,
    stepsize=None,
    r2_gate=None,
    scan_count=None,
    max_n_peaks=None,
    n_jobs=None,
    data_root=None,
):
    """Run peak fitting for one fine scan. Uses module-level defaults if args are None."""
    stepsize = STEP_SIZE_MHZ if stepsize is None else stepsize
    r2_gate = R2_THRESHOLD if r2_gate is None else r2_gate
    scan_count = SCAN_COUNT if scan_count is None else scan_count
    max_n_peaks = MAX_N_PEAKS if max_n_peaks is None else max_n_peaks
    n_jobs = N_JOBS if n_jobs is None else n_jobs

    modes = ["lost", "trapped"]
    x, ys = load_data(timestamp, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)

    y = {"lost": ys["ratio_lost"], "trapped": ys["ratio_signal"]}
    result = {}

    for mode in modes:
        result[mode] = {"best": None, "all": []}
        n_peaks = 1

        # Generate initial guess of mu0
        if mode == "lost":
            mu0 = [float(x[np.argmax(y[mode])])]
        elif mode == "trapped":
            mu0 = [float(x[np.argmin(y[mode])])]

        while (n_peaks <= max_n_peaks):
            result_n_peaks, _ = fit_n_peaks(
                x, y[mode], n_peaks, mode,
                stepsize=stepsize,
                init_mus=mu0,
                scan_count=scan_count,
                n_jobs=n_jobs,
            )

            if result_n_peaks is None:
                print(f"[Analyze Fine Scan] Fitting for experiment {timestamp} is failed at n_peaks = {n_peaks}")
                break

            result[mode]["all"].append(result_n_peaks)
            if (result[mode]["best"] is None) or (result_n_peaks["aicc"] < result[mode]["best"]["aicc"]):
                result[mode]["best"] = result_n_peaks

            if result_n_peaks["r2"] > r2_gate:
                break
            if (n_peaks >= 2) and (result[mode]["best"] is not None) and (result_n_peaks["aicc"] > result[mode]["best"]["aicc"]):
                break
            if not result_n_peaks.get("ok") or result_n_peaks.get("popt") is None:
                break

            mu0 = _extract_mus_from_popt(result_n_peaks["popt"], n_peaks)
            n_peaks += 1
    
    return result


def analyze_fine_scan_from_arrays(
    x,
    y,
    stepsize=None,
    r2_gate=None,
    scan_count=None,
    max_n_peaks=None,
    n_jobs=None,
):
    """
    Same peak-fitting logic as analyze_fine_scan but on pre-loaded arrays.
    y: dict with keys "lost" and "trapped" (each 1D array, same length as x).
    Returns same result structure as analyze_fine_scan.
    """
    stepsize = STEP_SIZE_MHZ if stepsize is None else stepsize
    r2_gate = R2_THRESHOLD if r2_gate is None else r2_gate
    scan_count = SCAN_COUNT if scan_count is None else scan_count
    max_n_peaks = MAX_N_PEAKS if max_n_peaks is None else max_n_peaks
    n_jobs = N_JOBS if n_jobs is None else n_jobs

    x = np.asarray(x, dtype=float)
    modes = ["lost", "trapped"]
    result = {}

    for mode in modes:
        result[mode] = {"best": None, "all": []}
        n_peaks = 1
        ym = np.asarray(y[mode], dtype=float)

        if mode == "lost":
            mu0 = [float(x[np.argmax(ym)])]
        elif mode == "trapped":
            mu0 = [float(x[np.argmin(ym)])]

        while (n_peaks <= max_n_peaks):
            result_n_peaks, _ = fit_n_peaks(
                x, ym, n_peaks, mode,
                stepsize=stepsize,
                init_mus=mu0,
                scan_count=scan_count,
                n_jobs=n_jobs,
            )

            if result_n_peaks is None:
                break

            result[mode]["all"].append(result_n_peaks)
            if (result[mode]["best"] is None) or (result_n_peaks["aicc"] < result[mode]["best"]["aicc"]):
                result[mode]["best"] = result_n_peaks

            if result_n_peaks["r2"] > r2_gate:
                break
            if (n_peaks >= 2) and (result[mode]["best"] is not None) and (result_n_peaks["aicc"] > result[mode]["best"]["aicc"]):
                break
            if not result_n_peaks.get("ok") or result_n_peaks.get("popt") is None:
                break

            mu0 = _extract_mus_from_popt(result_n_peaks["popt"], n_peaks)
            n_peaks += 1

    return result


def fit_result_to_csv_rows_averaged(group_meta, fit_result, rf_amplitude=None):
    """
    Convert averaged fit result to CSV rows (one per mode), for run_best_models_averaged_*.csv.
    group_meta: dict with U2, line_id, min_scan, max_scan (and optionally RF_amplitude).
    Returns list of 2 dicts (lost, trapped) with timestamp="averaged", rep="avg".
    """
    rows = []
    for mode in ("lost", "trapped"):
        best = fit_result.get(mode, {}).get("best")
        if best is None or not best.get("ok"):
            row = {
                "timestamp": "averaged", "mode": mode,
                "U2": group_meta.get("U2", ""), "RF_amplitude": rf_amplitude or group_meta.get("RF_amplitude", ""),
                "line_id": group_meta.get("line_id", ""), "rep": "avg",
                "min_scan": group_meta.get("min_scan", ""), "max_scan": group_meta.get("max_scan", ""),
                "n_peaks": "", "r2": "", "aicc": "",
                "c0": "", "amp1": "", "mu1": "", "sigma1": "", "amp2": "", "mu2": "", "sigma2": "",
                "amp3": "", "mu3": "", "sigma3": "", "amp4": "", "mu4": "", "sigma4": "",
                "amp5": "", "mu5": "", "sigma5": "",
            }
            rows.append(row)
            continue
        _popt = best.get("popt")
        popt = [] if _popt is None else _popt
        n_peaks = int(best.get("n_peaks", 0))
        c0 = popt[0] if len(popt) > 0 else ""
        peak_cols = {}
        for i in range(1, MAX_N_PEAKS + 1):
            j = 1 + (i - 1) * 3
            if j + 2 < len(popt) and i <= n_peaks:
                peak_cols[f"amp{i}"] = popt[j]
                peak_cols[f"mu{i}"] = popt[j + 1]
                peak_cols[f"sigma{i}"] = popt[j + 2]
            else:
                peak_cols[f"amp{i}"] = peak_cols[f"mu{i}"] = peak_cols[f"sigma{i}"] = ""
        row = {
            "timestamp": "averaged", "mode": mode,
            "U2": group_meta.get("U2", ""), "RF_amplitude": rf_amplitude or group_meta.get("RF_amplitude", ""),
            "line_id": group_meta.get("line_id", ""), "rep": "avg",
            "min_scan": group_meta.get("min_scan", ""), "max_scan": group_meta.get("max_scan", ""),
            "n_peaks": n_peaks, "r2": best.get("r2", ""), "aicc": best.get("aicc", ""),
            "c0": c0, **peak_cols,
        }
        rows.append(row)
    return rows


def plot_fine_scan_from_arrays(x, y, fit_result, out_name, u2, title_label="averaged"):
    """Plot data and fit from pre-loaded arrays (e.g. averaged data). Saves to out_name.png."""
    modes = ["lost", "trapped"]
    y_dict = {"lost": y["ratio_lost"], "trapped": y["ratio_signal"]}
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    axes = {"lost": axs[0], "trapped": axs[1]}
    for mode in modes:
        ax = axes[mode]
        best_fit = fit_result[mode].get("best", None)
        if (best_fit is None) or (not best_fit.get("ok", False)):
            ax.scatter(x, y_dict[mode], label="data")
            ax.set_title(f"{title_label}: U2 = {u2:.3f} ({mode}) - [NO FIT]")
            ax.set_ylabel(f"{mode} count / loading count")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            continue
        popt = best_fit["popt"]
        n_peaks = best_fit["n_peaks"]
        xfit = np.linspace(np.min(x), np.max(x), 1000)
        yfit = gaussian_sum(xfit, *popt)
        ax.scatter(x, y_dict[mode], label="data")
        ax.plot(xfit, yfit, label=f"fitting with {n_peaks} peaks")
        ax.set_title(f"{title_label}: U2 = {u2:.3f} ({mode})")
        ax.set_ylabel(f"{mode} count / loading count")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    axes["trapped"].set_xlabel("Tickle Frequency (MHz)")
    path = f"{out_name}.png" if not out_name.lower().endswith(".png") else out_name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {path}")


def plot_fine_scan(timestamp, fit_result, out_name, data_root=None):

    modes = ["lost", "trapped"]
    x, ys = load_data(timestamp, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
    u2_val = load_configuration(timestamp, conf_names=["U2"], data_root=data_root)[0]
    u2 = float(u2_val) if u2_val is not None else float("nan")

    y = {"lost": ys["ratio_lost"], "trapped": ys["ratio_signal"]}

    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    axes = {"lost": axs[0], "trapped": axs[1]}

    for mode in modes:

        ax = axes[mode]

        best_fit = fit_result[mode].get("best", None)
        if (best_fit is None) or (not best_fit.get("ok", False)):
            ax.scatter(x, y[mode], label="data")
            ax.set_title(f"{timestamp}: U2 = {u2:.3f} ({mode}) - [NO FIT]")
            ax.set_ylabel(f"{mode} count / loading count")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            continue

        popt = best_fit["popt"]
        n_peaks = best_fit["n_peaks"]

        xfit = np.linspace(np.min(x), np.max(x), 1000)
        yfit = gaussian_sum(xfit, *popt)

        ax.scatter(x, y[mode], label="data")
        ax.plot(xfit, yfit, label=f"fitting with {n_peaks} peaks")

        ax.set_title(f"{timestamp}: U2 = {u2:.3f} ({mode})")
        ax.set_ylabel(f"{mode} count / loading count")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    axes["trapped"].set_xlabel("Tickle Frequency (MHz)")
    path = f"{out_name}.png" if not out_name.lower().endswith(".png") else out_name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[Plot] Saved: {path}")


if __name__ == "__main__":
    data_root = _get_data_root()
    out_dir = os.path.join(data_root, FINAL_ANALYSIS_OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    groups, run_tag, rf_amplitude = build_fine_scan_groups(data_root=data_root)
    timestamps = list_fine_scan_timestamps_from_json(data_root=data_root)
    meta_lookup, _ = build_timestamp_meta_from_json(data_root=data_root)

    if not groups and not timestamps:
        print("No fine-scan data found (no groups and no timestamps from run_summary_*.json).")
    else:
        # ----- Phase 1: Average-fit first (optional) -----
        if RUN_AVERAGED_PHASE and groups:
            csv_avg_name = f"run_best_models_averaged_{run_tag}.csv" if run_tag else "run_best_models_averaged_final.csv"
            csv_avg_path = os.path.join(out_dir, csv_avg_name)
            print(f"[Phase 1] Average-fit: {len(groups)} groups (U2, line_id). Output CSV: {csv_avg_path}")
            with open(csv_avg_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                writer.writeheader()
                for g in groups:
                    u2, lid = g["U2"], g["line_id"]
                    ts_list = g["timestamps"]
                    print(f"  avg U2={u2} line_id={lid} ({len(ts_list)} repeats)")
                    x, ys_avg = load_and_average_group(ts_list, data_root=data_root)
                    y = {"lost": ys_avg["ratio_lost"], "trapped": ys_avg["ratio_signal"]}
                    result_avg = analyze_fine_scan_from_arrays(x, y)
                    for row in fit_result_to_csv_rows_averaged(g, result_avg, rf_amplitude=rf_amplitude):
                        writer.writerow(row)
                    u2_str = f"{float(u2):.3f}".replace(".", "p")  # e.g. -0.21 -> -0p210, no long floats
                    plot_fine_scan_from_arrays(
                        x, ys_avg, result_avg,
                        os.path.join(out_dir, f"averaged_U2{u2_str}_line{lid}"),
                        u2, title_label=f"averaged (n={len(ts_list)})",
                    )
            print(f"[CSV] Saved: {csv_avg_path}")
        else:
            print("[Phase 1] Skipped (averaged-fit already completed elsewhere).")

        # ----- Phase 2: Piece-by-piece fit (each fine scan individually) -----
        if not timestamps:
            print("[Phase 2] No per-scan timestamps to fit.")
        else:
            csv_name = f"run_best_models_{run_tag}.csv" if run_tag else "run_best_models_final.csv"
            csv_path = os.path.join(out_dir, csv_name)
            print(f"[Phase 2] Per-scan fit: {len(timestamps)} runs. Output: {out_dir}, CSV: {csv_path}")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                writer.writeheader()
                for ts in timestamps:
                    print(f"  {ts}")
                    result = analyze_fine_scan(ts, data_root=data_root)
                    plot_fine_scan(ts, result, os.path.join(out_dir, ts), data_root=data_root)
                    for row in fit_result_to_csv_rows(ts, result, meta_lookup=meta_lookup, data_root=data_root):
                        writer.writerow(row)
            print(f"[CSV] Saved: {csv_path}")