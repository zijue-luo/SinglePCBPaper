"""
Configuration and constants for tickling/RF scan analysis.
Changing parameters here takes effect in the main pipeline. CSV headers depend on MAX_N_PEAKS.
"""
import numpy as np

# -----------------------------------------------------------------------------
# Fitting control
# -----------------------------------------------------------------------------
R2_THRESHOLD = 0.995   # stop adding peaks when fit R² exceeds this
STEP_SIZE_MHZ = 0.5    # frequency step for sigma bounds
SCAN_COUNT = 120       # initial mu scan points when adding a new peak
MAX_N_PEAKS = 9
MAX_NFEV = 4000
N_JOBS = 8
FINAL_ANALYSIS_OUTPUT_DIR = "final_analysis_plots"
RUN_AVERAGED_PHASE = True

# RF value source: setpoint = JSON config, act = act_RF_amplitude file
USE_RF_SOURCE = "setpoint"
USE_RF_UNIT = "dBm"
ACT_RF_TRIM_FRAC = 0.1

# Exclude dates: timestamps with prefix YYYYMMDD in this list are skipped
EXCLUDE_DATES = ["20260228", "20260301"]

# Paths
DATA_SUBDIR = "data_rf"
TICKLING_JSON = "tickling_experiment_run_job_list.json"
CONF_KEYS = ["RF_amplitude", "U2", "min_scan", "max_scan"]


# -----------------------------------------------------------------------------
# CSV headers (averaged = merged format, up to 2*MAX_N_PEAKS; per-scan = left+right segment format)
# -----------------------------------------------------------------------------
MAX_N_PEAKS_MERGED = 2 * MAX_N_PEAKS  # merged left+right can have up to 18 peaks


def _csv_header():
    peak_cols = []
    for i in range(1, MAX_N_PEAKS_MERGED + 1):
        peak_cols.extend([f"amp{i}", f"mu{i}", f"sigma{i}"])
    return [
        "timestamp", "mode", "U2", "RF_amplitude", "line_id", "rep", "min_scan", "max_scan",
        "split_x_MHz",
        "baseline_left_mean", "baseline_left_std", "baseline_right_mean", "baseline_right_std",
        "y_min", "y_max",
        "n_peaks", "r2", "aicc",
        "c0", *peak_cols,
        "RF_setpoint_dBm", "RF_act_trimmed_dBm",
    ]


def _csv_header_per_scan():
    """Per-scan CSV: left+right segment fits in same row, 2*MAX_N_PEAKS peak slots."""
    left_cols = []
    for i in range(1, MAX_N_PEAKS + 1):
        left_cols.extend([f"amp{i}_L", f"mu{i}_L", f"sigma{i}_L"])
    right_cols = []
    for i in range(1, MAX_N_PEAKS + 1):
        right_cols.extend([f"amp{i}_R", f"mu{i}_R", f"sigma{i}_R"])
    return [
        "timestamp", "mode", "U2", "RF_amplitude", "line_id", "rep", "min_scan", "max_scan",
        "split_x_MHz",
        "baseline_left_mean", "baseline_left_std", "baseline_right_mean", "baseline_right_std",
        "y_min", "y_max",
        "n_peaks_L", "n_peaks_R", "r2_L", "r2_R", "aicc_L", "aicc_R",
        "c0_L", "c0_R",
        *left_cols, *right_cols,
        "RF_setpoint_dBm", "RF_act_trimmed_dBm",
    ]


CSV_HEADER = _csv_header()
CSV_HEADER_PER_SCAN = _csv_header_per_scan()


# -----------------------------------------------------------------------------
# RF unit conversion and value selection for CSV
# -----------------------------------------------------------------------------
def _dBm_to_mW(dbm):
    return 10.0 ** (float(dbm) / 10.0)


def _mW_to_dBm(mw):
    if mw <= 0:
        return float("-inf")
    return 10.0 * np.log10(float(mw))


def choose_RF_value(rf_setpoint_dBm, rf_act_dBm):
    """Return (value_dBm, value_for_csv) based on USE_RF_SOURCE and USE_RF_UNIT."""
    use_setpoint = USE_RF_SOURCE == "setpoint"
    dbm = float(rf_setpoint_dBm) if use_setpoint or rf_act_dBm is None else float(rf_act_dBm)
    if rf_act_dBm is None and not use_setpoint:
        dbm = float(rf_setpoint_dBm)
    if USE_RF_UNIT == "linear":
        return dbm, _dBm_to_mW(dbm)
    return dbm, dbm
