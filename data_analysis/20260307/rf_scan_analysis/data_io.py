"""
Data loading for tickling/RF scan.
Reads column files from data_rf/<YYYYMMDD>/<timestamp>_<col>. Paths resolved via get_data_root.
"""
import os
import numpy as np

from config import (
    DATA_SUBDIR,
    TICKLING_JSON,
    ACT_RF_TRIM_FRAC,
    CONF_KEYS,
)


def get_data_root(data_root=None):
    """Return root dir for data (parent/data_rf by default)."""
    if data_root is not None:
        return os.path.abspath(data_root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(script_dir)
    return os.path.join(parent, DATA_SUBDIR)


def get_tickling_json_path(json_path=None):
    """Return path to tickling metadata JSON."""
    if json_path is not None:
        return os.path.abspath(json_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(script_dir)
    return os.path.join(parent, TICKLING_JSON)


def load_data(timestamp, ynames, data_root=None, extra_names=None):
    """
    Load x (arr_of_setpoints) and requested y arrays for one run.
    Returns: (x, dict of y arrays).
    """
    root = get_data_root(data_root)
    date = timestamp[:8]
    folder = os.path.join(root, date)

    def read_col(path):
        with open(path, "r", encoding="utf-8") as f:
            return np.array([float(line.strip()) for line in f if line.strip()])

    x_path = os.path.join(folder, f"{timestamp}_arr_of_setpoints")
    x = read_col(x_path)
    ys = {}
    for name in ynames:
        path = os.path.join(folder, f"{timestamp}_{name}")
        ys[name] = read_col(path)
    if extra_names:
        for name in extra_names:
            path = os.path.join(folder, f"{timestamp}_{name}")
            if os.path.isfile(path):
                ys[name] = read_col(path)
    return x, ys


def load_configuration(timestamp, conf_names, data_root=None):
    """Parse INI-style _conf and return values for conf_names."""
    root = get_data_root(data_root)
    date = timestamp[:8]
    conf_path = os.path.join(root, date, f"{timestamp}_conf")
    section_vals = {}
    current = None
    with open(conf_path, "r", encoding="utf-8") as f:
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


def load_act_RF_trimmed_mean(timestamp, data_root=None, trim_frac=None):
    """Trimmed mean of act_RF_amplitude (dBm). None if file missing."""
    if trim_frac is None:
        trim_frac = ACT_RF_TRIM_FRAC
    root = get_data_root(data_root)
    date = timestamp[:8]
    path = os.path.join(root, date, f"{timestamp}_act_RF_amplitude")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        arr = np.array([float(line.strip()) for line in f if line.strip()])
    if len(arr) == 0:
        return None
    k = max(1, int(len(arr) * trim_frac))
    trimmed = np.sort(arr)[k : len(arr) - k] if len(arr) > 2 * k else arr
    return float(np.mean(trimmed))


def load_and_average_group(timestamps, data_root=None):
    """Load and average ratio_signal, ratio_lost across timestamps. Returns (x, ys_avg)."""
    if not timestamps:
        raise ValueError("need at least one timestamp")
    x, ys_first = load_data(
        timestamps[0],
        ynames=["ratio_signal", "ratio_lost"],
        data_root=data_root,
    )
    sum_signal = np.asarray(ys_first["ratio_signal"], dtype=float)
    sum_lost = np.asarray(ys_first["ratio_lost"], dtype=float)
    for ts in timestamps[1:]:
        _, ys = load_data(ts, ynames=["ratio_signal", "ratio_lost"], data_root=data_root)
        sum_signal += np.asarray(ys["ratio_signal"], dtype=float)
        sum_lost += np.asarray(ys["ratio_lost"], dtype=float)
    k = len(timestamps)
    return x, {"ratio_signal": sum_signal / k, "ratio_lost": sum_lost / k}
