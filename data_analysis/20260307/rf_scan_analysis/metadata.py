"""
Metadata and grouping for tickling experiments.
Builds RF groups from tickling JSON, timestamp lookup, and date exclusion for fitting pipeline.
"""
import os
import json

from config import EXCLUDE_DATES
from data_io import get_tickling_json_path, load_act_RF_trimmed_mean


def filter_excluded_timestamps(timestamps, exclude_dates=None):
    """Filter out timestamps whose YYYYMMDD prefix is in exclude_dates."""
    if not exclude_dates:
        return list(timestamps)
    return [ts for ts in timestamps if (ts[:8] if len(ts) >= 8 else ts) not in exclude_dates]


def build_tickling_groups(json_path=None):
    """
    Build groups from tickling_experiment_run_job_list.json RF.setpoints.
    Returns: (groups, run_tag, tickle_range).
    groups = list of {"RF_amplitude", "U2", "min_scan", "max_scan", "timestamps"}.
    """
    jpath = get_tickling_json_path(json_path)
    if not os.path.isfile(jpath):
        return [], None, None
    with open(jpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    run_tag = data.get("run_tag")
    tickle_range = data.get("tickle_range", [20.0, 140.0, 241])
    min_scan = tickle_range[0] if len(tickle_range) >= 1 else 20.0
    max_scan = tickle_range[1] if len(tickle_range) >= 2 else 140.0

    groups = []
    rf = data.get("RF", {})
    for sp in rf.get("setpoints", []):
        timestamps = [t for t in sp.get("timestamps", []) if t]
        if not timestamps:
            continue
        groups.append({
            "RF_amplitude": sp.get("RF_amplitude"),
            "U2": sp.get("U2", -0.35),
            "min_scan": min_scan,
            "max_scan": max_scan,
            "timestamps": timestamps,
        })
    return groups, run_tag, tickle_range


def build_timestamp_meta_tickling(json_path=None, data_root=None):
    """Build lookup: timestamp -> meta dict (RF, U2, rep, etc). Loads act_RF per timestamp."""
    groups, _, tickle_range = build_tickling_groups(json_path)
    min_scan = tickle_range[0] if tickle_range and len(tickle_range) >= 1 else 20.0
    max_scan = tickle_range[1] if tickle_range and len(tickle_range) >= 2 else 140.0

    lookup = {}
    for g in groups:
        rf_set = g["RF_amplitude"]
        u2 = g["U2"]
        for rep, ts in enumerate(g["timestamps"]):
            act_dBm = load_act_RF_trimmed_mean(ts, data_root=data_root)
            lookup[ts] = {
                "RF_amplitude_setpoint_dBm": rf_set,
                "RF_amplitude_act_trimmed_dBm": act_dBm,
                "U2": u2,
                "min_scan": min_scan,
                "max_scan": max_scan,
                "line_id": 0,
                "rep": rep,
            }
    return lookup


def list_tickling_timestamps(json_path=None):
    """All timestamps from tickling JSON RF setpoints."""
    groups, _, _ = build_tickling_groups(json_path)
    seen = set()
    out = []
    for g in groups:
        for ts in g.get("timestamps", []):
            if ts and ts not in seen:
                seen.add(ts)
                out.append(ts)
    return out
