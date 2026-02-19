"""
Load and analyze fitting results from run_best_models_*.csv files.
Separate trapped/lost, clean peaks (group by proximity, keep one per group by composite max).
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Optional: for baseline fluctuation from raw data
try:
    from perform_fitting import load_data, load_and_average_group, build_fine_scan_groups
    from fitting_functions import gaussian_sum
    _HAS_RAW_DATA_LOADER = True
except ImportError:
    _HAS_RAW_DATA_LOADER = False


# Default filenames (actual files use "models" plural)
# Uncomment only the dataset(s) you have; RUN_WHICH (below) must match.
DEFAULT_FILES = {
    "auto": "run_best_models_auto.csv",
    "improved": "run_best_models_improved.csv",
    "averaged": "run_best_models_averaged.csv",
}

# ---------------------------------------------------------------------------
# Run mode when executing this script (e.g. "Run" button or python data_cleaning.py)
# ---------------------------------------------------------------------------
# RUN_WHICH : "all" | "averaged" | "auto" | "improved"
#     "all" = load and clean every dataset listed in DEFAULT_FILES above.
#     Otherwise = load and clean only that one (must be a key in DEFAULT_FILES).
RUN_WHICH = "averaged"


# ---------------------------------------------------------------------------
# Data cleaning parameters (adjust here to tune peak selection and filtering)
# ---------------------------------------------------------------------------
#
# Grouping
# --------
# GROUPING_FACTOR : float
#     Two consecutive peaks (in mu) are merged into the same group if their
#     separation <= GROUPING_FACTOR * sqrt(sigma_i^2 + sigma_j^2). Larger
#     values group more aggressively so multiple fitted peaks for one
#     physical line are merged; one peak per group is then chosen by
#     composite value at center (bump: max, dip: min).
# MAX_GROUPING_DISTANCE : float
#     Cap on the grouping threshold (same units as mu, e.g. MHz). The
#     effective threshold is min(factor * sqrt(sigma_i^2 + sigma_j^2), this cap).
#     Prevents distant peaks (e.g. ~80 MHz vs ~62 MHz) from being merged when
#     one peak has a very large sigma.
# MAX_SIGMA_FOR_REPRESENTATIVE : float or None
#     Peaks with sigma > this value (same units as mu, e.g. MHz) are excluded
#     from the peak list entirely before grouping—they never enter any group.
#     Use to drop broad fit artifacts (e.g. sigma ~140 MHz). None = no filter.
# MIN_REPRESENTATIVE_AMP_FRACTION : float or None
#     Representative peak's amplitude must be >= this fraction of the group's
#     strength (composite maximum - baseline, or baseline - composite minimum
#     for dips). E.g. 0.2 = at least 20%. None = no filter.
#
# Amplitude (minimum strength to keep a group)
# --------------------------------------------
# AMP_FACTOR1 : float
#     Term1 = AMP_FACTOR1 * baseline_fluctuation (SNR-style bar). Higher
#     values require a stronger peak above noise.
# AMP_FACTOR2 : float
#     Term2 = AMP_FACTOR2 * (composite_max - baseline) (relative to full
#     scan range). Higher values require a larger fraction of the scan’s
#     dynamic range.
# USE_MIN_THRESHOLD : bool
#     If True, keep group if group_amplitude >= min(term1, term2)
#     (satisfy either bar). If False, require >= max(term1, term2) (both).
# AMP_INCLUDE_BASELINE : bool
#     If True, term1 is c0 + AMP_FACTOR1*fluctuation; else term1 is
#     AMP_FACTOR1*fluctuation only. Rarely needed.
#
# Prominence (peak must stand out from sides)
# -------------------------------------------
# PROMINENCE_FLUCTUATION_FACTOR : float
#     Prominence required >= this factor * baseline_fluctuation. Lower
#     values make the prominence check looser.
# PROMINENCE_COMPOSITE_FACTOR : float
#     Prominence required >= this factor * (group_composite_max - baseline).
#     Lower values make the check looser.
#
# Local maximum (peak vs immediate neighbors)
# ------------------------------------------
# LOCAL_MAX_FRACTION : float in (0, 1]
#     Selected peak is accepted if its composite value is at least this
#     fraction of the max of the two neighboring points. 1.0 = strict
#     local max; smaller (e.g. 0.9) allows overlapping peaks that blur
#     the top.
#
GROUPING_FACTOR = 4.0
MAX_GROUPING_DISTANCE = 10.0
MAX_SIGMA_FOR_REPRESENTATIVE = 10.0  # peaks with sigma > this are excluded from groups
MIN_REPRESENTATIVE_AMP_FRACTION = 0.2  # representative |amp| >= this fraction of group strength
AMP_FACTOR1 = 6.0
AMP_FACTOR2 = 0.20
USE_MIN_THRESHOLD = True
AMP_INCLUDE_BASELINE = False
PROMINENCE_FLUCTUATION_FACTOR = 0.0
PROMINENCE_COMPOSITE_FACTOR = 0.0
LOCAL_MAX_FRACTION = 1.0


def load_fitting_results(
    data_dir: str | Path | None = None,
    which: Literal["all", "auto", "improved", "averaged"] = "all",
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """
    Load fitting result CSVs into pandas DataFrames.

    Parameters
    ----------
    data_dir : str | Path | None, optional
        Directory containing the CSV files. If None, uses the directory
        of this script.
    which : {"all", "auto", "improved", "averaged"}, optional
        Which dataset(s) to load. "all" returns a dict with all three;
        otherwise returns the single DataFrame.

    Returns
    -------
    dict[str, pd.DataFrame] | pd.DataFrame
        If which="all", returns {"auto": df, "improved": df, "averaged": df}.
        Otherwise returns the single DataFrame.

    Raises
    ------
    FileNotFoundError
        If any requested CSV file does not exist.
    """
    data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent

    datasets = list(DEFAULT_FILES.keys()) if which == "all" else [which]
    for name in datasets:
        if name not in DEFAULT_FILES:
            raise KeyError(f"Unknown dataset '{which}'; must be one of {list(DEFAULT_FILES.keys())} or 'all'.")
    result: dict[str, pd.DataFrame] = {}

    for name in datasets:
        filepath = data_dir / DEFAULT_FILES[name]
        if not filepath.exists():
            raise FileNotFoundError(f"Fitting result file not found: {filepath}")
        df = pd.read_csv(filepath)
        result[name] = df

    if which == "all":
        return result
    return result[which]


# ---------------------------------------------------------------------------
# Data separation (trapped / lost) and peak cleaning
# ---------------------------------------------------------------------------

def _estimate_fluctuation_from_raw_data(
    df: pd.DataFrame,
    triplets: list[tuple[str, str, str]],
    data_root: Path,
    groups_lookup: list[dict] | None = None,
) -> float | None:
    """
    Estimate baseline fluctuation from raw scan data as std of fit residuals.
    For each row: load x,y, compute fit, residual_std = std(y - fit).
    Returns median of per-scan residual stds, or None if raw data unavailable.
    """
    if not _HAS_RAW_DATA_LOADER or df.empty:
        return None
    data_root = Path(data_root)
    residual_stds = []
    for _, row in df.iterrows():
        ts = str(row.get("timestamp", ""))
        mode = str(row.get("mode", ""))
        n_peaks = int(row.get("n_peaks", 0))
        if n_peaks <= 0:
            continue
        try:
            popt = _row_to_popt(row, triplets, n_peaks)
            if len(popt) < 4:  # need at least c0 + one peak
                continue
            if ts == "averaged":
                if groups_lookup is None:
                    continue
                u2, lid = row.get("U2"), row.get("line_id")
                group = None
                for g in groups_lookup:
                    if g.get("U2") == u2 and g.get("line_id") == lid:
                        group = g
                        break
                if group is None or not group.get("timestamps"):
                    continue
                x, ys_avg = load_and_average_group(group["timestamps"], data_root=str(data_root))
                y = ys_avg["ratio_lost"] if mode == "lost" else ys_avg["ratio_signal"]
            else:
                ynames = ["ratio_signal", "ratio_lost"]
                x, ys = load_data(ts, ynames, data_root=str(data_root))
                y = ys["ratio_lost"] if mode == "lost" else ys["ratio_signal"]
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if len(x) != len(y) or len(x) == 0:
                continue
            y_fit = gaussian_sum(x, *popt)
            residual = y - y_fit
            std_val = float(np.std(residual))
            if std_val > 0 and np.isfinite(std_val):
                residual_stds.append(std_val)
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
            continue
    if not residual_stds:
        return None
    return float(np.median(residual_stds))


# Identifier columns to carry through cleaning (no peak-specific columns)
SCAN_ID_COLUMNS = [
    "timestamp", "mode", "U2", "RF_amplitude", "line_id", "rep",
    "min_scan", "max_scan", "n_peaks", "r2", "aicc", "c0",
]


def _get_peak_triplets(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return list of (amp_col, mu_col, sigma_col) for all peak columns in df."""
    triplets = []
    i = 1
    while True:
        amp_c, mu_c, sigma_c = f"amp{i}", f"mu{i}", f"sigma{i}"
        if amp_c not in df.columns or mu_c not in df.columns or sigma_c not in df.columns:
            break
        triplets.append((amp_c, mu_c, sigma_c))
        i += 1
    return triplets


def _row_to_popt(row: pd.Series, triplets: list[tuple[str, str, str]], n_peaks: int) -> list[float]:
    """Build popt = [c0, amp1, mu1, sigma1, ...] from row for gaussian_sum."""
    c0 = float(row["c0"]) if pd.notna(row["c0"]) else 0.0
    popt = [c0]
    for i, (ac, mc, sc) in enumerate(triplets):
        if i >= n_peaks:
            break
        a, m, s = row[ac], row[mc], row[sc]
        if pd.isna(a) or pd.isna(m) or pd.isna(s):
            break
        popt.extend([float(a), float(m), float(s)])
    return popt


def _peaks_from_row(row: pd.Series, triplets: list[tuple[str, str, str]], n_peaks: int) -> list[tuple[float, float, float]]:
    """Extract (amp, mu, sigma) for the first n_peaks valid peaks in row."""
    out = []
    for i, (ac, mc, sc) in enumerate(triplets):
        if i >= n_peaks:
            break
        amp, mu, sigma = row[ac], row[mc], row[sc]
        if pd.isna(amp) or pd.isna(mu) or pd.isna(sigma):
            break
        out.append((float(amp), float(mu), float(sigma)))
    return out


def _gaussian(x: float, amp: float, mu: float, sigma: float) -> float:
    """Single Gaussian: amp * exp(-(x-mu)^2 / (2*sigma^2))."""
    if sigma <= 0:
        return 0.0
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _composite_at_x(x: float, c0: float, peaks: list[tuple[float, float, float]]) -> float:
    """Composite function at x: baseline c0 + sum of Gaussians."""
    return c0 + sum(_gaussian(x, a, m, s) for a, m, s in peaks)


def _composite_maximum(
    c0: float,
    peaks: list[tuple[float, float, float]],
    x_min: float,
    x_max: float,
    n_grid: int = 1000,
) -> float:
    """Maximum of composite function over [x_min, x_max]."""
    if not peaks:
        return c0
    x = np.linspace(x_min, x_max, n_grid)
    values = np.array([_composite_at_x(xi, c0, peaks) for xi in x])
    return float(np.max(values))


def _composite_minimum(
    c0: float,
    peaks: list[tuple[float, float, float]],
    x_min: float,
    x_max: float,
    n_grid: int = 1000,
) -> float:
    """Minimum of composite function over [x_min, x_max]."""
    if not peaks:
        return c0
    x = np.linspace(x_min, x_max, n_grid)
    values = np.array([_composite_at_x(xi, c0, peaks) for xi in x])
    return float(np.min(values))


def _group_close_peaks(
    peaks_sorted: list[tuple[float, float, float]],
    grouping_factor: float | None = None,
) -> list[list[tuple[float, float, float]]]:
    """
    Group peaks so that two consecutive (in mu) peaks are in the same group
    iff separation <= grouping_factor * sqrt(sigma_i^2 + sigma_j^2).

    Larger grouping_factor (e.g. 3.0) groups more aggressively so multiple
    fitted peaks for one physical line are merged before selecting one.
    """
    if grouping_factor is None:
        grouping_factor = GROUPING_FACTOR
    if not peaks_sorted:
        return []
    groups: list[list[tuple[float, float, float]]] = [[peaks_sorted[0]]]
    for (amp, mu, sigma) in peaks_sorted[1:]:
        prev_amp, prev_mu, prev_sigma = groups[-1][-1]
        sep = abs(mu - prev_mu)
        threshold = grouping_factor * np.sqrt(prev_sigma**2 + sigma**2)
        threshold = min(threshold, MAX_GROUPING_DISTANCE)
        if sep <= threshold:
            groups[-1].append((amp, mu, sigma))
        else:
            groups.append([(amp, mu, sigma)])
    return groups


def _select_peak_per_group(
    c0: float,
    group: list[tuple[float, float, float]],
    group_strength: float | None = None,
    min_representative_amp_fraction: float | None = None,
) -> tuple[float, float, float]:
    """
    Within a group, choose the peak whose center gives the most extreme
    composite function value (baseline + sum of all Gaussians in group).

    For bumps (positive amplitudes): select the peak whose center has the
    maximum composite value. For dips (negative amplitudes): select the
    peak whose center has the minimum composite value (the dominant dip).

    If min_representative_amp_fraction and group_strength are set, only consider
    peaks whose |amplitude| >= that fraction of group_strength (group strength
    = composite max - baseline or baseline - composite min).
    """
    if len(group) == 1:
        return group[0]
    candidates = group
    if (
        min_representative_amp_fraction is not None
        and group_strength is not None
        and group_strength > 0
    ):
        min_amp = min_representative_amp_fraction * group_strength
        strong = [(a, m, s) for (a, m, s) in candidates if abs(a) >= min_amp]
        if strong:
            candidates = strong
    if len(candidates) == 1:
        return candidates[0]
    values_at_centers = [_composite_at_x(mu, c0, group) for (_, mu, _) in candidates]
    vals = np.array(values_at_centers)
    is_bump = (np.max(vals) - c0) >= (c0 - np.min(vals))
    idx = int(np.argmax(vals) if is_bump else np.argmin(vals))
    return candidates[idx]


def _amplitude_threshold(
    c0: float,
    composite_max: float,
    fluctuation_baseline: float,
    factor1: float = 6.0,
    factor2: float = 0.15,
    include_baseline_in_threshold: bool = False,
    use_min_threshold: bool = False,
) -> float:
    """
    Minimum acceptable amplitude (for group composite) to keep a group.
    term1 = factor1*fluctuation (SNR bar), term2 = factor2*(composite_max - baseline) (relative bar).
    If use_min_threshold: keep if group_amp >= min(term1, term2) — satisfy either criterion.
    Else (default): keep if group_amp >= max(term1, term2) — satisfy both.
    """
    if include_baseline_in_threshold:
        term1 = c0 + factor1 * fluctuation_baseline
        term2 = c0 + factor2 * (composite_max - c0)
    else:
        term1 = factor1 * fluctuation_baseline
        term2 = factor2 * (composite_max - c0)
    return max(term1, term2) if not use_min_threshold else min(term1, term2)


def _load_raw_xy_for_row(
    row: pd.Series,
    data_root: Path,
    groups_lookup: list[dict] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load raw (x, y) arrays for a fitted scan row."""
    if not _HAS_RAW_DATA_LOADER:
        return None
    ts = str(row.get("timestamp", ""))
    mode = str(row.get("mode", ""))
    try:
        if ts == "averaged":
            if groups_lookup is None:
                return None
            u2 = row.get("U2")
            line_id = row.get("line_id")
            group = next(
                (g for g in groups_lookup if g.get("U2") == u2 and g.get("line_id") == line_id),
                None,
            )
            if group is None or not group.get("timestamps"):
                return None
            x, ys_avg = load_and_average_group(group["timestamps"], data_root=str(data_root))
            y = ys_avg["ratio_lost"] if mode == "lost" else ys_avg["ratio_signal"]
        else:
            ynames = ["ratio_signal", "ratio_lost"]
            x, ys = load_data(ts, ynames, data_root=str(data_root))
            y = ys["ratio_lost"] if mode == "lost" else ys["ratio_signal"]
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0 or x.size != y.size:
            return None
        return x, y
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
        return None


def _group_prominence_ok(
    c0: float,
    group: list[tuple[float, float, float]],
    mu_peak: float,
    fluctuation_baseline: float,
    group_composite_max: float,
    x_min: float,
    x_max: float,
    raw_xy: tuple[np.ndarray, np.ndarray] | None = None,
    prominence_fluctuation_factor: float | None = None,
    prominence_composite_factor: float | None = None,
    local_max_fraction: float | None = None,
) -> bool:
    """
    Require selected peak to be a clear local maximum with prominence on both sides.
    Prominence threshold = max(prominence_fluctuation_factor*fluctuation,
                               prominence_composite_factor*(group_composite_max - c0)).
    """
    if prominence_fluctuation_factor is None:
        prominence_fluctuation_factor = PROMINENCE_FLUCTUATION_FACTOR
    if prominence_composite_factor is None:
        prominence_composite_factor = PROMINENCE_COMPOSITE_FACTOR
    if local_max_fraction is None:
        local_max_fraction = LOCAL_MAX_FRACTION
    if not group:
        return False
    peak_value = _composite_at_x(mu_peak, c0, group)
    sigma_max = max((sigma for (_, _, sigma) in group if sigma > 0), default=0.0)
    span = max(3.0 * sigma_max, 0.5)

    group_x_min = min((mu - 3.0 * sigma for (_, mu, sigma) in group), default=mu_peak)
    group_x_max = max((mu + 3.0 * sigma for (_, mu, sigma) in group), default=mu_peak)
    left_start = max(x_min, group_x_min, mu_peak - span)
    right_end = min(x_max, group_x_max, mu_peak + span)

    if left_start >= mu_peak or right_end <= mu_peak:
        return False

    prominence_required = max(
        prominence_fluctuation_factor * fluctuation_baseline,
        prominence_composite_factor * max(group_composite_max - c0, 0.0),
    )
    if prominence_required <= 0:
        return True

    # Evaluate composite on dense grids left/right of the peak
    xs_left = np.linspace(left_start, mu_peak, 200, endpoint=False)
    xs_right = np.linspace(mu_peak, right_end, 200, endpoint=False)[1:]

    if xs_left.size == 0 or xs_right.size == 0:
        return False

    vals_left = np.array([_composite_at_x(x, c0, group) for x in xs_left])
    vals_right = np.array([_composite_at_x(x, c0, group) for x in xs_right])
    left_min = float(np.min(vals_left))
    right_min = float(np.min(vals_right))

    left_prominence = peak_value - left_min
    right_prominence = peak_value - right_min

    if raw_xy is not None:
        x_raw, y_raw = raw_xy
        if x_raw[0] <= left_start and x_raw[-1] >= right_end:
            y_left = np.interp(xs_left, x_raw, y_raw)
            y_right = np.interp(xs_right, x_raw, y_raw)
            y_peak = float(np.interp(mu_peak, x_raw, y_raw))
            left_prominence = y_peak - float(np.min(y_left))
            right_prominence = y_peak - float(np.min(y_right))
    if left_prominence < prominence_required or right_prominence < prominence_required:
        return False

    # Ensure local maximum (peak at least comparable to immediate neighbors; looser tolerance)
    eps_left = min(mu_peak - left_start, span) * 0.1
    eps_right = min(right_end - mu_peak, span) * 0.1
    eps_left = max(eps_left, 1e-3)
    eps_right = max(eps_right, 1e-3)

    left_val = _composite_at_x(mu_peak - eps_left, c0, group)
    right_val = _composite_at_x(mu_peak + eps_right, c0, group)
    neighbor_max = max(left_val, right_val)
    if peak_value < local_max_fraction * neighbor_max:
        return False

    return True


def clean_peaks_one_row(
    row: pd.Series,
    triplets: list[tuple[str, str, str]],
    id_columns: list[str],
    grouping_factor: float | None = None,
    fluctuation_baseline: float = 0.0,
    amp_factor1: float | None = None,
    amp_factor2: float | None = None,
    amp_include_baseline: bool | None = None,
    use_min_threshold: bool | None = None,
    raw_xy: tuple[np.ndarray, np.ndarray] | None = None,
    prominence_fluctuation_factor: float | None = None,
    prominence_composite_factor: float | None = None,
    local_max_fraction: float | None = None,
) -> list[dict]:
    """
    For a single scan row: extract peaks, sort by mu, group by proximity,
    select one peak per group by composite maximum, filter by amplitude.
    Return list of dicts (one per selected peak) with id columns + amp, mu, sigma.

    grouping_factor : float
        Multiplier for sqrt(sigma_i^2 + sigma_j^2) when deciding if two
        consecutive peaks belong to the same group.
    fluctuation_baseline : float
        Std of baseline (c0) across the dataset; used for amplitude filter.
    amp_factor1, amp_factor2 : float
        Groups kept if group_composite_amplitude >= threshold.
    use_min_threshold : bool
        If True, keep group if amplitude >= min(term1, term2); else >= max(term1, term2).
    prominence_fluctuation_factor, prominence_composite_factor : float
        Prominence required = max(fluctuation_factor*fluctuation, composite_factor*(group_max - c0)).
        Lower values make the prominence check looser.
    """
    if grouping_factor is None:
        grouping_factor = GROUPING_FACTOR
    if amp_factor1 is None:
        amp_factor1 = AMP_FACTOR1
    if amp_factor2 is None:
        amp_factor2 = AMP_FACTOR2
    if amp_include_baseline is None:
        amp_include_baseline = AMP_INCLUDE_BASELINE
    if use_min_threshold is None:
        use_min_threshold = USE_MIN_THRESHOLD
    if prominence_fluctuation_factor is None:
        prominence_fluctuation_factor = PROMINENCE_FLUCTUATION_FACTOR
    if prominence_composite_factor is None:
        prominence_composite_factor = PROMINENCE_COMPOSITE_FACTOR
    if local_max_fraction is None:
        local_max_fraction = LOCAL_MAX_FRACTION
    n_peaks = int(row["n_peaks"])
    peaks = _peaks_from_row(row, triplets, n_peaks)
    if not peaks:
        return []

    peaks_sorted = sorted(peaks, key=lambda p: p[1])  # by mu
    if MAX_SIGMA_FOR_REPRESENTATIVE is not None and MAX_SIGMA_FOR_REPRESENTATIVE > 0:
        peaks_sorted = [(a, m, s) for (a, m, s) in peaks_sorted if s <= MAX_SIGMA_FOR_REPRESENTATIVE]
    if not peaks_sorted:
        return []
    groups = _group_close_peaks(peaks_sorted, grouping_factor=grouping_factor)
    c0 = float(row["c0"]) if pd.notna(row["c0"]) else 0.0
    x_min = float(row["min_scan"]) if pd.notna(row["min_scan"]) else 0.0
    x_max = float(row["max_scan"]) if pd.notna(row["max_scan"]) else 100.0

    composite_max = _composite_maximum(c0, peaks, x_min, x_max)
    min_amp = _amplitude_threshold(
        c0,
        composite_max,
        fluctuation_baseline,
        amp_factor1,
        amp_factor2,
        include_baseline_in_threshold=amp_include_baseline,
        use_min_threshold=use_min_threshold,
    )

    selected = []
    for i, g in enumerate(groups):
        group_composite_max = _composite_maximum(c0, g, x_min, x_max)
        group_composite_min = _composite_minimum(c0, g, x_min, x_max)
        group_strength = max(
            group_composite_max - c0,
            c0 - group_composite_min,
        )
        rep = _select_peak_per_group(
            c0,
            g,
            group_strength=group_strength,
            min_representative_amp_fraction=MIN_REPRESENTATIVE_AMP_FRACTION,
        )
        selected.append((rep, group_composite_max, group_composite_min))

    out = []
    for i, ((amp, mu, sigma), group_composite_max, group_composite_min) in enumerate(selected):
        # Filter by group composite amplitude (handles both positive and negative peaks)
        group_composite_amplitude = max(
            group_composite_max - c0,
            c0 - group_composite_min,
        )
        if group_composite_amplitude < min_amp:
            continue
        if not _group_prominence_ok(
            c0=c0,
            group=groups[i],
            mu_peak=mu,
            fluctuation_baseline=fluctuation_baseline,
            group_composite_max=group_composite_max,
            x_min=x_min,
            x_max=x_max,
            raw_xy=raw_xy,
            prominence_fluctuation_factor=prominence_fluctuation_factor,
            prominence_composite_factor=prominence_composite_factor,
            local_max_fraction=local_max_fraction,
        ):
            continue
        rec = {c: row[c] for c in id_columns if c in row.index}
        rec["amp"] = amp
        rec["mu"] = mu
        rec["sigma"] = sigma
        # Weighted average of line width in group (weight = amp * sigma) for error-bar option in plotting
        group_peaks = groups[i]
        w_sum = sum(a * s for a, _, s in group_peaks)
        if w_sum > 0:
            rec["sigma_group"] = sum(a * s * s for a, _, s in group_peaks) / w_sum
        else:
            rec["sigma_group"] = sigma
        out.append(rec)
    return out


def separate_trapped_lost(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a fitting-result DataFrame into trapped and lost.
    No overlap: each row appears in exactly one of the two.
    """
    trapped = df.loc[df["mode"] == "trapped"].copy()
    lost = df.loc[df["mode"] == "lost"].copy()
    return trapped, lost


def clean_dataset_peaks(
    df: pd.DataFrame,
    id_columns: list[str] | None = None,
    grouping_factor: float | None = None,
    amp_factor1: float | None = None,
    amp_factor2: float | None = None,
    amp_include_baseline: bool | None = None,
    use_min_threshold: bool | None = None,
    data_root: str | Path | None = None,
    groups_lookup: list[dict] | None = None,
    prominence_fluctuation_factor: float | None = None,
    prominence_composite_factor: float | None = None,
    local_max_fraction: float | None = None,
) -> pd.DataFrame:
    """
    Run peak cleaning on a single DataFrame (e.g. trapped or lost only).
    Returns a DataFrame where each row is one identified real peak.

    data_root : str | Path | None
        Root dir for raw scan data; if set, baseline fluctuation is estimated
        from fit residuals in raw data (fallback: MAD of c0).
    groups_lookup : list[dict] | None
        For "averaged" dataset: list from build_fine_scan_groups to load
        averaged raw data.
    """
    if id_columns is None:
        id_columns = [c for c in SCAN_ID_COLUMNS if c in df.columns]
    triplets = _get_peak_triplets(df)
    if not triplets:
        return pd.DataFrame()

    # Baseline fluctuation: from raw data residuals when available
    data_root_path = Path(data_root) if data_root is not None else Path(__file__).resolve().parent
    fluctuation_baseline = None
    if data_root is not None:
        fluctuation_baseline = _estimate_fluctuation_from_raw_data(
            df, triplets, data_root_path, groups_lookup=groups_lookup
        )
    if fluctuation_baseline is None or fluctuation_baseline <= 0:
        c0_median = df["c0"].median()
        mad = np.median(np.abs(df["c0"].values - c0_median))
        fluctuation_baseline = 1.4826 * mad if mad > 0 else 1e-10

    rows_out = []
    for _, row in df.iterrows():
        raw_xy = None
        if data_root is not None:
            raw_xy = _load_raw_xy_for_row(row, data_root_path, groups_lookup)
        rows_out.extend(
            clean_peaks_one_row(
                row,
                triplets,
                id_columns,
                grouping_factor=grouping_factor,
                fluctuation_baseline=fluctuation_baseline,
                amp_factor1=amp_factor1,
                amp_factor2=amp_factor2,
                amp_include_baseline=amp_include_baseline,
                use_min_threshold=use_min_threshold,
                raw_xy=raw_xy,
                prominence_fluctuation_factor=prominence_fluctuation_factor,
                prominence_composite_factor=prominence_composite_factor,
                local_max_fraction=local_max_fraction,
            )
        )
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Peak classification
# ---------------------------------------------------------------------------

# Band boundaries (MHz): [0, 50), [50, 75), [75, 95), [95, 115), [115, 135), [135, 200]
BAND_BINS = [0, 50, 75, 95, 115, 135, 200]
BAND_LABELS = ["band_38", "band_61", "band_80", "band_102", "band_120", "band_144"]
SIGMA_NARROW_THRESHOLD = 0.8  # MHz: sigma < this -> narrow, else broad


def classify_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add classification columns to a cleaned peak DataFrame.

    Adds:
        band : frequency band (band_38, band_61, band_80, band_102, band_120, band_144)
        width_class : "narrow" (sigma < 0.8 MHz) or "broad"
        primary_in_scan : True for the peak with highest amp in each (U2, line_id, mode)
    """
    if df.empty:
        return df
    out = df.copy()

    # 1. Band assignment by mu
    out["band"] = pd.cut(
        out["mu"],
        bins=BAND_BINS,
        labels=BAND_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)

    # Handle mu outside bins (pd.cut returns NaN)
    out.loc[out["band"] == "nan", "band"] = "unknown"

    # 2. Width class
    out["width_class"] = np.where(out["sigma"] < SIGMA_NARROW_THRESHOLD, "narrow", "broad")

    # 3. Primary-in-scan: per (U2, line_id, mode), max amp -> primary
    group_cols = ["U2", "line_id"]
    if "mode" in out.columns:
        group_cols.append("mode")
    out["primary_in_scan"] = False
    idx_max = out.groupby(group_cols)["amp"].idxmax()
    out.loc[idx_max, "primary_in_scan"] = True

    return out


def run_cleaning_and_save(
    data_dir: str | Path | None = None,
    which: Literal["all", "auto", "improved", "averaged"] = "all",
    output_prefix: str = "cleaned",
    grouping_factor: float | None = None,
    amp_factor1: float | None = None,
    amp_factor2: float | None = None,
    amp_include_baseline: bool | None = None,
    use_min_threshold: bool | None = None,
    use_raw_data_fluctuation: bool = True,
    prominence_fluctuation_factor: float | None = None,
    prominence_composite_factor: float | None = None,
    local_max_fraction: float | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load dataset(s), split trapped/lost, clean peaks, and save to CSV.
    Returns structure: result[dataset_name][mode] = cleaned DataFrame.

    use_raw_data_fluctuation : bool
        If True (default), estimate baseline fluctuation from raw scan
        residuals when perform_fitting is available. Else use MAD of c0.
    """
    data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent
    data = load_fitting_results(data_dir=data_dir, which=which)
    if not isinstance(data, dict):
        data = {which: data}

    groups_lookup = None
    if use_raw_data_fluctuation and _HAS_RAW_DATA_LOADER:
        try:
            groups, _, _ = build_fine_scan_groups(data_root=str(data_dir))
            groups_lookup = groups if groups else None
        except Exception:
            groups_lookup = None

    # Use module defaults when not overridden
    k_group = GROUPING_FACTOR if grouping_factor is None else grouping_factor
    k_amp1 = AMP_FACTOR1 if amp_factor1 is None else amp_factor1
    k_amp2 = AMP_FACTOR2 if amp_factor2 is None else amp_factor2
    k_inc_base = AMP_INCLUDE_BASELINE if amp_include_baseline is None else amp_include_baseline
    k_min_thr = USE_MIN_THRESHOLD if use_min_threshold is None else use_min_threshold
    k_prom_fluct = PROMINENCE_FLUCTUATION_FACTOR if prominence_fluctuation_factor is None else prominence_fluctuation_factor
    k_prom_comp = PROMINENCE_COMPOSITE_FACTOR if prominence_composite_factor is None else prominence_composite_factor
    k_local_max = LOCAL_MAX_FRACTION if local_max_fraction is None else local_max_fraction
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for name, df in data.items():
        trapped, lost = separate_trapped_lost(df)
        data_root = str(data_dir) if use_raw_data_fluctuation else None
        gl = groups_lookup if (name == "averaged" and groups_lookup) else None
        result[name] = {
            "trapped": clean_dataset_peaks(
                trapped,
                grouping_factor=k_group,
                amp_factor1=k_amp1,
                amp_factor2=k_amp2,
                amp_include_baseline=k_inc_base,
                use_min_threshold=k_min_thr,
                data_root=data_root,
                groups_lookup=gl,
                prominence_fluctuation_factor=k_prom_fluct,
                prominence_composite_factor=k_prom_comp,
                local_max_fraction=k_local_max,
            ),
            "lost": clean_dataset_peaks(
                lost,
                grouping_factor=k_group,
                amp_factor1=k_amp1,
                amp_factor2=k_amp2,
                amp_include_baseline=k_inc_base,
                use_min_threshold=k_min_thr,
                data_root=data_root,
                groups_lookup=gl,
                prominence_fluctuation_factor=k_prom_fluct,
                prominence_composite_factor=k_prom_comp,
                local_max_fraction=k_local_max,
            ),
        }
        for mode in ("trapped", "lost"):
            out_df = result[name][mode]
            if out_df.empty:
                continue
            out_df = classify_peaks(out_df)
            result[name][mode] = out_df
            out_path = data_dir / f"{output_prefix}_{name}_{mode}.csv"
            out_df.to_csv(out_path, index=False)

    return result


if __name__ == "__main__":
    # ----- Load test -----
    print("Loading fitting results...")
    data = load_fitting_results(which=RUN_WHICH)
    if isinstance(data, dict):
        assert set(data.keys()) <= set(DEFAULT_FILES.keys()), f"Unexpected keys in data: {set(data.keys())}"
    else:
        data = {RUN_WHICH: data}

    # ----- Separate trapped / lost (no crossing) -----
    print("\n--- Separate trapped / lost ---")
    for name, df in data.items():
        trapped, lost = separate_trapped_lost(df)
        n_t, n_l = len(trapped), len(lost)
        assert n_t + n_l == len(df), f"{name}: trapped+lost should equal total"
        assert set(trapped["mode"].unique()) == {"trapped"}
        assert set(lost["mode"].unique()) == {"lost"}
        print(f"  {name}: total={len(df)}, trapped={n_t}, lost={n_l}")

    # ----- Peak cleaning and save -----
    print("\n--- Run cleaning and save ---")
    script_dir = Path(__file__).resolve().parent
    try:
        cleaned = run_cleaning_and_save(which=RUN_WHICH, output_prefix="cleaned")
    except PermissionError as e:
        print(f"  WARNING: Could not save (file may be open): {e}")
        # Run without saving to still verify cleaning logic
        data = load_fitting_results(which=RUN_WHICH)
        if not isinstance(data, dict):
            data = {RUN_WHICH: data}
        groups_lookup = None
        if _HAS_RAW_DATA_LOADER:
            try:
                groups, _, _ = build_fine_scan_groups(data_root=str(script_dir))
                groups_lookup = groups if groups else None
            except Exception:
                pass
        cleaned = {}
        for name, df in data.items():
            trapped, lost = separate_trapped_lost(df)
            gl = groups_lookup if (name == "averaged" and groups_lookup) else None
            cleaned[name] = {
                "trapped": clean_dataset_peaks(
                    trapped,
                    grouping_factor=GROUPING_FACTOR,
                    amp_factor1=AMP_FACTOR1,
                    amp_factor2=AMP_FACTOR2,
                    use_min_threshold=USE_MIN_THRESHOLD,
                    data_root=str(script_dir),
                    groups_lookup=gl,
                    prominence_fluctuation_factor=PROMINENCE_FLUCTUATION_FACTOR,
                    prominence_composite_factor=PROMINENCE_COMPOSITE_FACTOR,
                    local_max_fraction=LOCAL_MAX_FRACTION,
                ),
                "lost": clean_dataset_peaks(
                    lost,
                    grouping_factor=GROUPING_FACTOR,
                    amp_factor1=AMP_FACTOR1,
                    amp_factor2=AMP_FACTOR2,
                    use_min_threshold=USE_MIN_THRESHOLD,
                    data_root=str(script_dir),
                    groups_lookup=gl,
                    prominence_fluctuation_factor=PROMINENCE_FLUCTUATION_FACTOR,
                    prominence_composite_factor=PROMINENCE_COMPOSITE_FACTOR,
                    local_max_fraction=LOCAL_MAX_FRACTION,
                ),
            }
    for name in cleaned:
        for mode in ("trapped", "lost"):
            df = cleaned[name][mode]
            path = script_dir / f"cleaned_{name}_{mode}.csv"
            if path.exists():
                print(f"  {name} {mode}: {len(df)} peaks -> {path.name}")
            else:
                print(f"  {name} {mode}: {len(df)} peaks (save skipped)")
            if not df.empty:
                assert "amp" in df.columns and "mu" in df.columns and "sigma" in df.columns

    # ----- Sanity: no row in both trapped and lost -----
    print("\n--- No crossing check ---")
    for name, df in data.items():
        trapped, lost = separate_trapped_lost(df)
        # Each original row has a single mode, so no row appears in both subsets
        assert len(trapped) + len(lost) == len(df)
        assert trapped["mode"].eq("trapped").all() and lost["mode"].eq("lost").all()
    print("  Trapped and lost are disjoint (each row has exactly one mode).")

    # ----- Spot-check first available cleaned trapped -----
    first_name = next(iter(cleaned))
    print(f"\n--- Spot-check cleaned_{first_name}_trapped ---")
    c_t = cleaned[first_name]["trapped"]
    print(c_t.head().to_string())
    print(f"  Columns: {list(c_t.columns)}")

    # ----- Verify averaged lost line_id=0 when averaged was run -----
    if "averaged" in cleaned:
        print("\n--- Verify averaged lost line_id=0 (U2=-0.2) ---")
        c_avg_lost = cleaned["averaged"]["lost"]
        subset = c_avg_lost[(c_avg_lost["line_id"] == 0) & (c_avg_lost["U2"] == -0.2)]
        n_peaks = len(subset)
        print(f"  Peaks for this scan: {n_peaks} (expected 1 after grouping)")
        if n_peaks > 1:
            print(subset[["line_id", "mu", "sigma", "amp"]].to_string())
        assert n_peaks == 1, f"Expected 1 peak for averaged lost line_id=0, got {n_peaks}"

    print("\nAll tests passed.")
