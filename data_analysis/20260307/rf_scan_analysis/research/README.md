# Algorithm Research

Scripts for algorithm development, method comparison, and validation.
Separate from production pipeline (`perform_fitting_tickling.py`, `fitting_functions.py`, `split_experiment.py`).

## Contents

| Script | Purpose |
|--------|---------|
| `baseline_estimation.py` | Baseline estimation methods comparison, validation, plotting |
| `analyze_raw_data.py` | Raw data structure, noise, rep variability, peak shapes |
| `analyze_double_peak.py` | Double-peak structure, sigma ratios, scan_count implications |
| `ab_test_max_nfev.py` | A/B test max_nfev 20k vs 10k vs 5k (SCAN=120, MAX_N=9, R2=0.995) |
| `test_timing_3runs.py` | Timing test for 3 timestamps (SCAN=120, MAX_N=9, R2=0.995) |
| `analyze_fit_history_nfev.py` | Full fit-history nfev/init-mu analysis with plots/report |
| `stage2_split_with_repair.py` | Baseline/signal repair, central safe split, all-run plotting |
| `test_find_peaks_from_split.py` | Conservative find_peaks init (no smoothing), plots per experiment |
| `plot_repeats_per_RF.py` | Plot all repeats on one figure per RF value (2 subplots: lost, trapped) |

## Usage

Run from `rf_scan_analysis` or `research` directory:
```
python research/baseline_estimation.py
python research/analyze_raw_data.py
python research/analyze_double_peak.py
python research/ab_test_max_nfev.py
python research/test_timing_3runs.py
python research/analyze_fit_history_nfev.py
python research/stage2_split_with_repair.py
python research/test_find_peaks_from_split.py
python research/plot_repeats_per_RF.py
```
