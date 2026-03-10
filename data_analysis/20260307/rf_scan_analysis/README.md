# RF Scan / Tickling Data Analysis

Ion trap tickling experiment RF scan analysis: Gaussian peak fitting for ratio_signal (trapped) and ratio_lost; outputs peak center (mu), amplitude, sigma, etc.

## Data Layout

- Root: `data_rf/` (relative to script parent)
- Structure: `data_rf/<YYYYMMDD>/<timestamp>_<column>`
- Key files: `arr_of_setpoints` (frequency x), `ratio_signal`, `ratio_lost`, `_conf`, `act_RF_amplitude`
- Metadata: `tickling_experiment_run_job_list.json` defines RF groups and run_tag

## Main Pipeline

```bash
python perform_fitting_tickling.py
```

- **Phase 1**: Averaged fit per RF group → `run_best_models_averaged_*.csv` and `final_analysis_plots/averaged_RF*.png`
- **Phase 2**: Per-scan fit, incremental write to `run_best_models_*.csv`, single-scan plots

## Module Structure

| Module | Purpose |
|--------|---------|
| `config.py` | Analysis params, CSV headers, RF unit conversion |
| `data_io.py` | Data loading, conf parsing, act_RF trimmed mean |
| `metadata.py` | Build groups from JSON, timestamp list, date exclusion |
| `analysis.py` | Left/right segment fit, merge, incremental peak adding |
| `csv_export.py` | Fit result to CSV (averaged and per-scan formats) |
| `plotting.py` | Plotting (stitched left+right curves, not merged model) |
| `fitting_functions.py` | Low-level Gaussian fit, curve_fit, multiprocessing |
| `split_experiment.py` | Baseline estimate, split, find_peaks initial guess |

## Configuration

- `config.EXCLUDE_DATES`: Date prefixes (YYYYMMDD) to skip in averaging and fitting
- `config.R2_THRESHOLD`: Stop adding peaks when R² exceeds this
- `config.USE_RF_SOURCE`: `setpoint` or `act` for CSV RF value

## research/

Experimental scripts, A/B tests, algorithm studies. See `research/README.md`.
