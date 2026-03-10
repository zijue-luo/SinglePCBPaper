"""
Simulate nfev filtering: exclude fits with nfev > threshold, then re-apply selection.

R2 layer: within (group, rf, mode, segment, n_peaks), fit_n_peaks picks best by max R².
AICC layer: within (group, rf, mode, segment), picks best by min AICC across n_peaks.
Measures: change rate and R²/AICC loss when filtering by nfev threshold.
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "report_fit_history_nfev", "fit_attempt_history.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "report_nfev_filtering")
os.makedirs(OUT_DIR, exist_ok=True)

# Thresholds: user's list + extra points for smoother curve (log scale)
THRESHOLDS = [
    400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1400, 1500, 1800, 2000,
    2200, 2500, 3000, 3500, 4000, 5000, 6000, 8000, 10000
]


def load_data():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    for r in rows:
        r["ok"] = r.get("ok") == "1" or r.get("ok") == 1
        nfev = r.get("nfev", "").strip()
        r["nfev"] = int(nfev) if nfev and nfev.isdigit() else None
        r2 = r.get("r2", "").strip()
        r["r2"] = float(r2) if r2 and r2 != "-inf" and r2 != "inf" else None
        aicc = r.get("aicc", "").strip()
        r["aicc"] = float(aicc) if aicc and aicc != "inf" and aicc != "-inf" else None
        r["is_stage_best"] = int(r.get("is_stage_best", 0) or 0)
        r["is_segment_selected"] = int(r.get("is_segment_selected", 0) or 0)
        r["group_idx"] = int(r.get("group_idx", 0))
        r["n_peaks_target"] = int(r.get("n_peaks_target", 0))

    return rows


def run_analysis(rows):
    # Stage = (group_idx, rf_amplitude, mode, segment, n_peaks_target)
    # Segment = (group_idx, rf_amplitude, mode, segment)

    ok_rows = [r for r in rows if r["ok"] and r["nfev"] is not None and r["r2"] is not None]

    def stage_key(r):
        return (r["group_idx"], r["rf_amplitude"], r["mode"], r["segment"], r["n_peaks_target"])

    def segment_key(r):
        return (r["group_idx"], r["rf_amplitude"], r["mode"], r["segment"])

    # Index by stage
    stages = defaultdict(list)
    for r in ok_rows:
        stages[stage_key(r)].append(r)

    # Index segments: each segment has multiple n_peaks
    segments = defaultdict(set)
    for r in ok_rows:
        segments[segment_key(r)].add(r["n_peaks_target"])

    results = []
    for thresh in THRESHOLDS:
        # ----- Stage level (R2 selection) -----
        r2_changed = 0
        r2_total = 0
        r2_losses = []

        for sk, arr in stages.items():
            orig_best = next((a for a in arr if a["is_stage_best"]), None)
            if orig_best is None:
                continue
            r2_total += 1
            filtered = [a for a in arr if a["nfev"] <= thresh]
            if not filtered:
                r2_changed += 1
                r2_losses.append(orig_best["r2"])  # lost everything, "loss" = full R2
                continue
            new_best = max(filtered, key=lambda a: a["r2"])
            if new_best["r2"] < orig_best["r2"] - 1e-10:
                r2_changed += 1
                r2_losses.append(orig_best["r2"] - new_best["r2"])

        r2_change_rate = (r2_changed / r2_total * 100) if r2_total > 0 else 0
        r2_loss_mean = np.mean(r2_losses) * 100 if r2_losses else 0
        r2_loss_median = np.median(r2_losses) * 100 if r2_losses else 0

        # ----- Segment level (AICC selection) -----
        aicc_changed = 0
        aicc_total = 0
        aicc_losses = []

        for segk, n_set in segments.items():
            orig_sel = next(
                (r for r in ok_rows if segment_key(r) == segk and r["is_segment_selected"]),
                None
            )
            if orig_sel is None:
                continue
            aicc_total += 1
            # For each n in segment, get stage_best (among nfev<=thresh)
            stage_bests = {}
            for n in n_set:
                sk = (*segk, n)
                arr = stages.get(sk, [])
                filtered = [a for a in arr if a["nfev"] <= thresh]
                if not filtered:
                    continue
                stage_bests[n] = max(filtered, key=lambda a: a["r2"])

            if not stage_bests:
                aicc_changed += 1
                aicc_losses.append(np.inf)  # no valid model
                continue

            new_sel = min(stage_bests.values(), key=lambda a: a["aicc"])
            orig_aicc = orig_sel["aicc"]
            new_aicc = new_sel["aicc"]
            if new_aicc > orig_aicc + 1e-6:
                aicc_changed += 1
                aicc_losses.append(new_aicc - orig_aicc)

        aicc_change_rate = (aicc_changed / aicc_total * 100) if aicc_total > 0 else 0
        finite_aicc_losses = [x for x in aicc_losses if np.isfinite(x)]
        aicc_loss_mean = np.mean(finite_aicc_losses) if finite_aicc_losses else 0
        aicc_loss_median = np.median(finite_aicc_losses) if finite_aicc_losses else 0

        results.append({
            "threshold": thresh,
            "r2_change_pct": r2_change_rate,
            "r2_loss_mean": r2_loss_mean,
            "r2_loss_median": r2_loss_median,
            "r2_changed": r2_changed,
            "r2_total": r2_total,
            "aicc_change_pct": aicc_change_rate,
            "aicc_loss_mean": aicc_loss_mean,
            "aicc_loss_median": aicc_loss_median,
            "aicc_changed": aicc_changed,
            "aicc_total": aicc_total,
        })

    return results


def plot_results(results, out_dir):
    thresh = np.array([r["threshold"] for r in results])
    r2_pct = np.array([r["r2_change_pct"] for r in results])
    aicc_pct = np.array([r["aicc_change_pct"] for r in results])
    r2_loss = np.array([r["r2_loss_mean"] for r in results])
    aicc_loss = np.array([r["aicc_loss_mean"] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Top left: R2 change rate
    ax = axes[0, 0]
    ax.semilogx(thresh, r2_pct, "o-", color="C0", markersize=4)
    ax.set_xlabel("nfev filtering threshold")
    ax.set_ylabel("R2 selection change rate (%)")
    ax.set_title("Stage level: proportion of stages with different best after nfev filter")
    ax.grid(True, alpha=0.3)

    # Top right: AICC change rate
    ax = axes[0, 1]
    ax.semilogx(thresh, aicc_pct, "s-", color="C1", markersize=4)
    ax.set_xlabel("nfev filtering threshold")
    ax.set_ylabel("AICC selection change rate (%)")
    ax.set_title("Segment level: proportion of segments with different selection after nfev filter")
    ax.grid(True, alpha=0.3)

    # Bottom left: R2 loss when changed (scaled ×100 for readability)
    ax = axes[1, 0]
    ax.semilogx(thresh, r2_loss, "o-", color="C0", markersize=4)
    ax.set_xlabel("nfev filtering threshold")
    ax.set_ylabel("R2 loss when changed (×0.01)")
    ax.set_title("Stage level: mean R2 drop (orig - new) when selection changes")
    ax.grid(True, alpha=0.3)

    # Bottom right: AICC loss when changed
    ax = axes[1, 1]
    ax.semilogx(thresh, aicc_loss, "s-", color="C1", markersize=4)
    ax.set_xlabel("nfev filtering threshold")
    ax.set_ylabel("AICC loss (when changed)")
    ax.set_title("Segment level: mean AICC increase when selection changes")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nfev_filtering_impact.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Combined: both change rates on one plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(thresh, r2_pct, "o-", label="R2 (stage)", color="C0", markersize=5)
    ax.semilogx(thresh, aicc_pct, "s-", label="AICC (segment)", color="C1", markersize=5)
    ax.set_xlabel("nfev filtering threshold")
    ax.set_ylabel("Selection change rate (%)")
    ax.set_title("Impact of nfev filtering on R2 and AICC selection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nfev_filtering_change_rates.png"), dpi=150, bbox_inches="tight")
    plt.close()


def write_summary(results, out_dir):
    path = os.path.join(out_dir, "nfev_filtering_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "threshold", "r2_change_pct", "r2_loss_mean", "r2_loss_median",
            "r2_changed", "r2_total", "aicc_change_pct", "aicc_loss_mean",
            "aicc_loss_median", "aicc_changed", "aicc_total"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {path}")


def main():
    rows = load_data()
    results = run_analysis(rows)
    plot_results(results, OUT_DIR)
    write_summary(results, OUT_DIR)

    print("\nSample (threshold=3500):")
    r = next(x for x in results if x["threshold"] == 3500)
    print(f"  R2:   change rate {r['r2_change_pct']:.2f}%, mean loss {r['r2_loss_mean']:.4f} (x0.01)")
    print(f"  AICC: change rate {r['aicc_change_pct']:.2f}%, mean loss {r['aicc_loss_mean']:.2f}")


if __name__ == "__main__":
    main()
