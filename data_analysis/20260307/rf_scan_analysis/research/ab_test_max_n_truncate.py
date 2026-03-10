"""
Analyze impact of truncating max_n_peaks: if we force max_n_peaks to a lower value,
what R2 and AICC loss do we incur at the final selection layer (segment/AICC layer)?

Only consider segment-level selection; no per-n (R2 layer) breakdown.
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "report_fit_history_nfev", "fit_attempt_history.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "report_max_n_truncate")
os.makedirs(OUT_DIR, exist_ok=True)

# Truncation levels to test (original max is 9)
MAX_N_VALUES = [3, 4, 5, 6, 7, 8, 9]


def load_data():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    for r in rows:
        r["ok"] = r.get("ok") == "1" or r.get("ok") == 1
        r2 = r.get("r2", "").strip()
        r["r2"] = float(r2) if r2 and r2 != "-inf" and r2 != "inf" else None
        aicc = r.get("aicc", "").strip()
        r["aicc"] = float(aicc) if aicc and aicc != "inf" and aicc != "-inf" else None
        r["is_stage_best"] = int(r.get("is_stage_best", 0) or 0)
        r["is_segment_selected"] = int(r.get("is_segment_selected", 0) or 0)
        r["n_peaks_target"] = int(r.get("n_peaks_target", 0))

    return rows


def run_analysis(rows):
    ok_rows = [r for r in rows if r["ok"] and r["r2"] is not None and r["aicc"] is not None]

    def stage_key(r):
        return (r["group_idx"], r["rf_amplitude"], r["mode"], r["segment"], r["n_peaks_target"])

    def segment_key(r):
        return (r["group_idx"], r["rf_amplitude"], r["mode"], r["segment"])

    stages = defaultdict(list)
    for r in ok_rows:
        stages[stage_key(r)].append(r)

    segments = defaultdict(set)
    for r in ok_rows:
        segments[segment_key(r)].add(r["n_peaks_target"])

    results = []
    for max_n in MAX_N_VALUES:
        r2_losses = []
        aicc_losses = []
        n_changed = 0
        n_total = 0

        for segk, n_set in segments.items():
            orig_sel = next(
                (r for r in ok_rows if segment_key(r) == segk and r["is_segment_selected"]),
                None
            )
            if orig_sel is None:
                continue
            n_total += 1
            n_orig = orig_sel["n_peaks_target"]

            # Stage bests for n <= max_n
            stage_bests = {}
            for n in n_set:
                if n > max_n:
                    continue
                sk = (*segk, n)
                arr = stages.get(sk, [])
                if not arr:
                    continue
                best = max(arr, key=lambda a: a["r2"])
                stage_bests[n] = best

            if not stage_bests:
                n_changed += 1
                r2_losses.append(orig_sel["r2"])
                aicc_losses.append(np.inf)
                continue

            new_sel = min(stage_bests.values(), key=lambda a: a["aicc"])

            if n_orig > max_n:
                # Forced to pick different (n <= max_n)
                n_changed += 1
                r2_losses.append(orig_sel["r2"] - new_sel["r2"])
                aicc_losses.append(new_sel["aicc"] - orig_sel["aicc"])

        r2_mean = np.mean(r2_losses) * 100 if r2_losses else 0
        r2_median = np.median(r2_losses) * 100 if r2_losses else 0
        aicc_finite = [x for x in aicc_losses if np.isfinite(x)]
        aicc_mean = np.mean(aicc_finite) if aicc_finite else 0
        aicc_median = np.median(aicc_finite) if aicc_finite else 0
        change_pct = (n_changed / n_total * 100) if n_total > 0 else 0

        results.append({
            "max_n": max_n,
            "n_changed": n_changed,
            "n_total": n_total,
            "change_pct": change_pct,
            "r2_loss_mean": r2_mean,
            "r2_loss_median": r2_median,
            "aicc_loss_mean": aicc_mean,
            "aicc_loss_median": aicc_median,
            "r2_losses": r2_losses,
            "aicc_losses": aicc_losses,
        })

    return results


def plot_results(results, out_dir):
    max_n = np.array([r["max_n"] for r in results])
    change_pct = np.array([r["change_pct"] for r in results])
    r2_loss = np.array([r["r2_loss_mean"] for r in results])
    aicc_loss = np.array([r["aicc_loss_mean"] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ax = axes[0]
    ax.bar(max_n, change_pct, color="C0")
    ax.set_xlabel("max_n_peaks (truncation)")
    ax.set_ylabel("Segment selection change rate (%)")
    ax.set_title("Proportion of segments with different selection")
    ax.set_xticks(max_n)

    ax = axes[1]
    ax.bar(max_n, r2_loss, color="C1")
    ax.set_xlabel("max_n_peaks (truncation)")
    ax.set_ylabel("R2 loss when changed (x0.01)")
    ax.set_title("R2 loss at final selection layer")
    ax.set_xticks(max_n)

    ax = axes[2]
    ax.bar(max_n, aicc_loss, color="C2")
    ax.set_xlabel("max_n_peaks (truncation)")
    ax.set_ylabel("AICC loss when changed")
    ax.set_title("AICC loss at final selection layer")
    ax.set_xticks(max_n)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "max_n_truncate_impact.png"), dpi=150, bbox_inches="tight")
    plt.close()


def write_summary(results, out_dir):
    path = os.path.join(out_dir, "max_n_truncate_summary.csv")
    rows = []
    for r in results:
        rows.append({
            "max_n": r["max_n"],
            "n_changed": r["n_changed"],
            "n_total": r["n_total"],
            "change_pct": r["change_pct"],
            "r2_loss_mean": r["r2_loss_mean"],
            "r2_loss_median": r["r2_loss_median"],
            "aicc_loss_mean": r["aicc_loss_mean"],
            "aicc_loss_median": r["aicc_loss_median"],
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main():
    rows = load_data()
    results = run_analysis(rows)
    plot_results(results, OUT_DIR)
    write_summary(results, OUT_DIR)

    print("\n=== max_n_peaks truncation impact (final selection layer) ===\n")
    for r in results:
        print(f"max_n={r['max_n']}: {r['n_changed']}/{r['n_total']} segments changed ({r['change_pct']:.1f}%)")
        if r["r2_losses"]:
            print(f"  R2 loss:   mean {r['r2_loss_mean']:.4f} (x0.01), median {r['r2_loss_median']:.4f}")
        if r["aicc_losses"]:
            finite = [x for x in r["aicc_losses"] if np.isfinite(x)]
            if finite:
                print(f"  AICC loss: mean {np.mean(finite):.2f}, median {np.median(finite):.2f}")
        print()


if __name__ == "__main__":
    main()
