"""
Simulate stride filtering: keep only every stride-th candidate (candidate_idx % stride == 0).
This corresponds to reducing scan_count (e.g. stride=2 => scan_count halved, stepsize doubled).

Filter out: 1/2 (stride=2), 2/3 (stride=3), 3/4 (stride=4), 4/5 (stride=5), 5/6 (stride=6).

Measure: R2 layer and AICC layer change rate and loss when selection differs from original.
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "report_fit_history_nfev", "fit_attempt_history.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "report_stride_filtering")
os.makedirs(OUT_DIR, exist_ok=True)

# stride=2 => keep 1/2, stride=3 => keep 1/3, etc.
STRIDES = [2, 3, 4, 5, 6]


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
        r["candidate_idx"] = int(r.get("candidate_idx", 0))

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
    for stride in STRIDES:
        # Filter: keep candidate_idx % stride == 0
        def filtered(arr):
            return [a for a in arr if a["candidate_idx"] % stride == 0]

        # ----- R2 layer -----
        r2_changed = 0
        r2_total = 0
        r2_losses = []

        for sk, arr in stages.items():
            orig_best = next((a for a in arr if a["is_stage_best"]), None)
            if orig_best is None:
                continue
            r2_total += 1
            f = filtered(arr)
            if not f:
                r2_changed += 1
                r2_losses.append(orig_best["r2"])
                continue
            new_best = max(f, key=lambda a: a["r2"])
            if new_best["r2"] < orig_best["r2"] - 1e-10:
                r2_changed += 1
                r2_losses.append(orig_best["r2"] - new_best["r2"])

        r2_change_rate = (r2_changed / r2_total * 100) if r2_total > 0 else 0
        r2_loss_mean = np.mean(r2_losses) * 100 if r2_losses else 0
        r2_loss_median = np.median(r2_losses) * 100 if r2_losses else 0

        # ----- AICC layer -----
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
            stage_bests = {}
            for n in n_set:
                sk = (*segk, n)
                arr = stages.get(sk, [])
                f = filtered(arr)
                if not f:
                    continue
                stage_bests[n] = max(f, key=lambda a: a["r2"])

            if not stage_bests:
                aicc_changed += 1
                aicc_losses.append(np.inf)
                continue

            new_sel = min(stage_bests.values(), key=lambda a: a["aicc"])
            if new_sel["aicc"] > orig_sel["aicc"] + 1e-6:
                aicc_changed += 1
                aicc_losses.append(new_sel["aicc"] - orig_sel["aicc"])

        aicc_change_rate = (aicc_changed / aicc_total * 100) if aicc_total > 0 else 0
        finite = [x for x in aicc_losses if np.isfinite(x)]
        aicc_loss_mean = np.mean(finite) if finite else 0
        aicc_loss_median = np.median(finite) if finite else 0

        results.append({
            "stride": stride,
            "keep_frac": 1.0 / stride,
            "filter_out_frac": 1.0 - 1.0 / stride,
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
    stride = np.array([r["stride"] for r in results])
    r2_pct = np.array([r["r2_change_pct"] for r in results])
    aicc_pct = np.array([r["aicc_change_pct"] for r in results])
    r2_loss = np.array([r["r2_loss_mean"] for r in results])
    aicc_loss = np.array([r["aicc_loss_mean"] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    ax.bar(stride - 0.2, r2_pct, 0.4, label="R2 layer", color="C0")
    ax.bar(stride + 0.2, aicc_pct, 0.4, label="AICC layer", color="C1")
    ax.set_xlabel("stride (keep 1/stride of candidates)")
    ax.set_ylabel("Selection change rate (%)")
    ax.set_title("R2 / AICC layer: selection change rate")
    ax.set_xticks(stride)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.bar(stride - 0.2, r2_loss, 0.4, label="R2 loss (x0.01)", color="C0")
    ax.set_xlabel("stride")
    ax.set_ylabel("R2 loss when changed (x0.01)")
    ax.set_title("R2 layer: mean R2 drop when selection changes")
    ax.set_xticks(stride)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.bar(stride, aicc_loss, 0.5, color="C1")
    ax.set_xlabel("stride")
    ax.set_ylabel("AICC loss when changed")
    ax.set_title("AICC layer: mean AICC increase when selection changes")
    ax.set_xticks(stride)
    ax.grid(True, alpha=0.3, axis="y")

    # Summary text
    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "stride=2: keep 1/2 (filter out 1/2)",
        "stride=3: keep 1/3 (filter out 2/3)",
        "stride=4: keep 1/4 (filter out 3/4)",
        "stride=5: keep 1/5 (filter out 4/5)",
        "stride=6: keep 1/6 (filter out 5/6)",
        "",
        "Simulates scan_count reduced, step size increased",
    ]
    ax.text(0.1, 0.9, "\n".join(lines), transform=ax.transAxes, fontsize=11, verticalalignment="top")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stride_filtering_impact.png"), dpi=150, bbox_inches="tight")
    plt.close()


def write_summary(results, out_dir):
    path = os.path.join(out_dir, "stride_filtering_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "stride", "keep_frac", "filter_out_frac",
            "r2_change_pct", "r2_loss_mean", "r2_loss_median", "r2_changed", "r2_total",
            "aicc_change_pct", "aicc_loss_mean", "aicc_loss_median", "aicc_changed", "aicc_total",
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {path}")


def main():
    rows = load_data()
    results = run_analysis(rows)
    plot_results(results, OUT_DIR)
    write_summary(results, OUT_DIR)

    print("\n=== Stride filtering impact ===\n")
    for r in results:
        print(f"stride={r['stride']} (filter掉{r['filter_out_frac']:.0%}):")
        print(f"  R2层:   变化率 {r['r2_change_pct']:.1f}%, 变化时平均R2损失 {r['r2_loss_mean']:.4f} (×0.01)")
        print(f"  AICC层: 变化率 {r['aicc_change_pct']:.1f}%, 变化时平均AICC损失 {r['aicc_loss_mean']:.2f}")
        print()


if __name__ == "__main__":
    main()
