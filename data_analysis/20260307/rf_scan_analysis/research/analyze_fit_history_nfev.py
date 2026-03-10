"""
Analyze full fitting history (nfev/r2/aicc/init-mu) across experiments.

Outputs:
- CSV with per-fit attempt records
- CSV with per-stage scan-density simulation
- PNG plots
- Markdown report with recommendations
"""
import csv
import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_SCAN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, RF_SCAN_DIR)

from perform_fitting_tickling import build_tickling_groups, load_and_average_group, analyze_fine_scan_from_arrays


SCAN_COUNT = 120
MAX_N_PEAKS = 9
R2_THRESHOLD = 0.995
MAX_NFEV = 20000
MAX_GROUPS = None  # None -> all groups
OUT_DIR = os.path.join(SCRIPT_DIR, "report_fit_history_nfev")


def _safe_mean(values):
    arr = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(arr)) if arr else None


def _quantiles(values, qs=(0.5, 0.9, 0.95, 0.99)):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {q: None for q in qs}
    return {q: float(np.quantile(arr, q)) for q in qs}


def _flatten_mode_history(group_idx, rf_amp, mode, mode_hist):
    rows = []
    stage_rows = []
    for seg in mode_hist:
        seg_name = seg.get("segment")
        seg_best = seg.get("best")
        seg_selected_n = int(seg_best["n_peaks"]) if seg_best and seg_best.get("ok") else None
        seg_selected_mu0 = tuple(seg_best.get("mu0", [])) if seg_best and seg_best.get("ok") else None
        x_min, x_max = seg.get("x_min"), seg.get("x_max")
        x_span = (x_max - x_min) if (x_min is not None and x_max is not None) else None

        for stage in seg.get("incremental", []):
            n_target = int(stage.get("n_peaks"))
            hist = stage.get("history") or []
            stage_best = stage.get("stage_best")
            stage_best_mu0 = tuple(stage_best.get("mu0", [])) if stage_best and stage_best.get("ok") else None
            stage_best_r2 = float(stage_best["r2"]) if stage_best and stage_best.get("ok") else None

            ok_entries = [e for e in hist if e and e.get("ok")]
            n_total = len(hist)
            n_ok = len(ok_entries)
            n_hit_max = sum(
                1 for e in ok_entries
                if (e.get("nfev") is not None and int(e.get("nfev")) >= MAX_NFEV)
            )

            # Coarse-grid simulation to estimate scan density impact.
            by_stride = {}
            for stride in (2, 3):
                sub = [e for i, e in enumerate(hist) if i % stride == 0 and e is not None and e.get("ok")]
                if sub and stage_best_r2 is not None:
                    coarse_best = max(sub, key=lambda e: e.get("r2", -np.inf))
                    dr2 = float(stage_best_r2 - coarse_best.get("r2", -np.inf))
                    by_stride[stride] = dr2
                else:
                    by_stride[stride] = None

            stage_rows.append({
                "group_idx": group_idx,
                "rf_amplitude": rf_amp,
                "mode": mode,
                "segment": seg_name,
                "n_peaks_target": n_target,
                "scan_count": int(stage.get("scan_count", 0)),
                "n_candidates": n_total,
                "n_ok": n_ok,
                "n_hit_max": n_hit_max,
                "hit_max_rate_ok": (float(n_hit_max) / n_ok) if n_ok > 0 else None,
                "stage_best_r2": stage_best_r2,
                "coarse_stride2_dr2": by_stride[2],
                "coarse_stride3_dr2": by_stride[3],
            })

            for idx, entry in enumerate(hist):
                if entry is None:
                    continue
                mu0 = entry.get("mu0", [])
                init_new_mu = None
                if len(mu0) >= n_target:
                    init_new_mu = float(mu0[-1])
                init_mu_norm = None
                if init_new_mu is not None and x_span is not None and x_span > 0:
                    init_mu_norm = float((init_new_mu - x_min) / x_span)

                ok = bool(entry.get("ok"))
                nfev = entry.get("nfev")
                if nfev is not None:
                    nfev = int(nfev)
                is_stage_best = bool(ok and stage_best_mu0 is not None and tuple(mu0) == stage_best_mu0)
                is_segment_selected = bool(
                    ok and seg_selected_n is not None and n_target == seg_selected_n and
                    seg_selected_mu0 is not None and tuple(mu0) == seg_selected_mu0
                )
                rows.append({
                    "group_idx": group_idx,
                    "rf_amplitude": rf_amp,
                    "mode": mode,
                    "segment": seg_name,
                    "n_peaks_target": n_target,
                    "scan_count": int(stage.get("scan_count", 0)),
                    "candidate_idx": int(idx),
                    "ok": int(ok),
                    "r2": float(entry.get("r2")) if entry.get("r2") is not None else None,
                    "aicc": float(entry.get("aicc")) if entry.get("aicc") is not None else None,
                    "nfev": nfev,
                    "hit_max_nfev": int(nfev is not None and nfev >= MAX_NFEV),
                    "ier": entry.get("ier"),
                    "is_stage_best": int(is_stage_best),
                    "is_segment_selected": int(is_segment_selected),
                    "init_new_mu": init_new_mu,
                    "init_mu_norm": init_mu_norm,
                })

    return rows, stage_rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_hist_nfev(ok_rows, out_path):
    vals = [r["nfev"] for r in ok_rows if r.get("nfev") is not None]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, max(vals), 60)
    ax.hist(vals, bins=bins, alpha=0.8)
    ax.set_title("nfev Distribution (Successful Fits)")
    ax.set_xlabel("nfev")
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_hitmax_by_npeaks(rows, out_path):
    by_n = {}
    for r in rows:
        if not r["ok"]:
            continue
        n = r["n_peaks_target"]
        by_n.setdefault(n, []).append(r["hit_max_nfev"])
    if not by_n:
        return
    xs = sorted(by_n.keys())
    ys = [float(np.mean(by_n[n])) for n in xs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xs, ys)
    ax.set_ylim(0, 1)
    ax.set_title("Hit-max-nfev Rate vs n_peaks_target")
    ax.set_xlabel("n_peaks_target")
    ax.set_ylabel("hit-max rate (successful fits)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_nfev_box_by_npeaks(rows, out_path):
    by_n = {}
    for r in rows:
        if not r["ok"] or r["nfev"] is None:
            continue
        by_n.setdefault(r["n_peaks_target"], []).append(r["nfev"])
    if not by_n:
        return
    xs = sorted(by_n.keys())
    data = [by_n[n] for n in xs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, labels=xs, showfliers=False)
    ax.set_title("nfev by n_peaks_target (Successful Fits)")
    ax.set_xlabel("n_peaks_target")
    ax.set_ylabel("nfev")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_init_mu_vs_r2(rows, out_path):
    x = []
    y = []
    c = []
    for r in rows:
        if not r["ok"]:
            continue
        if r["init_mu_norm"] is None or r["r2"] is None or r["nfev"] is None:
            continue
        x.append(r["init_mu_norm"])
        y.append(r["r2"])
        c.append(r["nfev"])
    if not x:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(x, y, c=c, s=8, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="nfev")
    ax.set_title("Init Mu Position vs Final R2")
    ax.set_xlabel("init mu normalized in segment [0,1]")
    ax.set_ylabel("R2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_stride_dr2(stage_rows, out_path):
    vals2 = [r["coarse_stride2_dr2"] for r in stage_rows if r["coarse_stride2_dr2"] is not None]
    vals3 = [r["coarse_stride3_dr2"] for r in stage_rows if r["coarse_stride3_dr2"] is not None]
    if not vals2 and not vals3:
        return
    data = []
    labels = []
    if vals2:
        data.append(vals2)
        labels.append("stride=2")
    if vals3:
        data.append(vals3)
        labels.append("stride=3")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("R2 Degradation from Coarser Scan Grid")
    ax.set_ylabel("ΔR2 (full best - coarse best)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _recommended_max_nfev(rows):
    selected = [r["nfev"] for r in rows if r["ok"] and r["is_segment_selected"] and r["nfev"] is not None]
    if not selected:
        return None, {}
    q = _quantiles(selected, qs=(0.9, 0.95, 0.99))
    # Round up to nearest 500 for a practical tuning value.
    q99 = q[0.99]
    rec = int(math.ceil(q99 / 500.0) * 500.0) if q99 is not None else None
    return rec, q


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    groups, _, _ = build_tickling_groups()
    if not groups:
        print("No groups found.")
        return
    if MAX_GROUPS is not None:
        groups = groups[: int(MAX_GROUPS)]

    print(f"[Run] groups={len(groups)} scan_count={SCAN_COUNT} max_n_peaks={MAX_N_PEAKS} r2={R2_THRESHOLD} max_nfev={MAX_NFEV}")
    t0 = time.perf_counter()
    all_rows = []
    all_stage_rows = []

    for gi, g in enumerate(groups):
        rf = g.get("RF_amplitude")
        ts_list = g.get("timestamps", [])
        print(f"  - group {gi+1}/{len(groups)} RF={rf} dBm, reps={len(ts_list)}")
        x, ys = load_and_average_group(ts_list)
        y = {"lost": ys["ratio_lost"], "trapped": ys["ratio_signal"]}
        result = analyze_fine_scan_from_arrays(
            x, y,
            scan_count=SCAN_COUNT,
            max_n_peaks=MAX_N_PEAKS,
            r2_gate=R2_THRESHOLD,
            max_nfev=MAX_NFEV,
            return_fit_history=True,
        )
        for mode in ("lost", "trapped"):
            rows, stage_rows = _flatten_mode_history(gi, rf, mode, result[mode].get("history", []))
            all_rows.extend(rows)
            all_stage_rows.extend(stage_rows)

    elapsed = time.perf_counter() - t0
    print(f"[Done] elapsed={elapsed:.1f}s attempts={len(all_rows)} stages={len(all_stage_rows)}")

    raw_csv = os.path.join(OUT_DIR, "fit_attempt_history.csv")
    stage_csv = os.path.join(OUT_DIR, "scan_stage_summary.csv")
    _write_csv(raw_csv, all_rows)
    _write_csv(stage_csv, all_stage_rows)

    ok_rows = [r for r in all_rows if r["ok"]]
    _plot_hist_nfev(ok_rows, os.path.join(OUT_DIR, "nfev_hist.png"))
    _plot_hitmax_by_npeaks(all_rows, os.path.join(OUT_DIR, "hitmax_rate_by_npeaks.png"))
    _plot_nfev_box_by_npeaks(all_rows, os.path.join(OUT_DIR, "nfev_box_by_npeaks.png"))
    _plot_init_mu_vs_r2(all_rows, os.path.join(OUT_DIR, "init_mu_norm_vs_r2.png"))
    _plot_stride_dr2(all_stage_rows, os.path.join(OUT_DIR, "coarse_scan_dr2.png"))

    selected_rows = [r for r in ok_rows if r["is_segment_selected"]]
    rec_max_nfev, q_sel = _recommended_max_nfev(all_rows)
    q_all = _quantiles([r["nfev"] for r in ok_rows if r["nfev"] is not None])
    q_stage_best = _quantiles([r["nfev"] for r in ok_rows if r["is_stage_best"] and r["nfev"] is not None])
    q_selected = _quantiles([r["nfev"] for r in selected_rows if r["nfev"] is not None])

    stride2 = [r["coarse_stride2_dr2"] for r in all_stage_rows if r["coarse_stride2_dr2"] is not None]
    stride3 = [r["coarse_stride3_dr2"] for r in all_stage_rows if r["coarse_stride3_dr2"] is not None]
    stride2_good = float(np.mean([v <= 1e-4 for v in stride2])) if stride2 else None
    stride3_good = float(np.mean([v <= 1e-4 for v in stride3])) if stride3 else None

    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Fit History nfev Analysis\n\n")
        f.write("## Setup\n")
        f.write(f"- groups analyzed: {len(groups)}\n")
        f.write(f"- scan_count: {SCAN_COUNT}\n")
        f.write(f"- max_n_peaks: {MAX_N_PEAKS}\n")
        f.write(f"- r2_threshold: {R2_THRESHOLD}\n")
        f.write(f"- max_nfev: {MAX_NFEV}\n")
        f.write(f"- elapsed_s: {elapsed:.1f}\n\n")

        f.write("## Attempt Counts\n")
        f.write(f"- total fit attempts: {len(all_rows)}\n")
        f.write(f"- successful attempts: {len(ok_rows)}\n")
        f.write(f"- success rate: {len(ok_rows)/max(len(all_rows),1):.3f}\n")
        f.write(f"- segment-selected attempts: {len(selected_rows)}\n\n")

        f.write("## nfev Quantiles (successful)\n")
        f.write(f"- all attempts: {q_all}\n")
        f.write(f"- stage-best attempts: {q_stage_best}\n")
        f.write(f"- segment-selected attempts: {q_selected}\n\n")

        hit_max_ok = [r["hit_max_nfev"] for r in ok_rows]
        hit_max_sel = [r["hit_max_nfev"] for r in selected_rows]
        f.write("## Hit-max-nfev Rate\n")
        f.write(f"- successful attempts: {_safe_mean(hit_max_ok)}\n")
        f.write(f"- segment-selected attempts: {_safe_mean(hit_max_sel)}\n\n")

        f.write("## Scan Density What-if (coarse grid)\n")
        f.write("- metric: ΔR2 = full-grid best R2 - coarse-grid best R2\n")
        f.write(f"- stride=2: median={np.median(stride2) if stride2 else None}, p95={np.quantile(stride2,0.95) if stride2 else None}, share(ΔR2<=1e-4)={stride2_good}\n")
        f.write(f"- stride=3: median={np.median(stride3) if stride3 else None}, p95={np.quantile(stride3,0.95) if stride3 else None}, share(ΔR2<=1e-4)={stride3_good}\n\n")

        f.write("## Recommendation\n")
        f.write(f"- recommended max_nfev (from selected-fit q99): {rec_max_nfev}\n")
        f.write(f"- selected-fit quantiles used: {q_sel}\n")
        if rec_max_nfev is not None and rec_max_nfev < MAX_NFEV:
            f.write("- suggested action: try this lower max_nfev in A/B test and verify final selected model consistency.\n")
        else:
            f.write("- suggested action: keep current max_nfev or run on more groups before lowering.\n")

    print(f"[Saved] {raw_csv}")
    print(f"[Saved] {stage_csv}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()

