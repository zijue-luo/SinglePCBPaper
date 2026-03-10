"""
Plot all repeats on one figure per RF value.
For each RF setpoint: 2 subplots (lost, trapped), each showing all repeat curves with lines.
Output: research/plot_repeats_per_RF/repeats_RF*.png
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RF_SCAN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, RF_SCAN_DIR)

from perform_fitting_tickling import (
    build_tickling_groups,
    load_data,
    load_act_RF_trimmed_mean,
    _get_data_root,
    DATA_SUBDIR,
)

OUTPUT_DIR = "plot_repeats_per_RF"


def plot_repeats_for_group(group, out_path, data_root=None):
    """
    Plot all repeats for one RF group on a single figure.
    2 subplots: lost (ratio_lost), trapped (ratio_signal).
    Each repeat: one scatter curve per subplot.
    """
    rf = group["RF_amplitude"]
    timestamps = group["timestamps"]
    if not timestamps:
        return

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax_lost = axs[0]
    ax_trapped = axs[1]

    for rep, ts in enumerate(timestamps):
        try:
            x, ys = load_data(
                ts, ynames=["ratio_signal", "ratio_lost"], data_root=data_root
            )
        except Exception as e:
            print(f"  [WARN] Skip {ts}: {e}")
            continue

        rs = np.asarray(ys["ratio_signal"])
        rl = np.asarray(ys["ratio_lost"])
        act_RF = load_act_RF_trimmed_mean(ts, data_root=data_root)
        act_str = f"{act_RF:.2f}" if act_RF is not None else "?"
        label = f"rep{rep} ({ts}, act_RF = {act_str})"

        order = np.argsort(x)
        x_s = np.asarray(x)[order]
        ax_lost.plot(x_s, rl[order], "-o", markersize=3, alpha=0.8, label=label)
        ax_trapped.plot(x_s, rs[order], "-o", markersize=3, alpha=0.8, label=label)

    ax_lost.set_title(f"RF={rf} dBm (n={len(timestamps)}) - lost")
    ax_lost.set_ylabel("ratio_lost")
    ax_lost.grid(True, linestyle="--", alpha=0.6)
    ax_lost.legend(loc="upper right", fontsize=7)

    ax_trapped.set_title(f"RF={rf} dBm (n={len(timestamps)}) - trapped")
    ax_trapped.set_ylabel("ratio_signal")
    ax_trapped.set_xlabel("Tickle Frequency (MHz)")
    ax_trapped.grid(True, linestyle="--", alpha=0.6)
    ax_trapped.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    path = f"{out_path}.png" if not str(out_path).lower().endswith(".png") else out_path
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved: {path}")


if __name__ == "__main__":
    parent = os.path.dirname(RF_SCAN_DIR)
    data_root = os.path.join(parent, DATA_SUBDIR)
    out_dir = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    groups, run_tag, _ = build_tickling_groups()

    if not groups:
        print("No tickling groups found.")
    else:
        print(f"[Config] data_root={data_root}")
        print(f"[Config] output_dir={out_dir}")
        print(f"[Plot] {len(groups)} RF groups, all repeats per figure")
        for g in groups:
            rf = g["RF_amplitude"]
            ts_list = g["timestamps"]
            rf_str = f"{float(rf):.2f}".replace(".", "p").replace("-", "m")
            out_path = os.path.join(out_dir, f"repeats_RF{rf_str}")
            print(f"  RF={rf} dBm ({len(ts_list)} repeats)", end=" ")
            plot_repeats_for_group(g, out_path, data_root=data_root)
