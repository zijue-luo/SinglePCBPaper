"""
Plot the electric potential field for user-defined electrode voltages.
Set VOLTAGES (and optionally FLD_FOLDER, PLOT_REGION, etc.) at the top, then run.
Uses the same .fld files and superposition as plot_multipole_field; no code duplication.
"""

import os

from helper_functions import (
    ELECTRODES,
    compute_total_field,
    plot_2d_slices,
    plot_3d_contour,
)

# ============== Custom voltages at top (edit here) ==============
# Electrode names: bl1..bl5, br1..br5, tl1..tl5, tr1..tr5. Omitted electrodes = 0 V.
VOLTAGES = {
    'tl1': -2.29, 'tl2': -2.81, 'tl3': +7.77, 'tl4': -1.57, 'tl5': -1.89,
    'tr1': -2.46, 'tr2': -3.65, 'tr3': +5.15, 'tr4': -2.39, 'tr5': -2.05,
    'bl1': -2.75, 'bl2': -4.08, 'bl3': +6.06, 'bl4': -2.86, 'bl5': -2.24,
    'br1': -2.90, 'br2': -4.91, 'br3': +3.47, 'br4': -3.69, 'br5': -2.37,
}

# Folder containing .fld files (subfolder of this script's directory)
FLD_FOLDER = 'DC_w_shielding'

# Plot region [xmin, xmax] in mm (same for y, z)
PLOT_REGION = [-0.25, 0.25]

# Label used in plot titles (e.g. 'Custom' or 'My trap')
TITLE_LABEL = 'Custom'

# Output filename base (without extension); 2D and 3D get _3d suffix for 3D
OUTPUT_BASE = 'field_arbitrary'
# ================================================================


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    design_dir = os.path.join(base, FLD_FOLDER)
    if not os.path.isdir(design_dir):
        print(f"Folder not found: {design_dir}")
        return

    # Ensure all electrodes have a value (default 0)
    voltages = {e: VOLTAGES.get(e, 0.0) for e in ELECTRODES}

    X, Y, Z, V = compute_total_field(design_dir, voltages)
    if X is None:
        print(f"No .fld files found in {design_dir}")
        return

    design_name = FLD_FOLDER
    out_2d = os.path.join(design_dir, f'{OUTPUT_BASE}.png')
    out_3d = os.path.join(design_dir, f'{OUTPUT_BASE}_3d.png')

    plot_2d_slices(X, Y, Z, V, design_name, TITLE_LABEL, out_2d, PLOT_REGION)
    plot_3d_contour(X, Y, Z, V, design_name, TITLE_LABEL, out_3d, PLOT_REGION)
    print("Done.")


if __name__ == '__main__':
    main()
