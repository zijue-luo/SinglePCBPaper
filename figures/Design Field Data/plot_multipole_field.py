"""
Plot the actual electric potential field generated when applying a given multipole=1,
using the control voltages defined by each design's Cfile.txt.

Superposes the per-electrode .fld files (potential when electrode at 1V)
weighted by the corresponding multipole column from Cfile.
"""

import os

from helper_functions import (
    MULTIPOLES,
    compute_total_field,
    get_multipole_voltages,
    plot_2d_slices,
    plot_3d_contour,
    plot_comparison,
)

# ============== User-configurable at top ==============
# Multipole to plot (Ex, Ey, Ez, U1, U2, U3, U4, U5)
MULTIPOLE = 'U2'

# Design folders (subfolders of script directory)
FOLDERS = ['DC_in_use', 'DC_w_shielding']

# Plot region [xmin, xmax] in mm (same for y, z); data range is [-0.25, 0.25]
PLOT_REGION = [-0.25, 0.25]

# Set to list of multipoles to batch-generate all; None = single run using MULTIPOLE
BATCH_MULTIPOLES = ['Ex', 'Ey', 'Ez', 'U1', 'U2', 'U3', 'U4', 'U5']  # set to None for single multipole

# Hybrid runs: (cfile_folder, fld_folder, output_folder) — use Cfile from first, .fld from second
# Example: Cfile from DC_in_use, field from DC_w_shielding
HYBRID_RUNS = [('DC_in_use', 'DC_w_shielding', 'hybrid_Cfile_in_use_fld_w_shielding')]
# ======================================================


def run_single(base, multipole, folders, plot_region):
    """Generate plots for one multipole."""
    for design in folders:
        design_dir = os.path.join(base, design)
        cfile_path = os.path.join(design_dir, 'Cfile.txt')
        if not os.path.isfile(cfile_path):
            print(f"Skipping {design}: Cfile.txt not found")
            continue
        voltages = get_multipole_voltages(cfile_path, multipole)
        X, Y, Z, V = compute_total_field(design_dir, voltages)
        if X is None:
            print(f"  No .fld files in {design_dir}")
            continue
        # Save in design folder (2D slices + 3D contour)
        out_path = os.path.join(design_dir, f'field_{multipole}.png')
        plot_2d_slices(X, Y, Z, V, design, multipole, out_path, plot_region)
        out_3d = os.path.join(design_dir, f'field_{multipole}_3d.png')
        plot_3d_contour(X, Y, Z, V, design, multipole, out_3d, plot_region)

    design_dirs = [(d, os.path.join(base, d)) for d in folders
                  if os.path.isfile(os.path.join(base, d, 'Cfile.txt'))]
    if len(design_dirs) >= 2:
        comp_path = os.path.join(base, f'field_{multipole}_comparison.png')
        plot_comparison(design_dirs, multipole, comp_path, plot_region)


def run_hybrid(base, cfile_folder, fld_folder, output_folder, multipole, plot_region):
    """Generate plots using Cfile from cfile_folder and .fld from fld_folder, save to output_folder."""
    cfile_dir = os.path.join(base, cfile_folder)
    fld_dir = os.path.join(base, fld_folder)
    out_dir = os.path.join(base, output_folder)
    os.makedirs(out_dir, exist_ok=True)

    cfile_path = os.path.join(cfile_dir, 'Cfile.txt')
    if not os.path.isfile(cfile_path):
        print(f"  Hybrid: Cfile not found at {cfile_path}")
        return
    voltages = get_multipole_voltages(cfile_path, multipole)
    X, Y, Z, V = compute_total_field(fld_dir, voltages)
    if X is None:
        print(f"  Hybrid: No .fld files in {fld_dir}")
        return

    design_name = f"Cfile:{cfile_folder}, fld:{fld_folder}"
    out_path = os.path.join(out_dir, f'field_{multipole}.png')
    plot_2d_slices(X, Y, Z, V, design_name, multipole, out_path, plot_region)
    out_3d = os.path.join(out_dir, f'field_{multipole}_3d.png')
    plot_3d_contour(X, Y, Z, V, design_name, multipole, out_3d, plot_region)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    plot_region = PLOT_REGION
    multipoles = BATCH_MULTIPOLES if BATCH_MULTIPOLES is not None else [MULTIPOLE]

    for mp in multipoles:
        if mp not in MULTIPOLES:
            print(f"Skipping unknown multipole: {mp}")
            continue
        print(f"\n--- {mp}=1 ---")
        try:
            run_single(base, mp, FOLDERS, plot_region)
            # Hybrid runs: Cfile from one folder, .fld from another
            for cfile_folder, fld_folder, output_folder in (HYBRID_RUNS or []):
                print(f"\n--- Hybrid {mp}=1 (Cfile:{cfile_folder}, fld:{fld_folder}) ---")
                run_hybrid(base, cfile_folder, fld_folder, output_folder, mp, plot_region)
        except Exception as e:
            print(f"ERROR processing {mp}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
