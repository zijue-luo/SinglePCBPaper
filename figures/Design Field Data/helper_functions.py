"""
Shared utilities for multipole and arbitrary field plotting:
Cfile/electrode handling, .fld loading, field superposition, 2D/3D plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Cache for loaded .fld data: key = (design_dir_abspath, fld_name) -> (X, Y, Z, V)
_FLD_CACHE = {}

# Electrode order in Cfile (notebook order: bl, br, tl, tr - per PaulTrapAnalysis)
CFILE_ELECTRODE_ORDER = ['bl1', 'bl2', 'bl3', 'bl4', 'bl5',
                         'br1', 'br2', 'br3', 'br4', 'br5',
                         'tl1', 'tl2', 'tl3', 'tl4', 'tl5',
                         'tr1', 'tr2', 'tr3', 'tr4', 'tr5']
ELECTRODES = CFILE_ELECTRODE_ORDER.copy()
MULTIPOLES = ['Ex', 'Ey', 'Ez', 'U1', 'U2', 'U3', 'U4', 'U5']
N_ELEC = len(ELECTRODES)
N_MULT = len(MULTIPOLES)

# Permutation from Ansys to control-system coords (PaulTrapAnalysis perm=[2,0,1]):
# control_x = Ansys_Z, control_y = Ansys_X, control_z = Ansys_Y
PERM_ANSYS_TO_CTRL = (2, 0, 1)

# Electrode → .fld file mapping (PaulTrapAnalysis convention, mirror=True for tr/br)
ELECTRODE_FLD_MAP = {
    'tl1': ('tl1', False), 'tl2': ('tl2', False), 'tl3': ('tl3', False),
    'tl4': ('tl4', False), 'tl5': ('tl5', False),
    'tr1': ('tl1', True),  'tr2': ('tl2', True),  'tr3': ('tl3', True),
    'tr4': ('tl4', True),  'tr5': ('tl5', True),
    'bl1': ('bl1', False), 'bl2': ('bl2', False), 'bl3': ('bl3', False),
    'bl4': ('bl4', False), 'bl5': ('bl5', False),
    'br1': ('bl1', True),  'br2': ('bl2', True),  'br3': ('bl3', True),
    'br4': ('bl4', True),  'br5': ('bl5', True),
}


def load_cfile(cfile_path):
    """Load Cfile and return multipole matrix: elec -> mult -> value.
    Cfile uses notebook order: multipoles [Ex,Ey,Ez,U1,U2,U3,U4,U5], electrodes [bl,br,tl,tr].
    """
    with open(cfile_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    body = [ln.split() for ln in lines if ':' not in ln]
    matrix = {}
    for eindex, elec in enumerate(CFILE_ELECTRODE_ORDER):
        matrix[elec] = {}
        for mindex, mult in enumerate(MULTIPOLES):
            idx = eindex + mindex * N_ELEC
            if idx < len(body) and body[idx]:
                matrix[elec][mult] = float(body[idx][0])
            else:
                matrix[elec][mult] = 0.0
    return matrix


def get_multipole_voltages(cfile_path, multipole):
    """Get control voltages when multipole=1 (all others 0)."""
    matrix = load_cfile(cfile_path)
    return {elec: matrix[elec].get(multipole, 0.0) for elec in ELECTRODES}


def load_fld(fld_path):
    """Load .fld file using pandas; return X, Y, Z (in m), V (potential)."""
    df = pd.read_csv(fld_path, skiprows=2, sep=r'\s+', header=None,
                     usecols=[0, 1, 2, 3], engine='c')
    return df[0].values, df[1].values, df[2].values, df[3].values


# Binary format: magic "FLDB", version byte, n (uint64), then X,Y,Z,V as float64
_FLDB_MAGIC = b'FLDB'


def load_fld_binary(fldb_path):
    """Load .fldb binary file; return X, Y, Z (in m), V (potential)."""
    with open(fldb_path, 'rb') as f:
        magic = f.read(4)
        if magic != _FLDB_MAGIC:
            raise ValueError(f"Invalid fldb magic: {magic}")
        f.read(1)  # version
        n = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
        X = np.fromfile(f, dtype=np.float64, count=n)
        Y = np.fromfile(f, dtype=np.float64, count=n)
        Z = np.fromfile(f, dtype=np.float64, count=n)
        V = np.fromfile(f, dtype=np.float64, count=n)
    return X, Y, Z, V


def load_fld_cached(design_dir, fld_name):
    """Load field file with caching. Prefers .fldb if present, else .fld.
    Returns (X, Y, Z, V) or None if file missing."""
    design_abs = os.path.abspath(design_dir)
    fldb_path = os.path.join(design_dir, f"{fld_name}.fldb")
    fld_path = os.path.join(design_dir, f"{fld_name}.fld")
    if os.path.isfile(fldb_path):
        path_to_load = fldb_path
        loader = load_fld_binary
    elif os.path.isfile(fld_path):
        path_to_load = fld_path
        loader = load_fld
    else:
        return None
    key = (design_abs, fld_name)
    if key not in _FLD_CACHE:
        _FLD_CACHE[key] = loader(path_to_load)
    return _FLD_CACHE[key]


def compute_total_field(design_dir, voltages):
    """Superpose fields: sum over electrodes of (voltage[elec] * phi_elec)."""
    X, Y, Z, V = None, None, None, None
    for elec in ELECTRODES:
        fld_name, mirror = ELECTRODE_FLD_MAP[elec]
        data = load_fld_cached(design_dir, fld_name)
        if data is None:
            continue
        xi, yi, zi, vi = data
        if mirror:
            x_vals = np.unique(xi)
            y_vals = np.unique(yi)
            z_vals = np.unique(zi)
            nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
            vi_3d = vi.reshape(nx, ny, nz)
            vi = vi_3d[::-1, :, :].ravel()
        v_elec = voltages.get(elec, 0.0)
        if X is None:
            X, Y, Z = xi, yi, zi
            V = v_elec * vi
        else:
            V += v_elec * vi
    return X, Y, Z, V


def plot_2d_cross_sections(X, Y, Z, V, design_name, title_label, out_path, plot_region,
                           planes=('XY', 'XZ', 'YZ'), offsets=None, axes_in=None):
    """Plot 2D cross sections at XY, XZ, and/or YZ planes with optional offsets (in mm).
    planes: tuple of 'XY', 'XZ', 'YZ' to include.
    offsets: dict {'XY': z_offset_mm, 'XZ': y_offset_mm, 'YZ': x_offset_mm}; missing keys use 0.
    axes_in: if provided (sequence of matplotlib axes), plot to these and skip create/save.
    Uses control-system coords (perm applied).
    """
    if offsets is None:
        offsets = {}
    x_vals = np.unique(X)
    y_vals = np.unique(Y)
    z_vals = np.unique(Z)
    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
    V3 = V.reshape(nx, ny, nz)
    V3_ctrl = np.transpose(V3, PERM_ANSYS_TO_CTRL)  # shape (ctrl_x, ctrl_y, ctrl_z)
    ctrl_x_mm = z_vals * 1e3
    ctrl_y_mm = x_vals * 1e3
    ctrl_z_mm = y_vals * 1e3

    # Index for each plane: XY fixes iz, XZ fixes iy, YZ fixes ix
    def _idx(arr_mm, offset):
        return np.argmin(np.abs(arr_mm - offset))

    slices_data = []
    for plane in planes:
        off = offsets.get(plane, 0.0)
        if plane == 'XY':
            iz = _idx(ctrl_z_mm, off)
            x_ax, y_ax = ctrl_x_mm, ctrl_y_mm
            data = V3_ctrl[:, :, iz].T
            xlab, ylab = 'X (mm)', 'Y (mm)'
            title = f'{design_name}: XY at Z={ctrl_z_mm[iz]:.3f} mm'
        elif plane == 'XZ':
            iy = _idx(ctrl_y_mm, off)
            x_ax, y_ax = ctrl_x_mm, ctrl_z_mm
            data = V3_ctrl[:, iy, :].T
            xlab, ylab = 'X (mm)', 'Z (mm)'
            title = f'{design_name}: XZ at Y={ctrl_y_mm[iy]:.3f} mm'
        elif plane == 'YZ':
            ix = _idx(ctrl_x_mm, off)
            x_ax, y_ax = ctrl_y_mm, ctrl_z_mm
            data = V3_ctrl[ix, :, :].T
            xlab, ylab = 'Y (mm)', 'Z (mm)'
            title = f'{design_name}: YZ at X={ctrl_x_mm[ix]:.3f} mm'
        else:
            continue
        slices_data.append((x_ax, y_ax, data, xlab, ylab, title))

    if not slices_data:
        return

    vmin = min(s[2].min() for s in slices_data)
    vmax = max(s[2].max() for s in slices_data)
    if vmax <= vmin:
        vmin -= 1e-12
        vmax += 1e-12

    xlim = (plot_region[0], plot_region[1])
    if axes_in is not None:
        axes = list(axes_in)
    else:
        fig, axes = plt.subplots(1, len(slices_data), figsize=(5 * len(slices_data), 5))
        if len(slices_data) == 1:
            axes = [axes]
    for ax, (x_ax, y_ax, data, xlab, ylab, title) in zip(axes, slices_data):
        im = ax.pcolormesh(x_ax, y_ax, data, cmap='RdBu_r', shading='auto', vmin=vmin, vmax=vmax)
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Potential (V)')

    if axes_in is None:
        plt.suptitle(f'{title_label} field — {design_name}')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")


def plot_2d_slices(X, Y, Z, V, design_name, title_label, out_path, plot_region):
    """Plot XY at Z=0 and XZ at Y=0 (convenience wrapper for plot_2d_cross_sections)."""
    plot_2d_cross_sections(X, Y, Z, V, design_name, title_label, out_path, plot_region,
                           planes=('XY', 'XZ'), offsets={'XY': 0, 'XZ': 0})


def plot_axis_line(X, Y, Z, V, design_name, title_label, out_path, plot_region,
                   axis='X', offsets=None, ax_in=None):
    """Plot potential V along one axis at fixed offsets for the other two (control-system coords).
    axis: 'X', 'Y', or 'Z' — axis to vary.
    offsets: dict of fixed coordinates in mm, e.g. for axis='X': {'Y': 0, 'Z': 0}.
             Missing keys default to 0.
    ax_in: if provided, plot to this axis and skip create/save.
    """
    if offsets is None:
        offsets = {}
    x_vals = np.unique(X)
    y_vals = np.unique(Y)
    z_vals = np.unique(Z)
    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
    V3 = V.reshape(nx, ny, nz)
    V3_ctrl = np.transpose(V3, PERM_ANSYS_TO_CTRL)  # shape (ctrl_x, ctrl_y, ctrl_z)
    ctrl_x_mm = z_vals * 1e3
    ctrl_y_mm = x_vals * 1e3
    ctrl_z_mm = y_vals * 1e3

    def _idx(arr_mm, offset):
        return np.argmin(np.abs(arr_mm - offset))

    axis = axis.upper()
    if axis == 'X':
        iy = _idx(ctrl_y_mm, offsets.get('Y', 0))
        iz = _idx(ctrl_z_mm, offsets.get('Z', 0))
        pos_mm = ctrl_x_mm
        line = V3_ctrl[:, iy, iz]
        xlab = 'X (mm)'
        title = f'{design_name}: along X at Y={ctrl_y_mm[iy]:.3f}, Z={ctrl_z_mm[iz]:.3f} mm'
    elif axis == 'Y':
        ix = _idx(ctrl_x_mm, offsets.get('X', 0))
        iz = _idx(ctrl_z_mm, offsets.get('Z', 0))
        pos_mm = ctrl_y_mm
        line = V3_ctrl[ix, :, iz]
        xlab = 'Y (mm)'
        title = f'{design_name}: along Y at X={ctrl_x_mm[ix]:.3f}, Z={ctrl_z_mm[iz]:.3f} mm'
    elif axis == 'Z':
        ix = _idx(ctrl_x_mm, offsets.get('X', 0))
        iy = _idx(ctrl_y_mm, offsets.get('Y', 0))
        pos_mm = ctrl_z_mm
        line = V3_ctrl[ix, iy, :]
        xlab = 'Z (mm)'
        title = f'{design_name}: along Z at X={ctrl_x_mm[ix]:.3f}, Y={ctrl_y_mm[iy]:.3f} mm'
    else:
        raise ValueError("axis must be 'X', 'Y', or 'Z'")

    if ax_in is not None:
        ax = ax_in
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pos_mm, line, 'b-', linewidth=1.5)
    ax.set_xlim(plot_region[0], plot_region[1])
    ax.set_xlabel(xlab)
    ax.set_ylabel('Potential (V)')
    ax.set_title(title)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    if ax_in is None:
        plt.suptitle(f'{title_label} field — {design_name}', y=1.02)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")


def plot_six_panel(X, Y, Z, V, design_name, title_label, out_path, plot_region,
                   offsets=None):
    """Plot 6 panels on one figure: XY, XZ, YZ cross sections (row 0) and X, Y, Z axis lines (row 1).
    All at zero offset by default. offsets: dict for cross-section and axis offsets (see plot_2d_cross_sections and plot_axis_line).
    """
    if offsets is None:
        offsets = {}
    cs_offsets = {p: offsets.get(p, 0.0) for p in ('XY', 'XZ', 'YZ')}
    ax_offsets = {a: offsets.get(a, 0.0) for a in ('X', 'Y', 'Z')}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plot_2d_cross_sections(X, Y, Z, V, design_name, title_label, None, plot_region,
                           planes=('XY', 'XZ', 'YZ'), offsets=cs_offsets, axes_in=axes[0, :])
    for col, axis in enumerate(('X', 'Y', 'Z')):
        off = {'Y': ax_offsets['Y'], 'Z': ax_offsets['Z']} if axis == 'X' else \
              {'X': ax_offsets['X'], 'Z': ax_offsets['Z']} if axis == 'Y' else \
              {'X': ax_offsets['X'], 'Y': ax_offsets['Y']}
        plot_axis_line(X, Y, Z, V, design_name, title_label, None, plot_region,
                       axis=axis, offsets=off, ax_in=axes[1, col])
    plt.suptitle(f'{title_label} field — {design_name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_3d_contour(X, Y, Z, V, design_name, title_label, out_path, plot_region, n_isosurfaces=6):
    """Plot true 3D isosurfaces (marching cubes). title_label: e.g. multipole name or 'Custom'."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("  Skipping 3D contour: scikit-image required for marching_cubes")
        return

    x_vals = np.unique(X)
    y_vals = np.unique(Y)
    z_vals = np.unique(Z)
    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
    V3 = V.reshape(nx, ny, nz)
    V3_ctrl = np.transpose(V3, PERM_ANSYS_TO_CTRL)
    ctrl_x_mm = z_vals * 1e3
    ctrl_y_mm = x_vals * 1e3
    ctrl_z_mm = y_vals * 1e3

    vmin = float(np.nanmin(V3_ctrl))
    vmax = float(np.nanmax(V3_ctrl))
    if vmax <= vmin:
        vmin -= 1e-12
        vmax += 1e-12

    origin = (float(ctrl_x_mm[0]), float(ctrl_y_mm[0]), float(ctrl_z_mm[0]))
    dx = (ctrl_x_mm[-1] - ctrl_x_mm[0]) / max(1, len(ctrl_x_mm) - 1)
    dy = (ctrl_y_mm[-1] - ctrl_y_mm[0]) / max(1, len(ctrl_y_mm) - 1)
    dz = (ctrl_z_mm[-1] - ctrl_z_mm[0]) / max(1, len(ctrl_z_mm) - 1)
    spacing = (dx, dy, dz)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    try:
        cmap = plt.colormaps['RdBu_r']
    except AttributeError:
        cmap = plt.cm.get_cmap('RdBu_r')
    levels = np.linspace(vmin, vmax, n_isosurfaces + 2)[1:-1]

    for level in levels:
        try:
            verts, faces, _, _ = marching_cubes(V3_ctrl, level=level, spacing=spacing)
        except (ValueError, RuntimeError):
            continue
        verts_phys = verts + np.array(origin)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        tri_verts = verts_phys[faces]
        poly = Poly3DCollection(tri_verts, alpha=0.6, linewidths=0.1, edgecolors='none')
        color = cmap((level - vmin) / (vmax - vmin)) if vmax > vmin else (0.5, 0.5, 0.5, 0.6)
        poly.set_facecolor(color)
        ax.add_collection3d(poly)

    ax.set_xlim(plot_region[0], plot_region[1])
    ax.set_ylim(plot_region[0], plot_region[1])
    ax.set_zlim(plot_region[0], plot_region[1])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'{title_label} field — {design_name} (3D isosurfaces)')
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Potential (V)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_comparison(design_dirs, multipole, out_path, plot_region):
    """Plot both designs side by side with shared color scale (multipole=1 from Cfile)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    xlim = (plot_region[0], plot_region[1])
    zlim = (plot_region[0], plot_region[1])

    all_V3_ctrl = []
    all_ix0 = []
    all_iy0 = []
    all_ctrl_x_mm = []
    all_ctrl_y_mm = []
    all_ctrl_z_mm = []
    design_names = []
    for design_name, design_dir in design_dirs:
        cfile_path = os.path.join(design_dir, 'Cfile.txt')
        if not os.path.isfile(cfile_path):
            continue
        voltages = get_multipole_voltages(cfile_path, multipole)
        X, Y, Z, V = compute_total_field(design_dir, voltages)
        if X is None:
            continue
        x_vals = np.unique(X)
        y_vals = np.unique(Y)
        z_vals = np.unique(Z)
        nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
        V3 = V.reshape(nx, ny, nz)
        V3_ctrl = np.transpose(V3, PERM_ANSYS_TO_CTRL)
        ix0 = np.argmin(np.abs(x_vals))
        iy0 = np.argmin(np.abs(y_vals))
        all_V3_ctrl.append(V3_ctrl)
        all_ix0.append(ix0)
        all_iy0.append(iy0)
        all_ctrl_x_mm.append(z_vals * 1e3)
        all_ctrl_y_mm.append(x_vals * 1e3)
        all_ctrl_z_mm.append(y_vals * 1e3)
        design_names.append(design_name)

    if not all_V3_ctrl:
        plt.close()
        return
    vmin = min(
        min(V3[:, :, iy0].min(), V3[:, ix0, :].min())
        for V3, ix0, iy0 in zip(all_V3_ctrl, all_ix0, all_iy0)
    )
    vmax = max(
        max(V3[:, :, iy0].max(), V3[:, ix0, :].max())
        for V3, ix0, iy0 in zip(all_V3_ctrl, all_ix0, all_iy0)
    )
    if vmax <= vmin:
        vmin -= 1e-12
        vmax += 1e-12

    for col, (V3_ctrl, ix0, iy0, ctrl_x_mm, ctrl_y_mm, ctrl_z_mm, design_name) in enumerate(
            zip(all_V3_ctrl, all_ix0, all_iy0, all_ctrl_x_mm, all_ctrl_y_mm, all_ctrl_z_mm, design_names)):
        ax = axes[0, col]
        im = ax.pcolormesh(ctrl_x_mm, ctrl_y_mm, V3_ctrl[:, :, iy0].T,
                           cmap='RdBu_r', shading='auto', vmin=vmin, vmax=vmax)
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title(f'{design_name} — XY at Z≈0')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Potential (V)')

        ax = axes[1, col]
        im = ax.pcolormesh(ctrl_x_mm, ctrl_z_mm, V3_ctrl[:, ix0, :].T,
                           cmap='RdBu_r', shading='auto', vmin=vmin, vmax=vmax)
        ax.set_xlim(xlim)
        ax.set_ylim(zlim)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title(f'{design_name} — XZ at Y≈0')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Potential (V)')

    plt.suptitle(f'{multipole}=1 actual field comparison (Cfile control voltages)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison: {out_path}")
