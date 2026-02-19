"""
Convert .fld (ASCII) field files to binary .fldb format for faster loading and less disk use.
Run once after adding new .fld files; helper_functions will prefer .fldb when available.
"""

import os
import numpy as np

from helper_functions import load_fld

# Binary format: magic "FLDB", version 1, n (uint64), then X, Y, Z, V as float64 arrays
FLDB_MAGIC = b'FLDB'
FLDB_VERSION = 1


def write_fld_binary(fldb_path, X, Y, Z, V):
    """Write (X, Y, Z, V) to binary .fldb file."""
    n = len(X)
    assert len(Y) == len(Z) == len(V) == n
    with open(fldb_path, 'wb') as f:
        f.write(FLDB_MAGIC)
        f.write(bytes([FLDB_VERSION]))
        f.write(np.uint64(n).tobytes())
        X.astype(np.float64).tofile(f)
        Y.astype(np.float64).tofile(f)
        Z.astype(np.float64).tofile(f)
        V.astype(np.float64).tofile(f)


def convert_fld_to_binary(fld_path):
    """Convert single .fld to .fldb. Returns path to .fldb or None on error."""
    if not fld_path.endswith('.fld') or not os.path.isfile(fld_path):
        return None
    fldb_path = fld_path[:-4] + '.fldb'
    try:
        X, Y, Z, V = load_fld(fld_path)
        write_fld_binary(fldb_path, X, Y, Z, V)
        return fldb_path
    except Exception as e:
        print(f"  Error converting {fld_path}: {e}")
        return None


def convert_folder(folder_path):
    """Convert all .fld files in folder to .fldb."""
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return 0
    count = 0
    for name in sorted(os.listdir(folder_path)):
        if name.endswith('.fld'):
            fld_path = os.path.join(folder_path, name)
            out = convert_fld_to_binary(fld_path)
            if out:
                count += 1
                print(f"  {name} -> {name[:-4]}.fldb")
    return count


# ============== Config at top ==============
# Folders (relative to script dir) to convert
FOLDERS = ['DC_in_use', 'DC_w_shielding']
# ==========================================


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    total = 0
    for folder in FOLDERS:
        path = os.path.join(base, folder)
        print(f"Converting {folder}...")
        total += convert_folder(path)
    print(f"Converted {total} files total.")


if __name__ == '__main__':
    main()
