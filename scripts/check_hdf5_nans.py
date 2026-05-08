"""
Check an HDF5 dataset (raw v2.0 format) for NaN / Inf values.

Usage:
    python check_hdf5_nans.py path/to/dataset.h5 [--verbose]

Prints a summary of which arrays contain NaN/Inf and which structure
indices are affected.  Exit code 0 = clean, 1 = issues found.
"""

import argparse
import sys

import h5py
import numpy as np


def check_array(name: str, data: np.ndarray, offsets=None, per_config=True):
    """
    Returns a list of bad config indices (empty = clean).

    Args:
        name: Dataset name for reporting.
        data: Flat float array from HDF5.
        offsets: CSR offsets array (num_configs+1,) for per-atom arrays.
                 If None, data is treated as per-config already.
        per_config: If True each row is one config; if False use offsets.
    """
    bad_configs = []
    nan_mask = np.isnan(data) | np.isinf(data)
    if not nan_mask.any():
        return bad_configs

    if offsets is not None:
        # Per-atom array: map bad atom indices back to configs
        bad_atom_idx = np.where(nan_mask.any(axis=-1) if data.ndim > 1
                                else nan_mask)[0]
        for atom_i in bad_atom_idx:
            # Binary search: offsets[cfg] <= atom_i < offsets[cfg+1]
            cfg = int(np.searchsorted(offsets, atom_i, side="right")) - 1
            if cfg not in bad_configs:
                bad_configs.append(cfg)
    else:
        # Per-config array
        if data.ndim > 1:
            bad_row = np.where(nan_mask.any(axis=tuple(range(1, data.ndim))))[0]
        else:
            bad_row = np.where(nan_mask)[0]
        bad_configs = bad_row.tolist()

    return bad_configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5_path", help="Path to the HDF5 dataset file")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print the bad config indices for each affected array"
    )
    args = parser.parse_args()

    found_issues = False

    with h5py.File(args.hdf5_path, "r") as f:
        fmt = f.attrs.get("format_version", "unknown")
        fmt_type = f.attrs.get("format_type", "unknown")
        num_configs = int(f.attrs.get("num_configs", -1))
        print(f"File    : {args.hdf5_path}")
        print(f"Format  : v{fmt} ({fmt_type})")
        print(f"Configs : {num_configs}")
        print()

        offsets = f["offsets"][:] if "offsets" in f else None

        # ── per-config scalar / small arrays ──────────────────────────────
        per_config_paths = [
            "cell",
            "pbc",
            "positions",   # per-atom — handled separately with offsets
        ]

        # Explicit per-config datasets (one row = one config)
        for key in ["cell"]:
            if key not in f:
                continue
            data = f[key][:]
            if not np.issubdtype(data.dtype, np.floating):
                continue
            bad = check_array(key, data, offsets=None)
            if bad:
                found_issues = True
                print(f"[BAD] {key}: {len(bad)} config(s) with NaN/Inf")
                if args.verbose:
                    print(f"      indices: {bad[:50]}{'...' if len(bad) > 50 else ''}")
            else:
                print(f"[ ok] {key}")

        # per-atom arrays (use offsets to map back to configs)
        for key in ["positions", "atomic_numbers"]:
            if key not in f:
                continue
            data = f[key][:]
            if not np.issubdtype(data.dtype, np.floating):
                continue
            bad = check_array(key, data, offsets=offsets)
            if bad:
                found_issues = True
                print(f"[BAD] {key}: {len(bad)} config(s) with NaN/Inf")
                if args.verbose:
                    print(f"      indices: {bad[:50]}{'...' if len(bad) > 50 else ''}")
            else:
                print(f"[ ok] {key}")

        # ── property_weights ──────────────────────────────────────────────
        if "property_weights" in f:
            for key in f["property_weights"]:
                data = f[f"property_weights/{key}"][:]
                if not np.issubdtype(data.dtype, np.floating):
                    continue
                # NaN is the sentinel for "absent" in property_weights — skip
                # (NaN here just means the property isn't set for that config)

        # ── properties/info  (per-config) ─────────────────────────────────
        if "properties/info" in f:
            for key in f["properties/info"]:
                path = f"properties/info/{key}"
                data = f[path][:]
                if not np.issubdtype(data.dtype, np.floating):
                    continue
                bad = check_array(path, data, offsets=None)
                if bad:
                    found_issues = True
                    print(f"[BAD] {path}: {len(bad)} config(s) with NaN/Inf")
                    if args.verbose:
                        print(f"      indices: {bad[:50]}{'...' if len(bad) > 50 else ''}")
                else:
                    print(f"[ ok] {path}")

        # ── properties/arrays  (per-atom) ─────────────────────────────────
        if "properties/arrays" in f:
            for key in f["properties/arrays"]:
                path = f"properties/arrays/{key}"
                data = f[path][:]
                if not np.issubdtype(data.dtype, np.floating):
                    continue
                bad = check_array(path, data, offsets=offsets)
                if bad:
                    found_issues = True
                    print(f"[BAD] {path}: {len(bad)} config(s) with NaN/Inf")
                    if args.verbose:
                        print(f"      indices: {bad[:50]}{'...' if len(bad) > 50 else ''}")
                else:
                    print(f"[ ok] {path}")

    print()
    if found_issues:
        print("Result: ISSUES FOUND (see [BAD] lines above)")
        sys.exit(1)
    else:
        print("Result: OK — no NaN/Inf found")
        sys.exit(0)


if __name__ == "__main__":
    main()
