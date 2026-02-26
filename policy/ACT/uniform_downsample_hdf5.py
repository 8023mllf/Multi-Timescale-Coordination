#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from typing import List
import h5py
import numpy as np
from glob import glob
from tempfile import mkstemp

DOWNSAMPLE_KEYS = [
    "/action",
    "/observations/qpos",
    "/observations/qvel",
    "/labels",
    "/labels_dp",
    "/entropy",
]

def _list_episode_files(dataset_dir: str) -> List[str]:
    # episode_*.hdf5 sorted by index
    files = sorted(glob(os.path.join(dataset_dir, "episode_*.hdf5")))
    return files

def _read_first_dim_length(h5: h5py.File, key: str) -> int:
    if key not in h5:
        return -1
    return int(h5[key].shape[0])

def _discover_cameras(h5: h5py.File) -> List[str]:
    cams = []
    if "/observations/images" in h5:
        img_group = h5["/observations/images"]
        for name, obj in img_group.items():
            if isinstance(obj, h5py.Dataset):
                cams.append(name)
    return cams

def _check_time_alignment(h5: h5py.File, cameras: List[str]) -> int:
    # Returns the reference T (prefer /action, else qpos), and checks shapes.
    candidates = [k for k in ["/action", "/observations/qpos"] if k in h5]
    if not candidates:
        raise RuntimeError("No /action or /observations/qpos found to infer length.")
    ref_key = candidates[0]
    T = h5[ref_key].shape[0]

    # Ensure all present keys share the same first dim (when applicable).
    keys_to_check = list(DOWNSAMPLE_KEYS)
    # Expand cameras
    keys_to_check += [f"/observations/images/{c}" for c in cameras]
    for key in keys_to_check:
        if key in h5:
            arr = h5[key]
            if arr.ndim == 0:
                continue
            if arr.shape[0] != T:
                raise RuntimeError(f"Time-length mismatch: {key} has {arr.shape[0]} vs reference {ref_key}={T}")
    return T

def _copy_attrs(src_group: h5py.Group, dst_group: h5py.Group):
    for k, v in src_group.attrs.items():
        dst_group.attrs[k] = v

def _create_like(dst_file: h5py.File, name: str, src_ds: h5py.Dataset, shape: tuple):
    # Create dataset with same dtype and (if available) compression/chunking
    kwargs = dict(dtype=src_ds.dtype, shape=shape)
    # Preserve compression settings when possible
    for attr in ["compression", "compression_opts", "shuffle", "fletcher32", "chunks"]:
        try:
            val = getattr(src_ds, attr)
        except Exception:
            val = None
        if val:
            kwargs[attr] = val
    return dst_file.create_dataset(name, **kwargs)

def _ensure_group(dst_file: h5py.File, group_path: str) -> h5py.Group:
    if group_path in dst_file:
        return dst_file[group_path]
    parts = [p for p in group_path.split("/") if p]
    cur = dst_file["/"]
    for p in parts:
        if p not in cur:
            cur = cur.create_group(p)
        else:
            cur = cur[p]
    return cur

def _downsample_array(arr: np.ndarray, stride: int) -> np.ndarray:
    return arr[::stride]

def _get_downsampled_indices(length: int, stride: float) -> np.ndarray:
    if stride <= 0:
        raise ValueError(f"Stride must be positive, got {stride}")

    # If stride is effectively an integer, we can just use slicing logic later,
    # but here we return the indices for consistency if needed.
    # However, the caller will handle integer strides separately for efficiency.

    out_len = int(length / stride)
    out_idxs = np.arange(out_len)
    # Map output index to input index: floor(i * stride)
    in_idxs = np.floor(out_idxs * stride).astype(int)
    # Clip just in case, though logic above should be safe
    in_idxs = np.clip(in_idxs, 0, length - 1)
    return in_idxs

def _process_one_file(src_path: str, stride: float, backup_ext: str = "") -> None:
    # Read original to get cameras and sanity check
    with h5py.File(src_path, "r") as src:
        cameras = _discover_cameras(src)
        T = _check_time_alignment(src, cameras)
        if stride <= 1.0:
            raise ValueError("Stride must be > 1.0 for downsampling.")
        if T < stride:
            print(f"[skip] {os.path.basename(src_path)} T={T} < stride={stride}")
            return

    # Pre-calculate indices if stride is float
    is_int_stride = (abs(stride - round(stride)) < 1e-9)
    indices = None
    if not is_int_stride:
        indices = _get_downsampled_indices(T, stride)

    # Backup (optional)
    if backup_ext:
        backup_path = src_path + backup_ext
        if not os.path.exists(backup_path):
            shutil.copy2(src_path, backup_path)

    # Write to temp file, then replace
    fd, tmp_path = mkstemp(prefix=os.path.basename(src_path) + ".", suffix=".tmp", dir=os.path.dirname(src_path))
    os.close(fd)
    try:
        with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
            # Copy root attrs
            _copy_attrs(src, dst)

            # Ensure groups exist
            _ensure_group(dst, "/observations")
            _ensure_group(dst, "/observations/images")

            # 1) Always try to downsample known keys if present
            for key in DOWNSAMPLE_KEYS:
                if key in src:
                    sds = src[key]
                    if sds.ndim == 0:
                        # Scalar attrs-like dataset: copy as-is
                        d = dst.create_dataset(key, data=sds[()])
                    else:
                        if is_int_stride:
                            arr = sds[::int(stride)]
                        else:
                            # For float stride, read all then subsample to avoid slow point-selection in h5py
                            # and to support arbitrary indices.
                            # Warning: this loads full dataset into memory.
                            arr = sds[:][indices]

                        d = _create_like(dst, key, sds, shape=arr.shape)
                        d[...] = arr

            # 2) Cameras
            for cam in _discover_cameras(src):
                key = f"/observations/images/{cam}"
                sds = src[key]
                if is_int_stride:
                    arr = sds[::int(stride)]
                else:
                    arr = sds[:][indices]
                d = _create_like(dst, key, sds, shape=arr.shape)
                d[...] = arr

            # 3) Copy through any other groups/datasets untouched (metadata, etc.)
            # We recursively walk the source and copy entries that we didn't just write.
            def _copy_other(name, obj):
                if isinstance(obj, h5py.Group):
                    if name not in dst:
                        _ensure_group(dst, name)
                        _copy_attrs(obj, dst[name])
                elif isinstance(obj, h5py.Dataset):
                    if name in dst:
                        return  # already written (downsampled)
                    # Copy dataset as-is
                    sds = src[name]
                    d = _create_like(dst, name, sds, shape=sds.shape)
                    d[...] = sds[()]
                    _copy_attrs(sds, d)
            src.visititems(_copy_other)

        # Replace atomically
        os.replace(tmp_path, src_path)
        print(f"[ok] Downsampled {os.path.basename(src_path)} by {stride}x")
    except Exception as e:
        # Clean temp file on error
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise e

def main():
    parser = argparse.ArgumentParser(description="Uniformly downsample episode_*.hdf5 files in-place.")
    parser.add_argument("--dataset-dirs", nargs="+", required=True, help="One or more dataset directories.")
    parser.add_argument("--stride", type=float, required=True, help="Downsample stride (e.g., 2 or 1.5).")
    parser.add_argument("--backup-ext", type=str, default="", help="If set (e.g., .bak), write a backup copy alongside each file.")
    args = parser.parse_args()

    if args.stride <= 1.0:
        print("ERROR: --stride must be > 1.0")
        sys.exit(2)

    for d in args.dataset_dirs:
        if not os.path.isdir(d):
            print(f"[warn] skip: not a directory: {d}")
            continue
        files = _list_episode_files(d)
        if not files:
            print(f"[warn] no episode_*.hdf5 found in {d}")
            continue
        print(f"Processing {len(files)} files in: {d}")
        for f in files:
            try:
                _process_one_file(f, args.stride, args.backup_ext)
            except Exception as e:
                print(f"[fail] {f}: {e}")

if __name__ == "__main__":
    main()
