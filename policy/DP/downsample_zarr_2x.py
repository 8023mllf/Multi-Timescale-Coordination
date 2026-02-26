#!/usr/bin/env python3
"""
Downsample a DP-style Zarr dataset by temporal factor (supports 2x and 1.5x, etc).

Rules:
- Episodes are defined by meta/episode_ends (cumulative end indices).
- Any array anywhere in the tree with shape[0] == total_steps will be downsampled along time,
  episode-by-episode, consistently across arrays.
- meta/episode_ends is ALWAYS recomputed and overwritten in output.
- Other arrays (per-episode arrays, constants, etc.) are copied as-is.

factor semantics:
- factor = 2.0  -> keep roughly 1/2 frames (classic stride=2)
- factor = 1.5  -> keep roughly 2/3 frames by indices floor([0, 1.5, 3.0, 4.5, ...]) per episode
"""

import argparse
import math
import os
from typing import Tuple, List

import numpy as np

try:
    import zarr
except ImportError as e:
    raise SystemExit("Missing dependency: zarr. Install with: pip install zarr") from e


def _is_array(obj) -> bool:
    return hasattr(obj, "shape") and hasattr(obj, "dtype") and hasattr(obj, "__getitem__")


def _copy_group_attrs(src_group, dst_group):
    try:
        dst_group.attrs.update(dict(src_group.attrs))
    except Exception:
        pass


def _copy_array(src_arr, dst_group, name: str):
    chunks = getattr(src_arr, "chunks", None)
    dst_arr = dst_group.create_dataset(
        name,
        shape=src_arr.shape,
        dtype=src_arr.dtype,
        chunks=chunks,
        compressor=getattr(src_arr, "compressor", None),
        fill_value=getattr(src_arr, "fill_value", None),
        order=getattr(src_arr, "order", "C"),
        overwrite=True,
    )
    dst_arr[...] = src_arr[...]
    return dst_arr


def _ensure_group(dst_root, path: str):
    g = dst_root
    if path.strip("/") == "":
        return g
    for part in path.strip("/").split("/"):
        g = g.require_group(part)
    return g


def _list_arrays(group, prefix: str = "") -> List[Tuple[str, object]]:
    out: List[Tuple[str, object]] = []
    for k, v in group.items():
        p = f"{prefix}/{k}" if prefix else k
        if _is_array(v):
            out.append((p, v))
        else:
            out.extend(_list_arrays(v, p))
    return out


def _get_episode_ends(src_root) -> np.ndarray:
    if "meta" not in src_root:
        raise ValueError("Source zarr is missing 'meta' group.")
    meta = src_root["meta"]
    if "episode_ends" not in meta:
        raise ValueError("Source zarr is missing 'meta/episode_ends'.")
    episode_ends = np.asarray(meta["episode_ends"][...], dtype=np.int64)
    if episode_ends.ndim != 1 or episode_ends.size == 0:
        raise ValueError("meta/episode_ends must be a non-empty 1D array.")
    if not np.all(episode_ends[1:] >= episode_ends[:-1]):
        raise ValueError("meta/episode_ends must be non-decreasing.")
    return episode_ends


def _is_effectively_int(x: float, tol: float = 1e-12) -> bool:
    return abs(x - round(x)) <= tol


def _compute_new_episode_ends(old_episode_ends: np.ndarray, factor: float) -> Tuple[np.ndarray, np.ndarray]:
    starts = np.concatenate(([0], old_episode_ends[:-1]))
    ends = old_episode_ends
    lengths = ends - starts
    new_lengths = np.array([int(math.ceil(L / factor)) for L in lengths], dtype=np.int64)
    new_episode_ends = np.cumsum(new_lengths)
    return new_episode_ends, new_lengths


def _create_downsampled_array(dst_group, name: str, src_arr, new_total_steps: int):
    new_shape = (new_total_steps,) + tuple(src_arr.shape[1:])
    src_chunks = getattr(src_arr, "chunks", None)
    if src_chunks is None:
        chunks = (min(1024, new_total_steps),) + new_shape[1:]
    else:
        chunks0 = min(int(src_chunks[0]), new_total_steps) if len(src_chunks) > 0 else min(1024, new_total_steps)
        chunks = (chunks0,) + tuple(src_chunks[1:])

    dst_arr = dst_group.create_dataset(
        name,
        shape=new_shape,
        dtype=src_arr.dtype,
        chunks=chunks,
        compressor=getattr(src_arr, "compressor", None),
        fill_value=getattr(src_arr, "fill_value", None),
        order=getattr(src_arr, "order", "C"),
        overwrite=True,
    )
    return dst_arr


def _gather_time_indices(src_arr, idx: np.ndarray):
    try:
        if hasattr(src_arr, "oindex"):
            return src_arr.oindex[idx]
        return src_arr[idx]
    except Exception:
        return np.stack([src_arr[int(i)] for i in idx], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--stride", "-s", type=int, default=None, help="Legacy integer stride (e.g., 2).")
    ap.add_argument("--factor", "-f", type=float, default=None, help="Downsample factor (e.g., 2.0 or 1.5).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--recompute_episode_lengths", action="store_true")
    args = ap.parse_args()

    if args.factor is None and args.stride is None:
        factor = 2.0
    elif args.factor is not None:
        factor = float(args.factor)
    else:
        factor = float(args.stride)

    if factor <= 0:
        raise SystemExit("factor must be > 0")
    if factor < 1.0:
        raise SystemExit("factor must be >= 1.0 (this script does not upsample)")

    use_stride_slice = _is_effectively_int(factor)
    stride_int = int(round(factor)) if use_stride_slice else None

    in_path = args.input
    out_path = args.output

    if not os.path.exists(in_path):
        raise SystemExit(f"Input path not found: {in_path}")

    if os.path.exists(out_path):
        if args.overwrite:
            import shutil
            shutil.rmtree(out_path)
        else:
            raise SystemExit(f"Output path exists: {out_path}. Use --overwrite.")

    src = zarr.open(in_path, mode="r")
    episode_ends = _get_episode_ends(src)
    total_steps = int(episode_ends[-1])

    new_episode_ends, new_episode_lengths = _compute_new_episode_ends(episode_ends, factor)
    new_total_steps = int(new_episode_ends[-1])

    print(f"[downsample_zarr] Input : {in_path}")
    print(f"[downsample_zarr] Output: {out_path}")
    print(f"[downsample_zarr] Episodes: {len(episode_ends)}")
    if use_stride_slice:
        print(f"[downsample_zarr] Total steps: {total_steps} -> {new_total_steps} (stride={stride_int})")
    else:
        print(f"[downsample_zarr] Total steps: {total_steps} -> {new_total_steps} (factor={factor})")

    dst = zarr.open(out_path, mode="w")
    _copy_group_attrs(src, dst)

    dst_data = dst.require_group("data")
    dst_meta = dst.require_group("meta")

    if "data" in src:
        _copy_group_attrs(src["data"], dst_data)
    if "meta" in src:
        _copy_group_attrs(src["meta"], dst_meta)

    src_arrays = _list_arrays(src)

    created = {}
    for path, arr in src_arrays:
        if path == "meta/episode_ends":
            continue

        parent_path, name = (path.rsplit("/", 1)) if "/" in path else ("", path)
        dst_parent = _ensure_group(dst, parent_path)

        if arr.ndim >= 1 and int(arr.shape[0]) == total_steps:
            created[path] = _create_downsampled_array(dst_parent, name, arr, new_total_steps)
        else:
            created[path] = _copy_array(arr, dst_parent, name)

    # ALWAYS rewrite meta/episode_ends
    dst_meta.create_dataset(
        "episode_ends",
        data=new_episode_ends.astype(np.int64),
        shape=new_episode_ends.shape,
        dtype=np.int64,
        overwrite=True,
    )

    # Optional: rewrite episode_lengths if user asks AND source has it
    if args.recompute_episode_lengths and "meta" in src and "episode_lengths" in src["meta"]:
        dst_meta.create_dataset(
            "episode_lengths",
            data=new_episode_lengths.astype(np.int64),
            shape=new_episode_lengths.shape,
            dtype=np.int64,
            overwrite=True,
        )

    starts = np.concatenate(([0], episode_ends[:-1]))
    ends = episode_ends

    downsampled_paths = [
        p for p, arr in src_arrays
        if p != "meta/episode_ends"
        and _is_array(arr)
        and arr.ndim >= 1
        and int(arr.shape[0]) == total_steps
    ]

    src_time_arrays = {}
    for p in downsampled_paths:
        a = src
        for part in p.split("/"):
            a = a[part]
        src_time_arrays[p] = a

    write_pos = 0
    for ep_idx, (s, e) in enumerate(zip(starts, ends)):
        s = int(s); e = int(e)
        ep_len = e - s

        if use_stride_slice:
            idx_len = int(math.ceil(ep_len / stride_int))
            src_sel = slice(s, e, stride_int)
            dst_sel = slice(write_pos, write_pos + idx_len)
            for path in downsampled_paths:
                created[path][dst_sel] = src_time_arrays[path][src_sel]
            write_pos += idx_len
        else:
            offsets = np.floor(np.arange(0, ep_len, factor)).astype(np.int64)
            idx = s + offsets
            idx_len = int(idx.shape[0])
            dst_sel = slice(write_pos, write_pos + idx_len)
            for path in downsampled_paths:
                created[path][dst_sel] = _gather_time_indices(src_time_arrays[path], idx)
            write_pos += idx_len

        if (ep_idx + 1) % 50 == 0 or (ep_idx + 1) == len(episode_ends):
            print(f"[downsample_zarr] Episodes processed: {ep_idx + 1}/{len(episode_ends)}")

    # sanity check
    if write_pos != new_total_steps:
        raise RuntimeError(f"Internal error: wrote {write_pos} steps, expected {new_total_steps}")

    print("[downsample_zarr] Done.")
    print(f"[downsample_zarr] New dataset written to: {out_path}")


if __name__ == "__main__":
    main()
