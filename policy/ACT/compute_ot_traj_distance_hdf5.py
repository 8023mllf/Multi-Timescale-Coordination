#!/usr/bin/env python3
"""
Compute OT (2-Wasserstein) distance per episode between
original and downsampled HDF5 episode datasets.

Assumptions:
- Episode files named: episode_{id}.hdf5
- Each episode file contains:
    /action                         (T, Da) float32
    /observations/qpos              (T, Dq) float32
    /observations/images/<cam_name> (T, H, W, 3) uint8  [NOT used by default]

Usage:
    python compute_ot_traj_distance_hdf5.py

Dependencies:
    pip install pot h5py numpy pandas
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import h5py
import numpy as np
import pandas as pd
import ot  # POT


# =========================
# Config (edit as needed)
# =========================

# Two dataset dirs that contain episode_*.hdf5
ROOT_ORI = Path("/home/mn/RoboTwin-main/policy/ACT/processed_data/sim_beat_block_hammer/demo_clean")
ROOT_DS  = Path("/home/mn/RoboTwin-main/policy/ACT/processed_data/sim_beat_block_hammer/demo_clean_2x")  # example

# Which features define "trajectory geometry"
# Supported keys (examples):
#   "observations/qpos"  -> /observations/qpos
#   "action"             -> /action
#   "observations/left_arm_dim" -> /observations/left_arm_dim
# FEATURE_KEYS: List[str] = ["observations/qpos"]
# If you want, you can do:
FEATURE_KEYS = ["observations/qpos", "action"]

# Add time as an extra dimension (recommended for downsample comparison)
TIME_SCALE = 1.0   # set 0 to disable time feature

# If True, try to use a timestamp dataset (if exists); otherwise fallback["observations/qpos", "action"] to linspace(0,1)
USE_TIMESTAMP = False
TIMESTAMP_CANDIDATES = [
    "observations/timestamp",  # /observations/timestamp
    "timestamp",               # /timestamp
]

# Sinkhorn entropy regularization
SINKHORN_REG = 1e-2
USE_SINKHORN = True
SINKHORN_NUMITER = 2000

# Normalize feature dimensions using global mean/std (recommended)
NORMALIZE = True
# stats computed from: "ori" or "both"
STATS_SOURCE = "ori"  # "ori" | "both"

# Output CSV
OUT_CSV = Path("ot_scores_per_episode_hdf5.csv")


# =========================
# Helpers
# =========================

def _episode_id_from_name(p: Path) -> Optional[int]:
    m = re.match(r"episode_(\d+)\.hdf5$", p.name)
    if not m:
        return None
    return int(m.group(1))

def list_episode_ids(root: Path) -> List[int]:
    ids = []
    for p in root.glob("episode_*.hdf5"):
        eid = _episode_id_from_name(p)
        if eid is not None:
            ids.append(eid)
    return sorted(ids)

def _key_to_h5_path(key: str) -> str:
    # allow "action" shorthand
    if key == "action":
        return "/action"
    # allow keys without leading slash
    if not key.startswith("/"):
        return "/" + key
    return key

def _read_dataset(f: h5py.File, key: str) -> np.ndarray:
    path = _key_to_h5_path(key)
    if path not in f:
        raise KeyError(f"Missing dataset '{path}' in file: {f.filename}")
    arr = f[path][()]
    # Ensure float32 for OT feature space
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32)
    return arr

def episode_to_array(
    episode_path: Path,
    feature_keys: List[str],
    time_scale: float = 0.0,
    use_timestamp: bool = False,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """
    Returns:
        X: (T, D_total [+1 time]) float32
        T: length
    """
    with h5py.File(episode_path, "r") as f:
        feats = []
        T_ref = None

        for key in feature_keys:
            arr = _read_dataset(f, key)
            # arr can be (T,) or (T, D)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.ndim != 2:
                raise ValueError(
                    f"Feature '{key}' must be 1D or 2D (got shape {arr.shape}). "
                    f"Do NOT use raw images here; embed them first if needed."
                )
            if T_ref is None:
                T_ref = arr.shape[0]
            else:
                if arr.shape[0] != T_ref:
                    raise ValueError(
                        f"Episode {episode_path.name} has inconsistent lengths: "
                        f"previous T={T_ref}, feature '{key}' has T={arr.shape[0]}"
                    )
            feats.append(arr)

        if T_ref is None:
            return np.zeros((0, 0), dtype=np.float32), 0

        X = np.concatenate(feats, axis=1)  # (T, D)

        # normalize feature dims if stats provided
        if feature_mean is not None and feature_std is not None:
            X = (X - feature_mean[None, :]) / (feature_std[None, :] + 1e-12)

        # optional time feature
        if time_scale is not None and time_scale > 0:
            t = None
            if use_timestamp:
                for cand in TIMESTAMP_CANDIDATES:
                    p = _key_to_h5_path(cand)
                    if p in f:
                        t_raw = f[p][()]
                        t_raw = np.array(t_raw, dtype=np.float32).reshape(-1, 1)
                        if t_raw.shape[0] != T_ref:
                            raise ValueError(
                                f"Timestamp '{p}' length mismatch: {t_raw.shape[0]} vs T={T_ref}"
                            )
                        t_min, t_max = float(t_raw.min()), float(t_raw.max())
                        if t_max > t_min:
                            t = (t_raw - t_min) / (t_max - t_min)
                        else:
                            t = np.zeros_like(t_raw, dtype=np.float32)
                        break
                if t is None:
                    # fallback
                    t = np.linspace(0.0, 1.0, num=T_ref, dtype=np.float32).reshape(-1, 1)
            else:
                t = np.linspace(0.0, 1.0, num=T_ref, dtype=np.float32).reshape(-1, 1)

            X = np.concatenate([X, t * float(time_scale)], axis=1)

    return X.astype(np.float32), int(T_ref)

def wasserstein2_distance(
    X: np.ndarray,
    Y: np.ndarray,
    reg: float = SINKHORN_REG,
    use_sinkhorn: bool = True,
    num_itermax: int = SINKHORN_NUMITER,
) -> float:
    """
    Compute 2-Wasserstein distance between two point clouds X, Y.
    Uses squared Euclidean cost -> W2.

    Returns:
        W2 distance (sqrt of optimal cost).
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X, Y must be 2D arrays (T, D).")
    n, m = X.shape[0], Y.shape[0]
    if n == 0 or m == 0:
        return float("nan")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Dim mismatch: X is {X.shape}, Y is {Y.shape}")

    a = np.ones((n,), dtype=np.float64) / n
    b = np.ones((m,), dtype=np.float64) / m

    # cost matrix: squared euclidean
    C = ot.dist(X.astype(np.float64), Y.astype(np.float64), metric="euclidean") ** 2

    cmax = float(C.max())
    if cmax > 0:
        C /= (cmax + 1e-12)

    if use_sinkhorn:
        w2_sq = ot.sinkhorn2(a, b, C, reg=reg, numItermax=num_itermax)
    else:
        w2_sq = ot.emd2(a, b, C)

    return float(np.sqrt(w2_sq))

def compute_feature_stats(
    roots: List[Path],
    episode_ids: List[int],
    feature_keys: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global mean/std for concatenated feature dims (without time dim).
    Uses sum/sumsq streaming to avoid storing all frames.
    """
    total_count = 0
    sum_vec = None
    sumsq_vec = None

    for root in roots:
        for eid in episode_ids:
            ep_path = root / f"episode_{eid}.hdf5"
            if not ep_path.exists():
                continue
            with h5py.File(ep_path, "r") as f:
                feats = []
                T_ref = None
                for key in feature_keys:
                    arr = _read_dataset(f, key)
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    if arr.ndim != 2:
                        raise ValueError(
                            f"Feature '{key}' must be 1D/2D for stats (got {arr.shape})."
                        )
                    if T_ref is None:
                        T_ref = arr.shape[0]
                    elif arr.shape[0] != T_ref:
                        raise ValueError(
                            f"Stats: length mismatch in {ep_path.name} for key {key}"
                        )
                    feats.append(arr)
                if T_ref is None or T_ref == 0:
                    continue
                X = np.concatenate(feats, axis=1).astype(np.float64)  # (T, D)

            if sum_vec is None:
                sum_vec = X.sum(axis=0)
                sumsq_vec = (X * X).sum(axis=0)
            else:
                sum_vec += X.sum(axis=0)
                sumsq_vec += (X * X).sum(axis=0)
            total_count += X.shape[0]

    if total_count == 0 or sum_vec is None or sumsq_vec is None:
        raise RuntimeError("No data found to compute feature stats.")

    mean = (sum_vec / total_count).astype(np.float32)
    var = (sumsq_vec / total_count) - (mean.astype(np.float64) ** 2)
    var = np.maximum(var, 1e-6)
    std = np.sqrt(var).astype(np.float32)

    return mean, std

def compute_ot_scores():
    ids_ori = set(list_episode_ids(ROOT_ORI))
    ids_ds = set(list_episode_ids(ROOT_DS))
    common_ids = sorted(ids_ori & ids_ds)
    if len(common_ids) == 0:
        raise RuntimeError(f"No common episode ids between:\n  {ROOT_ORI}\n  {ROOT_DS}")

    # compute normalization stats
    feat_mean = feat_std = None
    if NORMALIZE:
        if STATS_SOURCE == "ori":
            stat_roots = [ROOT_ORI]
        elif STATS_SOURCE == "both":
            stat_roots = [ROOT_ORI, ROOT_DS]
        else:
            raise ValueError("STATS_SOURCE must be 'ori' or 'both'")

        feat_mean, feat_std = compute_feature_stats(
            roots=stat_roots,
            episode_ids=common_ids,
            feature_keys=FEATURE_KEYS,
        )
        print(f"[stats] D={feat_mean.shape[0]}, source={STATS_SOURCE}, normalize=ON")

    records = []
    for eid in common_ids:
        p_ori = ROOT_ORI / f"episode_{eid}.hdf5"
        p_ds  = ROOT_DS  / f"episode_{eid}.hdf5"

        X, len_ori = episode_to_array(
            p_ori,
            FEATURE_KEYS,
            time_scale=TIME_SCALE,
            use_timestamp=USE_TIMESTAMP,
            feature_mean=feat_mean,
            feature_std=feat_std,
        )
        Y, len_ds = episode_to_array(
            p_ds,
            FEATURE_KEYS,
            time_scale=TIME_SCALE,
            use_timestamp=USE_TIMESTAMP,
            feature_mean=feat_mean,
            feature_std=feat_std,
        )

        dist = wasserstein2_distance(
            X, Y,
            reg=SINKHORN_REG,
            use_sinkhorn=USE_SINKHORN,
            num_itermax=SINKHORN_NUMITER,
        )

        records.append({
            "episode_id": int(eid),
            "len_ori": int(len_ori),
            "len_ds": int(len_ds),
            "w2_distance": float(dist),
        })
        print(f"[ep {eid:04d}] len_ori={len_ori:4d}, len_ds={len_ds:4d}, W2={dist:.6f}")

    df = pd.DataFrame.from_records(records).sort_values("w2_distance", ascending=True)

    print("\n=== Top-10 smallest W2 (best aligned) ===")
    print(df.head(10).to_string(index=False))

    # quality labels by quantiles (same logic as your LeRobot version)
    q1 = df["w2_distance"].quantile(0.5)
    q2 = df["w2_distance"].quantile(0.9)

    def quality_label(d: float) -> str:
        if d <= q1:
            return "high"
        elif d <= q2:
            return "medium"
        return "low"

    df["quality"] = df["w2_distance"].apply(quality_label)

    print(f"\nQuantile thresholds: Q1(0.5)={q1:.6f}, Q2(0.9)={q2:.6f}")
    print("Quality counts:\n", df["quality"].value_counts())

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    compute_ot_scores()
