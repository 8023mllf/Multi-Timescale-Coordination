#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot OT alignment between an original episode trajectory and its downsampled (e.g., 2x) version.

Outputs (high-res):
- <out_prefix>_traj.pdf/png : 3-panel figure (ori, ds, overlay+OT arrows)
- <out_prefix>_ori.png      : ORI-only trajectory figure (single panel)
- <out_prefix>_ds.png       : DS-only trajectory figure (single panel)
- <out_prefix>_plan.pdf/png : OT transport plan heatmap (optional)

Dependencies:
pip install pot h5py numpy matplotlib
(no sklearn required; PCA done via numpy SVD)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import ot  # POT


# -------------------------
# Feature construction (mirror your script)
# -------------------------

TIMESTAMP_CANDIDATES = [
    "observations/timestamp",
    "timestamp",
]

def _key_to_h5_path(key: str) -> str:
    if key == "action":
        return "/action"
    if not key.startswith("/"):
        return "/" + key
    return key

def _read_dataset(f: h5py.File, key: str) -> np.ndarray:
    path = _key_to_h5_path(key)
    if path not in f:
        raise KeyError(f"Missing dataset '{path}' in file: {f.filename}")
    arr = f[path][()]
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
    with h5py.File(episode_path, "r") as f:
        feats = []
        T_ref = None

        for key in feature_keys:
            arr = _read_dataset(f, key)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.ndim != 2:
                raise ValueError(f"Feature '{key}' must be 1D or 2D, got {arr.shape}")
            if T_ref is None:
                T_ref = arr.shape[0]
            elif arr.shape[0] != T_ref:
                raise ValueError(
                    f"Length mismatch in {episode_path.name}: previous T={T_ref}, {key} has T={arr.shape[0]}"
                )
            feats.append(arr)

        if T_ref is None:
            return np.zeros((0, 0), dtype=np.float32), 0

        X = np.concatenate(feats, axis=1).astype(np.float32)  # (T, D)

        if feature_mean is not None and feature_std is not None:
            X = (X - feature_mean[None, :]) / (feature_std[None, :] + 1e-12)

        if time_scale is not None and time_scale > 0:
            if use_timestamp:
                t = None
                for cand in TIMESTAMP_CANDIDATES:
                    p = _key_to_h5_path(cand)
                    if p in f:
                        t_raw = np.array(f[p][()], dtype=np.float32).reshape(-1, 1)
                        if t_raw.shape[0] != T_ref:
                            raise ValueError(f"Timestamp length mismatch: {t_raw.shape[0]} vs T={T_ref}")
                        t_min, t_max = float(t_raw.min()), float(t_raw.max())
                        if t_max > t_min:
                            t = (t_raw - t_min) / (t_max - t_min)
                        else:
                            t = np.zeros_like(t_raw, dtype=np.float32)
                        break
                if t is None:
                    t = np.linspace(0.0, 1.0, num=T_ref, dtype=np.float32).reshape(-1, 1)
            else:
                t = np.linspace(0.0, 1.0, num=T_ref, dtype=np.float32).reshape(-1, 1)

            X = np.concatenate([X, t * float(time_scale)], axis=1)

    return X.astype(np.float32), int(T_ref)

def compute_feature_stats(
    roots: List[Path],
    episode_ids: List[int],
    feature_keys: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
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
                        raise ValueError(f"Stats: Feature '{key}' must be 1D/2D, got {arr.shape}")
                    if T_ref is None:
                        T_ref = arr.shape[0]
                    elif arr.shape[0] != T_ref:
                        raise ValueError(f"Stats length mismatch in {ep_path.name} for key {key}")
                    feats.append(arr)
                if T_ref is None or T_ref == 0:
                    continue
                X = np.concatenate(feats, axis=1).astype(np.float64)

            if sum_vec is None:
                sum_vec = X.sum(axis=0)
                sumsq_vec = (X * X).sum(axis=0)
            else:
                sum_vec += X.sum(axis=0)
                sumsq_vec += (X * X).sum(axis=0)
            total_count += X.shape[0]

    if total_count == 0 or sum_vec is None:
        raise RuntimeError("No data found to compute feature stats.")

    mean = (sum_vec / total_count).astype(np.float32)
    var = (sumsq_vec / total_count) - (mean.astype(np.float64) ** 2)
    var = np.maximum(var, 1e-6)
    std = np.sqrt(var).astype(np.float32)
    return mean, std


# -------------------------
# OT + PCA + Plot
# -------------------------

def pca_2d(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z: (N, D)
    returns: Z2 (N,2), mean (D,), components (D,2)
    """
    mean = Z.mean(axis=0, keepdims=True)
    X = Z - mean
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    comps = vt[:2].T  # (D,2)
    Z2 = X @ comps
    return Z2, mean.squeeze(0), comps

def ot_coupling_and_w2(
    X: np.ndarray,
    Y: np.ndarray,
    use_sinkhorn: bool,
    reg: float,
    num_itermax: int,
    normalize_cost_by_max: bool = True,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    returns:
      G: (n,m) coupling
      w2: sqrt( sum_ij G_ij * C_ij )
      C: (n,m) cost matrix used
    """
    n, m = X.shape[0], Y.shape[0]
    a = np.ones((n,), dtype=np.float64) / n
    b = np.ones((m,), dtype=np.float64) / m

    C = ot.dist(X.astype(np.float64), Y.astype(np.float64), metric="euclidean") ** 2
    if normalize_cost_by_max:
        cmax = float(C.max())
        if cmax > 0:
            C = C / (cmax + 1e-12)

    if use_sinkhorn:
        G = ot.sinkhorn(a, b, C, reg=reg, numItermax=num_itermax)
    else:
        G = ot.emd(a, b, C)

    w2_sq = float(np.sum(G * C))
    return G, float(np.sqrt(w2_sq)), C

def top_mass_edges(G: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick top-k edges by transport mass.
    returns arrays (ii, jj, vv)
    """
    flat = G.ravel()
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    ii = idx // G.shape[1]
    jj = idx %  G.shape[1]
    vv = flat[idx]
    return ii, jj, vv

def _compute_shared_limits(X2: np.ndarray, Y2: np.ndarray, pad_ratio: float = 0.05):
    xmin = float(min(X2[:, 0].min(), Y2[:, 0].min()))
    xmax = float(max(X2[:, 0].max(), Y2[:, 0].max()))
    ymin = float(min(X2[:, 1].min(), Y2[:, 1].min()))
    ymax = float(max(X2[:, 1].max(), Y2[:, 1].max()))
    dx = max(xmax - xmin, 1e-9)
    dy = max(ymax - ymin, 1e-9)
    padx = pad_ratio * dx
    pady = pad_ratio * dy
    return (xmin - padx, xmax + padx, ymin - pady, ymax + pady)

def _save_single_traj_png(
    pts2: np.ndarray,
    t: np.ndarray,
    title: str,
    out_png: str,
    limits: tuple,
    dpi: int,
):
    fig = plt.figure(figsize=(5.0, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(pts2[:, 0], pts2[:, 1], linewidth=1.2)
    sc = ax.scatter(pts2[:, 0], pts2[:, 1], c=t, s=12)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Normalized time")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_episode(
    root_ori: Path,
    root_ds: Path,
    episode_id: int,
    feature_keys: List[str],
    time_scale: float,
    use_timestamp: bool,
    normalize: bool,
    stats_source: str,   # "ori" or "both"
    use_sinkhorn: bool,
    reg: float,
    num_itermax: int,
    arrows_k: int,
    out_prefix: str,
    save_plan: bool,
    dpi: int,
):
    p_ori = root_ori / f"episode_{episode_id}.hdf5"
    p_ds  = root_ds  / f"episode_{episode_id}.hdf5"
    if not p_ori.exists():
        raise FileNotFoundError(p_ori)
    if not p_ds.exists():
        raise FileNotFoundError(p_ds)

    feat_mean = feat_std = None
    if normalize:
        if stats_source == "ori":
            stat_roots = [root_ori]
        elif stats_source == "both":
            stat_roots = [root_ori, root_ds]
        else:
            raise ValueError("stats_source must be 'ori' or 'both'")
        feat_mean, feat_std = compute_feature_stats(stat_roots, [episode_id], feature_keys)

    X, len_ori = episode_to_array(
        p_ori, feature_keys,
        time_scale=time_scale,
        use_timestamp=use_timestamp,
        feature_mean=feat_mean, feature_std=feat_std
    )
    Y, len_ds = episode_to_array(
        p_ds, feature_keys,
        time_scale=time_scale,
        use_timestamp=use_timestamp,
        feature_mean=feat_mean, feature_std=feat_std
    )

    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise RuntimeError("Empty trajectory.")

    G, w2, C = ot_coupling_and_w2(
        X, Y,
        use_sinkhorn=use_sinkhorn,
        reg=reg,
        num_itermax=num_itermax,
        normalize_cost_by_max=True,
    )

    # PCA on combined points for a shared 2D embedding
    Z = np.concatenate([X, Y], axis=0)
    Z2, _, _ = pca_2d(Z)
    X2 = Z2[:X.shape[0]]
    Y2 = Z2[X.shape[0]:]

    # Prepare time colors (0..1)
    tx = np.linspace(0, 1, X2.shape[0])
    ty = np.linspace(0, 1, Y2.shape[0])

    # -------- NEW: save separate ORI / DS trajectory PNGs (single-panel) --------
    limits = _compute_shared_limits(X2, Y2, pad_ratio=0.06)
    out_ori_png = f"{out_prefix}_ori.png"
    out_ds_png  = f"{out_prefix}_ds.png"
    _save_single_traj_png(
        pts2=X2, t=tx,
        title=f"ORI (Episode {episode_id}, T={len_ori})",
        out_png=out_ori_png,
        limits=limits, dpi=dpi
    )
    _save_single_traj_png(
        pts2=Y2, t=ty,
        title=f"2x/DS (Episode {episode_id}, T={len_ds})",
        out_png=out_ds_png,
        limits=limits, dpi=dpi
    )
    # --------------------------------------------------------------------------

    # --- 3-panel trajectory figure (kept as-is)
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # ori
    ax1.plot(X2[:, 0], X2[:, 1], linewidth=1.2)
    sc1 = ax1.scatter(X2[:, 0], X2[:, 1], c=tx, s=10)
    ax1.set_title(f"ORI (T={len_ori})")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")

    # ds
    ax2.plot(Y2[:, 0], Y2[:, 1], linewidth=1.2)
    sc2 = ax2.scatter(Y2[:, 0], Y2[:, 1], c=ty, s=10)
    ax2.set_title(f"2x/DS (T={len_ds})")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")

    # overlay + OT arrows
    ax3.plot(X2[:, 0], X2[:, 1], linewidth=1.0, alpha=0.9, label="ORI")
    ax3.plot(Y2[:, 0], Y2[:, 1], linewidth=1.0, alpha=0.9, label="DS")
    ax3.scatter(X2[:, 0], X2[:, 1], s=8, alpha=0.9)
    ax3.scatter(Y2[:, 0], Y2[:, 1], s=8, alpha=0.9)
    ax3.set_title(f"Overlay + OT (W2={w2:.4f})")
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2")

    # Draw only top-k mass edges to avoid clutter
    ii, jj, vv = top_mass_edges(G, arrows_k)
    v_norm = (vv - vv.min()) / (vv.max() - vv.min() + 1e-12)
    for a_i, b_j, tval in zip(ii, jj, v_norm):
        x0, y0 = X2[a_i]
        x1, y1 = Y2[b_j]
        ax3.plot([x0, x1], [y0, y1], linewidth=0.3 + 1.2 * float(tval), alpha=0.25)

    ax3.legend(loc="best", frameon=False)

    # shared colorbar (time)
    cbar = fig.colorbar(sc1, ax=[ax1, ax2, ax3], fraction=0.02, pad=0.02)
    cbar.set_label("Normalized time")

    fig.suptitle(
        f"Episode {episode_id} | features={feature_keys} | time_scale={time_scale} | "
        f"{'Sinkhorn' if use_sinkhorn else 'EMD'}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_traj_pdf = f"{out_prefix}_traj.pdf"
    out_traj_png = f"{out_prefix}_traj.png"
    fig.savefig(out_traj_pdf, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_traj_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # --- optional: plan heatmap
    if save_plan:
        fig2 = plt.figure(figsize=(6, 5))
        ax = fig2.add_subplot(1, 1, 1)
        im = ax.imshow(G, aspect="auto", origin="lower")
        ax.set_title(f"OT transport plan G (Episode {episode_id})")
        ax.set_xlabel("DS index"); ax.set_ylabel("ORI index")
        cb = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Transport mass")
        fig2.tight_layout()

        out_plan_pdf = f"{out_prefix}_plan.pdf"
        out_plan_png = f"{out_prefix}_plan.png"
        fig2.savefig(out_plan_pdf, dpi=dpi, bbox_inches="tight")
        fig2.savefig(out_plan_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig2)

    print(f"[OK] W2={w2:.6f}")
    print(f"Saved: {out_prefix}_ori.png, {out_prefix}_ds.png")
    print(f"Saved: {out_traj_pdf}, {out_traj_png}")
    if save_plan:
        print(f"Saved: {out_prefix}_plan.pdf, {out_prefix}_plan.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-ori", type=str, required=True)
    ap.add_argument("--root-ds",  type=str, required=True)
    ap.add_argument("--episode-id", type=int, required=True)
    ap.add_argument("--feature-keys", nargs="+", default=["observations/qpos", "action"])
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--use-timestamp", action="store_true")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--stats-source", type=str, default="ori", choices=["ori", "both"])
    ap.add_argument("--use-sinkhorn", action="store_true")
    ap.add_argument("--reg", type=float, default=1e-2)
    ap.add_argument("--num-itermax", type=int, default=2000)
    ap.add_argument("--arrows-k", type=int, default=250, help="Top-k OT edges to draw")
    ap.add_argument("--out-prefix", type=str, default="ot_episode")
    ap.add_argument("--save-plan", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    plot_episode(
        root_ori=Path(args.root_ori),
        root_ds=Path(args.root_ds),
        episode_id=args.episode_id,
        feature_keys=args.feature_keys,
        time_scale=args.time_scale,
        use_timestamp=args.use_timestamp,
        normalize=args.normalize,
        stats_source=args.stats_source,
        use_sinkhorn=args.use_sinkhorn,
        reg=args.reg,
        num_itermax=args.num_itermax,
        arrows_k=args.arrows_k,
        out_prefix=args.out_prefix,
        save_plan=args.save_plan,
        dpi=args.dpi,
    )

if __name__ == "__main__":
    main()
