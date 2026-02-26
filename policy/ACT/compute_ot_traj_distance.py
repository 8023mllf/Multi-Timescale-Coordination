#!/usr/bin/env python
"""
Compute OT (Wasserstein) distance per episode between
original and downsampled LeRobot datasets.

- Original dataset root: /kuavo_data_challenge/leju_task2_ori/lerobot
- Downsampled dataset root: /kuavo_data_challenge/leju_task2_2x/lerobot

Usage:
    python compute_ot_traj_distance.py

Make sure you have installed:
    pip install lerobot pot
"""

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import ot  # Python Optimal Transport

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ==== config you may want to change ====

# 这两个是传给 LeRobotDataset 的 root（root 下面有 lerobot 目录）
ROOT_ORI = Path("/kuavo_data_challenge/leju_task2_ori/lerobot")
ROOT_DS  = Path("/kuavo_data_challenge/leju_task2_2x/lerobot")

# repo_id 随便填一个字符串都行，这里保持 lerobot 就好
REPO_ID = "lerobot"

# 用哪些特征定义“轨迹几何”
# 你可以根据自己的 features 改，比如 ["observation.state", "action"]
FEATURE_KEYS: List[str] = ["observation.state"]

# 是否把时间当作额外维度编码进去
# TIME_SCALE 控制“时间维对距离的影响强度”
TIME_SCALE = 1.0  # >0 表示启用时间特征

# 层次二：是否使用数据集里的真实 timestamp 作为时间特征
# True: 用 timestamp（归一化到 [0,1]）
# False: 用简单等间隔 0~1（层次一）
USE_TIMESTAMP = True

# Sinkhorn 熵正则化系数：越大越平滑、越快；越小越接近精确 OT 但更慢
SINKHORN_REG = 1e-2


# ==== OT core ====

def wasserstein2_distance(
    X: np.ndarray,
    Y: np.ndarray,
    reg: float = SINKHORN_REG,
    use_sinkhorn: bool = True,
) -> float:
    """
    Compute 2-Wasserstein distance between two point clouds X, Y using POT.

    X: (N, D)
    Y: (M, D)
    return: scalar distance
    """
    assert X.ndim == 2 and Y.ndim == 2, "X, Y must be 2D arrays"

    n, m = X.shape[0], Y.shape[0]
    if n == 0 or m == 0:
        # 没有数据，返回 NaN，后面可以过滤
        return float("nan")

    # 均匀权重：每一帧等权
    a = np.ones((n,), dtype=np.float64) / n
    b = np.ones((m,), dtype=np.float64) / m

    # 代价矩阵：点到点的平方欧氏距离（对应 2-Wasserstein）
    C = ot.dist(X, Y, metric="euclidean") ** 2
    C /= C.max() + 1e-12  # 归一化一下，避免数值太大

    if use_sinkhorn:
        # 带熵正则的近似 OT 距离，返回的是 W2^2
        w2_sq = ot.sinkhorn2(a, b, C, reg=reg)
    else:
        # 精确 EMD（慢一些），同样返回 W2^2
        w2_sq = ot.emd2(a, b, C)

    # 开根号得到 2-Wasserstein 距离
    return float(np.sqrt(w2_sq))


# ==== Trajectory extraction from LeRobotDataset ====

def load_lerobot_dataset(root: Path) -> LeRobotDataset:
    """
    root 直接是 lerobot 数据集根目录，比如 .../lerobot
    """
    ds = LeRobotDataset(repo_id=REPO_ID, root=str(root))
    return ds

def build_episode_index(hf_ds) -> Dict[int, np.ndarray]:
    """
    把 hf_dataset 里的所有样本按 episode_index 分组。

    返回:
        ep_to_indices: dict[episode_index] -> np.array(数据集行号列表)
    """
    ep_indices = np.array(hf_ds["episode_index"], dtype=np.int64)
    # 有些数据集 episode_index 的 shape 是 (T, 1)，展平成一维更稳
    ep_indices = ep_indices.reshape(-1)

    unique_eps = np.unique(ep_indices)
    ep_to_indices: Dict[int, np.ndarray] = {}
    for ep in unique_eps:
        idx = np.nonzero(ep_indices == ep)[0]
        ep_to_indices[int(ep)] = idx
    return ep_to_indices


def episode_to_array(
    hf_ds,
    indices: np.ndarray,
    feature_keys: List[str],
    time_scale: float | None = None,
    use_timestamp: bool = False,
) -> np.ndarray:
    """
    从一个 episode 的所有 index 中，抽取 FEATURE_KEYS 对应的向量并拼接。
    再根据需要加上时间维度（可选用真实 timestamp，或者等间隔 0~1）。

    indices: 一维 numpy array，里面是这个 episode 的行号。
    """
    if indices.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # 保持时间顺序
    indices_sorted = np.sort(indices)
    sub = hf_ds.select(indices_sorted.tolist())

    # 1. 抽取 feature_keys 指定的特征
    feats = []
    for key in feature_keys:
        if key not in sub.column_names:
            raise KeyError(
                f"Feature key '{key}' 不存在，当前列有: {sub.column_names}"
            )
        arr = np.array(sub[key], dtype=np.float32)
        # HF datasets 对于标量特征返回 (T,)，对于向量特征返回 (T, D)
        if arr.ndim == 1:
            arr = arr[:, None]
        feats.append(arr)

    # 在特征维度上拼接 => (T, D_total)
    X = np.concatenate(feats, axis=1)

    # 2. 可选：加入时间维
    if time_scale is not None and time_scale > 0:
        if use_timestamp:
            # 用真实 timestamp，归一化到 [0, 1]
            if "timestamp" not in sub.column_names:
                raise KeyError(
                    "USE_TIMESTAMP=True 但是数据集中没有 'timestamp' 列，"
                    "请检查数据或把 USE_TIMESTAMP 设为 False 使用层次一时间特征。"
                )
            t_raw = np.array(sub["timestamp"], dtype=np.float32)
            t_raw = t_raw.reshape(-1, 1)  # 兼容 shape [T] / [T,1]
            t_min = float(t_raw.min())
            t_max = float(t_raw.max())
            if t_max > t_min:
                t = (t_raw - t_min) / (t_max - t_min)
            else:
                # 所有 timestamp 一样，就都设成 0
                t = np.zeros_like(t_raw, dtype=np.float32)
        else:
            # 层次一：简单等间隔时间 0 ~ 1
            t = np.linspace(0.0, 1.0, num=X.shape[0], dtype=np.float32).reshape(-1, 1)

        # 时间维乘一个缩放系数，控制它对距离的影响强弱
        X = np.concatenate([X, t * time_scale], axis=1)

    return X


def compute_ot_scores():
    """
    主函数：对所有 episode 计算 (a: 原始, b: 下采样) 的 OT 距离，并排序 + 三档质量标签。
    """
    ds_ori = load_lerobot_dataset(ROOT_ORI)
    ds_ds = load_lerobot_dataset(ROOT_DS)

    hf_ori = ds_ori.hf_dataset
    hf_ds = ds_ds.hf_dataset

    ep_to_idx_ori = build_episode_index(hf_ori)
    ep_to_idx_ds = build_episode_index(hf_ds)

    # episode id 的交集（防止某个版本少了若干 episode）
    common_eps = sorted(set(ep_to_idx_ori.keys()) & set(ep_to_idx_ds.keys()))

    records = []
    for ep in common_eps:
        idx_ori = ep_to_idx_ori[ep]
        idx_ds = ep_to_idx_ds[ep]

        X = episode_to_array(
            hf_ori, idx_ori,
            FEATURE_KEYS,
            time_scale=TIME_SCALE,
            use_timestamp=USE_TIMESTAMP,
        )
        Y = episode_to_array(
            hf_ds, idx_ds,
            FEATURE_KEYS,
            time_scale=TIME_SCALE,
            use_timestamp=USE_TIMESTAMP,
        )

        dist = wasserstein2_distance(X, Y, reg=SINKHORN_REG, use_sinkhorn=True)

        records.append(
            {
                "episode_index": ep,
                "len_ori": int(X.shape[0]),
                "len_ds": int(Y.shape[0]),
                "w2_distance": dist,
            }
        )
        print(
            f"[ep {ep:04d}] len_ori={X.shape[0]:4d}, "
            f"len_ds={Y.shape[0]:4d}, W2={dist:.6f}"
        )

    df = pd.DataFrame.from_records(records).sort_values("w2_distance")
    print("\n=== 按 W2 距离从小到大排序的前 10 条 episode ===")
    print(df.head(10))

    # === 三档质量标签：高 / 中 / 低 ===
    # 例如：
    #   W2 <= Q1(0.5 分位数)           -> "high"  （优）
    #   Q1 < W2 <= Q2(0.9 分位数)      -> "medium"（良）
    #   W2 > Q2                        -> "low"   （差）
    q1 = df["w2_distance"].quantile(0.5)
    q2 = df["w2_distance"].quantile(0.9)

    def quality_label(d: float) -> str:
        if d <= q1:
            return "high"   # 优
        elif d <= q2:
            return "medium" # 良
        else:
            return "low"    # 差

    df["quality"] = df["w2_distance"].apply(quality_label)

    print(
        f"\n三档分位数阈值: Q1(0.5) = {q1:.6f}, Q2(0.9) = {q2:.6f}\n"
        f"质量分布统计:\n{df['quality'].value_counts()}"
    )

    # 保存到 csv 便于后处理 / 可视化
    out_path = Path("ot_scores_per_episode.csv")
    df.to_csv(out_path, index=False)
    print(f"\n已保存所有 episode 的打分到: {out_path.resolve()}")


if __name__ == "__main__":
    compute_ot_scores()
