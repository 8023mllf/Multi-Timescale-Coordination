#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 HDF5 文件中提取图像 / 视频帧并合成为 MP4。

支持常见几种格式：
1) 单段视频 / 单个相机：
   - (T, H, W, C)
   - (H, W, C, T)

2) 多段视频 / 多个 episode：
   - (N, T, H, W, C)
   - (N, H, W, C, T)

如果你的结构不一样，可以先用 --print_tree 看一下 HDF5 结构，再调整 dataset_path。
"""

import argparse
import os

import h5py
import numpy as np
import imageio.v2 as imageio


def print_h5_tree(h5_path):
    """打印 HDF5 文件结构，方便你找到正确的 dataset 路径。"""
    with h5py.File(h5_path, "r") as f:
        def _print(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"[DATASET] {name} shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"[GROUP  ] {name}")
        f.visititems(_print)


def normalize_frames(frames):
    """
    把帧数据变成 uint8 [0,255]，防止有 float32/float64。
    期望输入是 (T, H, W, C)
    """
    if frames.dtype in [np.float32, np.float64]:
        frames = np.clip(frames, 0.0, 1.0) * 255.0
        frames = frames.astype(np.uint8)
    elif frames.dtype != np.uint8:
        # 其他类型简单 clip 一下
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def extract_frames_from_dataset(ds, episode_idx=None):
    """
    根据 dataset shape 自动解析成 (T, H, W, C) 的帧序列。

    支持：
    - (T, H, W, C)
    - (H, W, C, T)
    - (N, T, H, W, C)
    - (N, H, W, C, T)
    """
    shape = ds.shape
    ndim = len(shape)

    if ndim == 4:
        # 可能是 (T, H, W, C) 或 (H, W, C, T)
        if shape[-1] in (1, 3, 4):  # 最后一维是通道
            # (T, H, W, C)
            frames = ds[()]  # 读成 numpy
        elif shape[0] in (1, 3, 4):
            # (C, H, W, T) 这种也有可能，但比较少
            data = ds[()]
            frames = np.moveaxis(data, -1, 0)        # (C,H,W,T) -> (C,H,W,T)
            frames = np.moveaxis(frames, 0, -1)      # 这里先不复杂化，按需要再改
            raise ValueError("遇到不常见的 (C,H,W,T) 格式，请手动改脚本处理。")
        else:
            # 认为是 (H, W, C, T)
            data = ds[()]
            frames = np.moveaxis(data, -1, 0)  # (H,W,C,T) -> (T,H,W,C)

    elif ndim == 5:
        # 可能是 (N, T, H, W, C) 或 (N, H, W, C, T)
        if episode_idx is None:
            raise ValueError(
                f"Dataset shape={shape} 看起来有多个 episode，请指定 --episode_idx"
            )
        if not (0 <= episode_idx < shape[0]):
            raise IndexError(
                f"episode_idx={episode_idx} 超出范围，dataset 第一维大小为 {shape[0]}"
            )

        if shape[-1] in (1, 3, 4):
            # (N, T, H, W, C)
            frames = ds[episode_idx]  # -> (T, H, W, C)
        else:
            # (N, H, W, C, T)
            data = ds[episode_idx]           # (H, W, C, T)
            frames = np.moveaxis(data, -1, 0)  # (T, H, W, C)
    else:
        raise ValueError(
            f"不支持的 dataset 维度: shape={shape}，需要你手动检查数据格式后改脚本。"
        )

    if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"解析后的帧形状不对，得到 shape={frames.shape}，期望 (T,H,W,C)，C=1/3/4"
        )

    return frames


def h5_to_mp4(
    h5_path,
    dataset_path,
    output_mp4,
    episode_idx=None,
    fps=20,
    codec="libx264"
):
    with h5py.File(h5_path, "r") as f:
        if dataset_path not in f:
            print(f"数据集路径 '{dataset_path}' 不在文件中，请确认。可用 --print_tree 查看结构。")
            print("当前文件中包含的顶层键：", list(f.keys()))
            raise KeyError(f"dataset_path '{dataset_path}' not found in {h5_path}")

        ds = f[dataset_path]
        print(f"读取 dataset '{dataset_path}', shape={ds.shape}, dtype={ds.dtype}")

        frames = extract_frames_from_dataset(ds, episode_idx=episode_idx)
        print(f"解析得到帧序列 shape={frames.shape} (T,H,W,C)")

        frames = normalize_frames(frames)

    T, H, W, C = frames.shape
    print(f"合成视频：{T} 帧, 分辨率={W}x{H}, 通道={C}, fps={fps}")

    # 这里修复目录为空的情况
    out_dir = os.path.dirname(output_mp4)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(output_mp4, fps=fps, codec=codec)
    for i in range(T):
        frame = frames[i]
        writer.append_data(frame)
    writer.close()

    print(f"已保存到 {output_mp4}")


def main():
    parser = argparse.ArgumentParser(
        description="从 HDF5 中提取图像/视频帧并合成 MP4"
    )
    parser.add_argument("--h5_path", type=str, required=True,
                        help="HDF5 文件路径")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="HDF5 内部 dataset 路径，比如 'observations/cam_high/frames'")
    parser.add_argument("--episode_idx", type=int, default=None,
                        help="如果 dataset 有多段视频 (N, T, H, W, C)，选第几个 episode")
    parser.add_argument("--fps", type=int, default=20,
                        help="输出 MP4 的帧率")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 mp4 文件路径 (默认与 h5 同名，后缀改为 .mp4)")
    parser.add_argument("--print_tree", action="store_true",
                        help="只打印 HDF5 文件结构，不导出视频")

    args = parser.parse_args()

    if args.print_tree:
        print_h5_tree(args.h5_path)
        return

    if args.dataset_path is None:
        raise ValueError(
            "必须指定 --dataset_path，比如 --dataset_path observations/cam_high/frames\n"
            "你可以先加 --print_tree 看一下 HDF5 结构。"
        )

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.h5_path))[0]
        args.output = f"{base}.mp4"

    h5_to_mp4(
        h5_path=args.h5_path,
        dataset_path=args.dataset_path,
        output_mp4=args.output,
        episode_idx=args.episode_idx,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
