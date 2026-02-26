#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量删除 ACT 处理后 HDF5 中多余的相机，只保留指定相机（默认 cam_high）。

适配的数据结构示例：
- action: (T, 14)
+ observations/
  + observations/images/
    - observations/images/cam_high
    - observations/images/cam_left_wrist
    - observations/images/cam_right_wrist
  - observations/qpos
  - observations/left_arm_dim
  - observations/right_arm_dim
"""

import argparse
from pathlib import Path
import shutil
import h5py


def process_file(path: Path, keep_cams, dry_run=False, backup=True, verbose=True):
    """
    处理单个 HDF5 文件：只保留 keep_cams 列表中的相机，其它从 observations/images 下删除。
    """
    if verbose:
        print(f"\n=== File: {path} ===")

    # 先只读，检测相机列表
    with h5py.File(path, "r") as f:
        if "observations" not in f or "images" not in f["observations"]:
            if verbose:
                print("  -> 跳过：未找到 observations/images 结构。")
            return

        img_group = f["observations"]["images"]
        existing = list(img_group.keys())  # 比如: ['cam_high', 'cam_left_wrist', 'cam_right_wrist']

        if verbose:
            print(f"  当前相机: {existing}")

        keep_set = set(keep_cams)
        will_keep = [c for c in existing if c in keep_set]
        will_delete = [c for c in existing if c not in keep_set]

        if not will_keep:
            if verbose:
                print(f"  -> 警告：keep 列表 {keep_cams} 中的相机一个都没在该文件中出现，出于安全考虑跳过。")
            return

        if not will_delete:
            if verbose:
                print("  -> 不需要删除：该文件里只有需要保留的相机。")
            return

        if verbose:
            print(f"  将保留: {will_keep}")
            print(f"  将删除: {will_delete}")

    if dry_run:
        if verbose:
            print("  [dry-run] 只预览，不做任何修改。")
        return

    # 备份一次
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            if verbose:
                print(f"  备份到: {backup_path}")
            shutil.copy2(path, backup_path)
        else:
            if verbose:
                print(f"  已存在备份文件: {backup_path}，不会覆盖。")

    # 真正删除不需要的相机
    with h5py.File(path, "r+") as f:
        img_group = f["observations"]["images"]
        for cam in will_delete:
            if cam in img_group:
                if verbose:
                    print(f"  删除相机: {cam}")
                del img_group[cam]

    if verbose:
        print("  -> 完成。")


def main():
    parser = argparse.ArgumentParser(
        description="删除 ACT 处理后 HDF5 中多余的相机，只保留指定相机（默认 cam_high）。"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="包含 episode_*.hdf5 的目录，比如 sim_beat_block_hammer/demo_clean-50",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="episode_*.hdf5",
        help="匹配 HDF5 文件名的 glob 模式，默认 episode_*.hdf5",
    )
    parser.add_argument(
        "--keep",
        nargs="+",
        default=["cam_high"],
        help="需要保留的相机名列表，默认为 ['cam_high']",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要删除的相机，不实际修改文件",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不自动生成 .bak 备份（默认会备份一份原始文件）",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"错误：{data_dir} 不是有效目录。")
        return

    h5_files = sorted(data_dir.glob(args.pattern))
    if not h5_files:
        print(f"在 {data_dir} 下找不到匹配 {args.pattern} 的 HDF5 文件。")
        return

    print(f"将处理目录：{data_dir}")
    print(f"文件数量：{len(h5_files)}")
    print(f"保留相机：{args.keep}")
    print(f"dry-run: {args.dry_run}, 备份: {not args.no_backup}")

    for p in h5_files:
        process_file(
            p,
            keep_cams=args.keep,
            dry_run=args.dry_run,
            backup=not args.no_backup,
            verbose=True,
        )

    print("\n全部处理完毕。建议先随机检查几个 episode_*.hdf5，确认相机只剩 cam_high。")


if __name__ == "__main__":
    main()
