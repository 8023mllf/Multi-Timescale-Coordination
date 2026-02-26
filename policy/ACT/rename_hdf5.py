#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
import time

def list_hdf5_files(folder: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in folder.rglob("*.hdf5") if p.is_file()]
    else:
        files = [p for p in folder.glob("*.hdf5") if p.is_file()]
    return files

def sort_files(files: list[Path], sort_by: str) -> list[Path]:
    if sort_by == "name":
        # Natural sort to handle episode_2 vs episode_10 correctly
        return sorted(files, key=lambda p: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', p.name)])
    if sort_by == "mtime":
        return sorted(files, key=lambda p: p.stat().st_mtime)
    raise ValueError(f"Unknown sort_by: {sort_by}")

def rename_two_phase(pairs: list[tuple[Path, Path]], apply: bool) -> None:
    # Phase 0: sanity
    dst_set = set(dst for _, dst in pairs)
    if len(dst_set) != len(pairs):
        raise RuntimeError("Target filenames have duplicates. This should not happen.")

    # Build a set of all source files involved in this renaming
    src_set = set(src.resolve() for src, _ in pairs)

    # Check if any target already exists (and is not the same file)
    collisions = []
    for src, dst in pairs:
        if dst.exists():
            # If dst is the same file as src, it's fine
            if dst.resolve() == src.resolve():
                continue
            # If dst is one of the source files we are moving, it's safe (will be moved away)
            if dst.resolve() in src_set:
                continue
            # Otherwise, it's a real collision
            collisions.append((src, dst))

    if collisions:
        msg = "\n".join([f"  src={s.name} -> dst={d.name} (dst exists)" for s, d in collisions])
        raise RuntimeError(
            "Some target filenames already exist and are not part of the rename set. Refusing to overwrite:\n" + msg
        )

    # Phase 1: rename to temporary names to avoid in-place collisions
    ts = int(time.time())
    tmp_pairs = []
    for i, (src, dst) in enumerate(pairs):
        tmp = src.with_name(f".__tmp_rename__{ts}__{i}__.hdf5")
        tmp_pairs.append((src, tmp))

    # Show plan
    print("=== Rename plan ===")
    for (src, tmp), (_, dst) in zip(tmp_pairs, pairs):
        print(f"{src.name} -> {dst.name}")

    if not apply:
        print("\n[DONE] Dry-run only (no changes). Add --apply to execute.")
        return

    # Execute
    for src, tmp in tmp_pairs:
        src.rename(tmp)

    for (_, tmp), (_, dst) in zip(tmp_pairs, pairs):
        tmp.rename(dst)

    print("\n[DONE] Renaming completed.")

def main():
    ap = argparse.ArgumentParser(description="Rename .hdf5 files to episode_{i}.hdf5")
    ap.add_argument("folder", type=str, help="Folder containing .hdf5 files")
    ap.add_argument("--apply", action="store_true", help="Actually perform renaming (default: dry-run)")
    ap.add_argument("--recursive", action="store_true", help="Rename recursively in subfolders")
    ap.add_argument("--sort-by", choices=["name", "mtime"], default="name",
                    help="How to order files before assigning episode indices")
    ap.add_argument("--start", type=int, default=0, help="Start index (default: 0)")
    ap.add_argument("--prefix", type=str, default="episode_", help="Prefix (default: episode_)")
    ap.add_argument("--ext", type=str, default=".hdf5", help="Extension (default: .hdf5)")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found or not a directory: {folder}")

    files = list_hdf5_files(folder, args.recursive)
    if not files:
        raise SystemExit(f"No .hdf5 files found in: {folder}")

    files = sort_files(files, args.sort_by)

    # If recursive, rename within each directory separately (avoid mixing dirs)
    if args.recursive:
        by_dir: dict[Path, list[Path]] = {}
        for p in files:
            by_dir.setdefault(p.parent, []).append(p)

        for d, flist in sorted(by_dir.items(), key=lambda x: str(x[0])):
            flist = sort_files(flist, args.sort_by)
            pairs = []
            for j, src in enumerate(flist):
                idx = args.start + j
                dst = src.with_name(f"{args.prefix}{idx}{args.ext}")
                pairs.append((src, dst))
            print(f"\n### Directory: {d} (count={len(flist)})")
            rename_two_phase(pairs, apply=args.apply)
    else:
        pairs = []
        for j, src in enumerate(files):
            idx = args.start + j
            dst = src.with_name(f"{args.prefix}{idx}{args.ext}")
            pairs.append((src, dst))
        rename_two_phase(pairs, apply=args.apply)

if __name__ == "__main__":
    main()
