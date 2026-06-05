import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
try:
    from PIL import Image
except Exception:
    Image = None


def list_files(paths: List[str], exts: Tuple[str, ...]) -> List[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in names:
                    if n.lower().endswith(exts):
                        files.append(os.path.join(root, n))
        elif os.path.isfile(p) and p.lower().endswith(exts):
            files.append(p)
    return files


def load_label(fp: str) -> np.ndarray:
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".npy":
        arr = np.load(fp)
        return arr
    if Image is None:
        raise RuntimeError("PIL not available for image reading")
    img = Image.open(fp)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img)


def save_label(src_fp: str, arr: np.ndarray, dst_fp: str):
    os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
    ext = os.path.splitext(dst_fp)[1].lower()
    if ext == ".npy":
        np.save(dst_fp, arr)
        return
    if Image is None:
        raise RuntimeError("PIL not available for image writing")
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img.save(dst_fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="annotation directories or files")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--exts", type=str, default=".png,.tif,.tiff,.bmp,.npy")
    parser.add_argument("--show-per-file", action="store_true")
    parser.add_argument("--fix-inplace", action="store_true", help="replace invalid labels with ignore_index and overwrite")
    parser.add_argument("--output-dir", type=str, default=None, help="save fixed labels to mirror tree under this directory")
    parser.add_argument("--dry-run", action="store_true", help="do not write files, only show what would be fixed")
    args = parser.parse_args()

    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    files = list_files(args.paths, exts)
    if not files:
        print("no files found")
        sys.exit(1)

    total = 0
    bad = 0
    fixed = 0
    global_vals = set()
    bad_files = []

    for fp in files:
        try:
            arr = load_label(fp)
        except Exception as e:
            print(f"read error: {fp}: {e}")
            bad += 1
            continue
        total += 1
        u = np.unique(arr)
        global_vals.update([int(v) for v in u.tolist()])
        invalid_mask = (arr < 0) | ((arr >= args.num_classes) & (arr != args.ignore_index))
        if invalid_mask.any():
            bad += 1
            bad_files.append((fp, np.unique(arr[invalid_mask]).tolist()))
            if args.fix_inplace or args.output_dir:
                arr_fixed = arr.copy()
                arr_fixed[invalid_mask] = args.ignore_index
                dst_fp = fp if args.fix_inplace and not args.output_dir else os.path.join(
                    args.output_dir, os.path.relpath(fp, start=os.path.commonpath([os.path.dirname(fp)] + args.paths))) if args.output_dir else fp
                if not args.dry_run:
                    try:
                        save_label(fp, arr_fixed, dst_fp)
                        fixed += 1
                    except Exception as e:
                        print(f"write error: {dst_fp}: {e}")
        if args.show_per_file:
            print(f"file: {fp}")
            print(f"unique: {u.tolist()}")

    print("summary")
    print(f"checked: {total}")
    print(f"invalid_files: {bad}")
    print(f"fixed_files: {fixed}")
    print(f"global_unique_values: {sorted(global_vals)}")
    if bad_files:
        print("invalid_files_detail")
        for fp, vals in bad_files[:100]:
            print(f"{fp} -> invalid_values: {vals}")


if __name__ == "__main__":
    main()
