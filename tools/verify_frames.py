"""Kiểm toàn vẹn mọi frame trong _keep_list.txt: đúng đầu/cuối JPEG (FF D8 ... FF D9) và giải mã được.

  python tools/verify_frames.py --frames_dir data/frames [--fix]
--fix: xóa frame hỏng và dấu _done của video chứa nó, để `extract_frames.py` chạy lại chỉ trích các video đó.
Lý do có công cụ này: tắt máy/ngủ giữa lúc ghi có thể để lại file đúng dung lượng nhưng toàn byte 0.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image


def check(path):
    try:
        with open(path, "rb") as f:
            head = f.read(2)
            f.seek(-2, os.SEEK_END)
            tail = f.read(2)
        if head != b"\xff\xd8" or tail != b"\xff\xd9":
            return path, f"sai marker JPEG (đầu {head.hex()}, cuối {tail.hex()})"
        with Image.open(path) as im:
            im.load()
        return path, None
    except Exception as e:
        return path, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    root = Path(a.frames_dir)
    keep = [l.strip() for l in open(root / "_keep_list.txt", encoding="utf-8") if l.strip()]
    missing = [k for k in keep if not (root / k).exists()]
    with ProcessPoolExecutor(a.workers) as ex:
        bad = [(p, e) for p, e in ex.map(check, [str(root / k) for k in keep if (root / k).exists()], chunksize=256) if e]
    vids = sorted({Path(p).parent.name for p, _ in bad} | {k.split("/")[0] for k in missing})
    res = {"frames": len(keep), "missing": len(missing), "bad": len(bad), "videos_to_redo": vids,
           "examples": [(str(Path(p).relative_to(root)), e) for p, e in bad[:10]]}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.fix:
        for p, _ in bad:
            Path(p).unlink(missing_ok=True)
        for v in vids:
            (root / "_done" / f"{v}.done").unlink(missing_ok=True)
        print(f"Đã xóa {len(bad)} frame hỏng và dấu _done của {len(vids)} video: chạy lại extract_frames.py để trích lại.")
    return 0 if not bad and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
